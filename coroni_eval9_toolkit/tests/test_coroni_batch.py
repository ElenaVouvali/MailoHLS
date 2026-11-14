import importlib.util
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

MODULE = pathlib.Path(__file__).resolve().parents[1] / 'coroni_batch.py'
spec = importlib.util.spec_from_file_location('coroni_batch', str(MODULE))
salt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(salt)

class TestBatch(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=pathlib.Path(self.temp.name)
        self.sources=self.root/'sources'/'trmm-opt'
        self.sources.mkdir(parents=True)
        (self.sources/'source.cpp').write_text('void kernel_trmm(){\n  /*L1:*/ for(int i=0;i<3;i++){}\n  /*L2:*/ for(int j=0;j<4;j++){}\n  /*L3:*/ for(int k=0;k<5;k++){}\n}\n')
        (self.sources/'kernel_info.txt').write_text('kernel_trmm\nL1,loop,3\nL2,loop,4\nL3,loop,5\n')
        self.pred=self.root/'out.jsonl'
        self.pred.write_text(json.dumps({'kernel':'trmm-opt','checkpoint_step':300,'directives':[
            {'action_id':'L1','kind':'loop','pipeline_ii':0,'unroll_factor':2},
            {'action_id':'L2','kind':'loop','pipeline_ii':1,'unroll_factor':0},
            {'action_id':'L3','kind':'loop','pipeline_ii':0,'unroll_factor':0}]})+'\n')
    def tearDown(self):self.temp.cleanup()
    def args(self):return SimpleNamespace(predictions=str(self.pred),sources_root=str(self.root/'sources'),out=str(self.root/'bundle'),device='zcu104',clock_ns=10.,budget_percent=100.,targets_json=None,require_nine=False)
    def test_materialization(self):
        top,actions=salt.read_metadata(self.sources/'kernel_info.txt')
        s=salt.source_template(self.sources/'source.cpp',actions)
        self.assertEqual(s.count('auto{'),6)
        row=salt.read_jsonl(self.pred)[0]
        text=salt.materialize(s,salt.get_assignments(row,actions))
        self.assertIn('#pragma HLS unroll factor=2',text)
        self.assertIn('#pragma HLS pipeline II=1',text)
        self.assertNotIn('factor=0',text)
        self.assertNotIn('auto{',text.lower())
    def test_prepare_parse_and_undefined(self):
        salt.prepare(self.args())
        b=self.root/'bundle';m=salt.verify_bundle(b)
        self.assertEqual(len(m['jobs']),1)
        j=m['jobs'][0];self.assertEqual(j['top'],'kernel_trmm')
        tcl=(b/'jobs'/j['job_id']/'run_hls.tcl').read_text()
        self.assertIn('xczu7ev-ffvc1156-2-e',tcl)
        report=self.root/'csynth.xml'
        tmpl='''<profile><ReportVersion><Version>2021.1</Version></ReportVersion><UserAssignments><Part>{part}</Part><TopModelName>{top}</TopModelName><TargetClockPeriod>10</TargetClockPeriod><FlowTarget>vivado</FlowTarget><unit>ns</unit></UserAssignments><PerformanceEstimates><SummaryOfOverallLatency><unit>clock cycles</unit><Worst-caseLatency>{lat}</Worst-caseLatency></SummaryOfOverallLatency><SummaryOfTimingAnalysis><EstimatedClockPeriod>8.2</EstimatedClockPeriod></SummaryOfTimingAnalysis></PerformanceEstimates><AreaEstimates><Resources><BRAM_18K>2</BRAM_18K><DSP>3</DSP><FF>4</FF><LUT>5</LUT></Resources><AvailableResources><BRAM_18K>624</BRAM_18K><DSP>1728</DSP><FF>460800</FF><LUT>230400</LUT></AvailableResources></AreaEstimates></profile>'''
        report.write_text(tmpl.format(part=j['part'],top=j['top'],lat='undef'))
        result=salt.parse_report(report,j)
        self.assertIsNone(result['latency_cycles'])
        self.assertGreater(result['percentages']['BRAM_18K'],0.)
        report.write_text(tmpl.format(part='wrong-part',top=j['top'],lat='123'))
        with self.assertRaises(ValueError):salt.parse_report(report,j)
    def test_reject_missing_action(self):
        top,actions=salt.read_metadata(self.sources/'kernel_info.txt')
        row=salt.read_jsonl(self.pred)[0]
        row['directives'].pop()
        with self.assertRaises(ValueError):salt.get_assignments(row,actions)
    def test_reject_broken_source(self):
        (self.sources/'source.cpp').write_text('void kernel_trmm(){}')
        with self.assertRaises(ValueError):salt.prepare(self.args())

if __name__=='__main__':unittest.main()

class TestArrayMaterialization(unittest.TestCase):
    def test_real_pragma_shapes(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.cpp'
            p.write_text('void render(){\n  /*L1:*/ static unsigned char z_buffer[256][256];\n}\n')
            actions={'L1':{'kind':'array','variable':'z_buffer','dims':{1:256,2:256}}}
            templ=salt.source_template(p,actions)
            self.assertIn('variable=z_buffer',templ)
            row={'directives':[{'action_id':'L1','kind':'array','variable':'z_buffer','partition_type':'complete','partition_factor':0,'partition_dim':2}]}
            vals=salt.get_assignments(row,actions)
            materialized=salt.materialize(templ,vals)
            self.assertIn('type=complete',materialized)
            self.assertNotIn('factor=0',materialized)
