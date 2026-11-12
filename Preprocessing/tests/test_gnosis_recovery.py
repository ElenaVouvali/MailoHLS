"""Regression cases for omitted-vector recovery and source-safe supervision."""
import csv
import gzip
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

from Preprocessing import gnosis_recovery as recovery
from experiments.gnosis_stage1_ab.rebuild_non_test import select_complete_csv_kernels


class RecoveryTests(unittest.TestCase):
    def fixture(self, root, *, eligible=True):
        split=root/'mailohls_runs/mailohls_final_family_split_s123.json'
        split.parent.mkdir();split.write_text(json.dumps({'train_kernels':['k'],'val_kernels':[],'test_kernels':[]}))
        raw=root/'raw.csv'
        rows=[dict(Version='f_'+str(i)+'.cpp',Device='fpga',Clock_Period_nsec='5',
                   OuterLoop_1='pipeline',**{c:'1' for c in recovery.QOR}) for i in range(2)]
        with raw.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        source=root/'source.cpp';source.write_text('void f(){L1:for(;;){} L2:for(;;){}}')
        states=[dict(version=row['Version'],device='fpga',clock='5',sqlite_rowid=i+1,
            sqlite_key='[0 '+str(i)+']',states={'L2':['pipeline','unroll_2'][i]},
            qor_sha256=recovery.object_sha({c:row[c] for c in recovery.QOR})) for i,row in enumerate(rows)]
        entry=dict(kernel='k',canonical_eligible=eligible,canonical_files={'raw.csv':recovery.sha(raw),'source.cpp':recovery.sha(source)},
            canonical_measurements='raw.csv',missing_actions=['L2'],source_actions=['L1','L2'],csv_to_action={'OuterLoop_1':'L1'},
            states={'path':'states.jsonl.gz','sha256':''})
        self.write(root,entry,states)
        return entry,states

    def write(self,root,entry,states):
        (root/'states.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r)+'\n' for r in states).encode(),mtime=0))
        entry['states']['sha256']=recovery.sha(root/'states.jsonl.gz')
        receipt=root/recovery.REGISTRY;receipt.parent.mkdir(parents=True,exist_ok=True)
        receipt.write_text(json.dumps({'schema':'salt-hls-gnosis-action-recovery-v1','kernels':[entry]}))

    def test_recover_variable_loop_without_editing_original_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);before=(root/'raw.csv').read_bytes()
            rows=list(recovery.complete_rows('k',root))
            self.assertEqual([r['source_action_states']['L2'] for r in rows],['pipeline','unroll_2'])
            self.assertEqual((root/'raw.csv').read_bytes(),before)
            self.assertNotIn('L2',rows[0]['measurement'])

    def test_missing_or_duplicate_configuration_is_not_complete(self):
        for change in ('missing','duplicate'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);entry,states=self.fixture(root)
                self.write(root,entry,states[:1] if change=='missing' else [states[0],states[0]])
                with self.assertRaisesRegex(ValueError,'every original measured configuration uniquely'):
                    list(recovery.complete_rows('k',root))

    def test_identically_named_arrays_keep_distinct_source_action_states(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);entry,states=self.fixture(root)
            source=root/'source.cpp'
            source.write_text('void f(){L1:for(;;){}} void g(){L20:float buf[4];} void h(){L21:float buf[4];}')
            entry['canonical_files']['source.cpp']=recovery.sha(source)
            entry['missing_actions']=['L20','L21'];entry['source_actions']=['L1','L20','L21']
            for state in states:state['states']={'L20':'block_2_1','L21':'cyclic_4_1'}
            self.write(root,entry,states)
            loaded=list(recovery.complete_rows('k',root))[0]['source_action_states']
            self.assertEqual(loaded['L20'],'block_2_1')
            self.assertEqual(loaded['L21'],'cyclic_4_1')

    def test_missing_action_and_wrong_qor_association_fail_after_hash_validation(self):
        for change in ('state','qor'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);entry,states=self.fixture(root)
                if change=='state':states[0]['states']={}
                else:states[0]['qor_sha256']='wrong'
                self.write(root,entry,states)
                with self.assertRaises(ValueError):list(recovery.complete_rows('k',root))

    def test_changed_source_or_sidecar_is_rejected(self):
        for name in ('source.cpp','states.jsonl.gz'):
            with self.subTest(path=name),tempfile.TemporaryDirectory() as d:
                root=Path(d);self.fixture(root);(root/name).write_bytes(b'changed')
                with self.assertRaises(ValueError):list(recovery.complete_rows('k',root))

    def test_historical_states_for_another_source_do_not_promote_canonical_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,eligible=False)
            with self.assertRaisesRegex(ValueError,'No authenticated complete canonical recovery'):
                list(recovery.complete_rows('k',root))

    def test_locked_test_is_rejected_before_a_measurement_is_opened(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            (root/'mailohls_runs/mailohls_final_family_split_s123.json').write_text(json.dumps({'test_kernels':['k']}))
            (root/'raw.csv').write_text('malformed rows must never be read')
            with self.assertRaisesRegex(ValueError,'Locked-test'):
                list(recovery.complete_rows('k',root))

    def test_invalid_indices_are_rejected_instead_of_clamped(self):
        self.assertEqual(recovery.decode_vector('[0 1]',[['pipeline'],['pipeline','unroll']]),['pipeline','unroll'])
        for key in ('[0 2]','[0]','[-1 0]'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                recovery.decode_vector(key,[['pipeline'],['pipeline','unroll']])

    def test_only_exact_or_label_spacing_equivalence_is_accepted(self):
        self.assertEqual(recovery.source_equivalence(b'L8:   for(;;){}',b'L8: for(;;){}'),'label-spacing-only')
        for old,new in [(b'void f(){}',b'extern "C" void f(){}'),
                        (b'#include <iostream>\n',b'// #include <iostream>\n'),
                        (b'typedef ap_uint<512> W;',b'typedef uint32_t W;')]:
            self.assertEqual(recovery.source_equivalence(old,new),'unproven-source-equivalence')

    def test_pickle_globals_are_not_executed(self):
        with self.assertRaisesRegex(ValueError,'executable pickle global'):
            recovery.MetricUnpickler(io.BytesIO(pickle.dumps(Path('/tmp/no')))).load()

    def test_legacy_bank_does_not_silently_include_sidecar_only_eligibility(self):
        common={'split':'train','complete_source_configuration_eligible':True}
        selected=select_complete_csv_kernels({'kernels':[
            dict(common,kernel='old'),dict(common,kernel='recovered',raw_csv_complete_source_configuration_eligible=False),
            dict(common,kernel='test',split='test')]})
        self.assertEqual(set(selected),{'old'})


if __name__=='__main__':unittest.main()
