import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import pandas as pd
from Preprocessing.data_preprocess import preprocess_kernel,ActionDefinition,UTILIZATION_COLUMNS

class Recovery(unittest.TestCase):
    def test_authentic_overflow_zero_and_unknown_outcomes(self):
        rows=[]
        for version,latency,synthesis,util in [('zero',2.,3.,0.),('overflow',4.,3.,150.),('failure',-1.,3.,50.),('ambiguous',1e6,3.,150.),('unknown_synthesis',5.,-1.,150.)]:
            rows.append(dict(Version=version,Device='device',Clock_Period_nsec=10.,Latency_msec=latency,Synthesis_Time_sec=synthesis,OuterLoop_1='unroll_2' if version=='overflow' else 'pipeline_1',**{name:util for name in UTILIZATION_COLUMNS}))
        with tempfile.TemporaryDirectory() as tmp,patch('Preprocessing.data_preprocess.load_action_definitions',return_value={'OuterLoop_1':ActionDefinition('L1','loop',trip_count=4)}):
            path=Path(tmp)/'fixture.csv';pd.DataFrame(rows).to_csv(path,index=False)
            corrected,stats=preprocess_kernel(path,'llm','device',10.,.1,2.,retain_measured_overflow=True)
            self.assertEqual(set(corrected.Version),{'zero','overflow'})
            self.assertEqual(float(corrected[corrected.Version=='overflow'].Latency_msec.iloc[0]),4.)
            self.assertEqual(float(corrected[corrected.Version=='zero'][UTILIZATION_COLUMNS[0]].iloc[0]),0.)
            self.assertEqual(stats['recovered_overcapacity_rows'],1)
            self.assertEqual(sum(stats['rejected_by_reason'].values()),3)

if __name__=='__main__':unittest.main()
