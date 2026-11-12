"""Exercise the maintained preprocessor with the canonical data policy."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from Preprocessing import data_preprocess as prep


class CanonicalPreprocessingTests(unittest.TestCase):
    def table(self, extra=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = root/'apps/kernel'
            app.mkdir(parents=True)
            (app/'kernel_info.txt').write_text('f\nL1,loop,4\nL2,array,buf,1,4\n')
            (app/'kernel.c').write_text('void f(){L1: for(;;){} L2: int buf[4]; L3: for(;;){}}')
            apl = root/'apl'
            apl.mkdir()
            (apl/'kernel.txt').write_text('OuterLoop_1,L1\nArray_1,L2\n')
            path = root/'kernel.csv'
            row = {'Version': 'v0', 'Device': 'fpga', 'Clock_Period_nsec': 5,
                   'Latency_msec': 1., 'OuterLoop_1': 'pipeline', 'Array_1': 'NDIR',
                   **{c: 125. for c in prep.UTILIZATION_COLUMNS}, **(extra or {})}
            pd.DataFrame([row]).to_csv(path, index=False)
            with patch.object(prep, 'APPLICATION_DIR', root/'apps'), patch.object(prep, 'APL_DIR', apl):
                return prep.preprocess_kernel(path, 'gnn', '', 10., .1, 2., all_targets=True,
                                              retain_measured_overflow=True, preserve_measured_actions=True)

    def test_recorded_ndir_and_over_capacity_measurements_are_retained(self):
        frame, stats = self.table()
        self.assertIn('Array_1', frame)
        self.assertEqual(frame['Array_1'].tolist(), [''])
        self.assertEqual(frame['LUT_Utilization_percentage'].tolist(), [125.])
        self.assertEqual(stats['active_actions'], 1)
        self.assertEqual(stats['measured_actions'], 2)
        self.assertNotIn('L3', frame)

    def test_unmapped_active_directive_still_fails(self):
        with self.assertRaisesRegex(ValueError, 'absent from its complete APL'):
            self.table({'OuterLoop_2': 'pipeline'})


if __name__ == '__main__':
    unittest.main()
