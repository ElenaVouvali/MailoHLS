"""Focused parser checks using real report fixtures; no HLS invocation."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from parse_csynth import parse_csynth

HERE = Path(__file__).resolve().parent
PART = 'xczu7ev-ffvc1156-2-e'


class ReportTests(unittest.TestCase):
    def test_all_nine_real_top_reports(self):
        reports = sorted((HERE / 'evidence').rglob('*_csynth.xml'))
        self.assertEqual(len(reports), 9)
        undefined = set()
        for path in reports:
            root = ET.parse(str(path)).getroot()
            top = root.findtext('UserAssignments/TopModelName')
            result = parse_csynth(path, top, PART, 10)
            self.assertTrue(result['resource_metrics_complete'])
            for name in ('BRAM_18K', 'DSP', 'FF', 'LUT'):
                self.assertAlmostEqual(result['resources_pct'][name],
                                       100.0 * result['resources_used'][name] / result['resources_available'][name])
            if result['latency_cycles'] is None:
                undefined.add(top)
        self.assertEqual(undefined, {'chstone_aes', 'chstone_jpeg', 'rendering'})

    def test_zero_dsp_is_real_zero_and_wrong_identity_is_rejected(self):
        path = HERE / 'evidence/project_aes/solution1/syn/report/chstone_aes_csynth.xml'
        result = parse_csynth(path, 'chstone_aes', PART, 10)
        self.assertEqual(result['resources_pct']['DSP'], 0.0)
        for top, part, clock in [('wrong_top', PART, 10), ('chstone_aes', 'xcu50-fsvh2104-2-e', 10),
                                 ('chstone_aes', PART, 5)]:
            with self.assertRaises(ValueError):
                parse_csynth(path, top, part, clock)

    def test_absent_or_zero_capacity_is_unknown_not_zero_percent(self):
        original = HERE / 'evidence/project_aes/solution1/syn/report/chstone_aes_csynth.xml'
        for missing in (True, False):
            root = ET.parse(str(original)).getroot()
            available = root.find('AreaEstimates/AvailableResources')
            dsp = available.find('DSP')
            if missing:
                available.remove(dsp)
            else:
                dsp.text = '0'
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'fixture.xml'
                ET.ElementTree(root).write(str(path))
                result = parse_csynth(path, 'chstone_aes', PART, 10)
                self.assertIsNone(result['resources_pct']['DSP'])
                self.assertFalse(result['resource_metrics_complete'])


if __name__ == '__main__':
    unittest.main()
