"""Measurement provenance and locked-test checks for data consolidation."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Preprocessing import consolidate_data as data


def csv_bytes(rows):
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


class MeasurementIdentityTests(unittest.TestCase):
    def row(self, **extra):
        return dict(Version='v0', Device='fpga', Clock_Period_nsec='5',
                    Latency_msec='1', **extra)

    def test_numeric_formatting_and_recorded_column_recovery(self):
        old = self.row(Array_1='complete_1')
        new = dict(self.row(), Clock_Period_nsec='5.0', Latency_msec='1.0000000000001')
        result = data.compare_rows(csv_bytes([old]), csv_bytes([new]))
        self.assertTrue(result['same_design_keys'])
        self.assertEqual(result['shared_column_difference_counts'], {})

    def test_changed_qor_or_directive_is_not_a_matching_measurement(self):
        old = self.row(Array_1='complete_1')
        new = dict(old, Latency_msec='2', Array_1='cyclic_2_1')
        result = data.compare_rows(csv_bytes([old]), csv_bytes([new]))
        self.assertEqual(result['shared_column_difference_counts'],
                         {'Array_1': 1, 'Latency_msec': 1})

    def test_different_clocks_and_ambiguous_versions_are_rejected(self):
        old = self.row()
        new = dict(old, Clock_Period_nsec='10')
        self.assertFalse(data.compare_rows(csv_bytes([old]), csv_bytes([new]))['same_design_keys'])
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            data.compare_rows(csv_bytes([old, old]), csv_bytes([old]))

    def test_source_equivalence_only_allows_spacing_after_labels(self):
        self.assertEqual(data.label_spacing(b'L9:     for(int i=0;i<4;i++){}'),
                         data.label_spacing(b'L9:  for(int i=0;i<4;i++){}'))
        self.assertNotEqual(data.label_spacing(b'L9: for(int i=0;i<4;i++){}'),
                            data.label_spacing(b'L9: for(int i=0;i<8;i++){}'))


class CanonicalRegistryTests(unittest.TestCase):
    def index(self, cohort, snapshot=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Path('Data/ApplicationDataset/kernel/kernel.c')
            (root/source).parent.mkdir(parents=True)
            (root/source).write_text('void f(){L1: for(;;){} L2: for(;;){}}')
            (root/source.parent/'kernel_info.txt').write_text('f\nL1,loop,4\n')
            apl = root/'Data/ApplicationAPLMapping/kernel.txt'
            apl.parent.mkdir()
            apl.write_text('OuterLoop_1,L1\n')
            raw = root/'Data/CSVS/kernel.csv'
            raw.parent.mkdir()
            row = {'Version': 'v0', 'Device': 'fpga', 'Clock_Period_nsec': '5',
                   'Latency_msec': '1', 'OuterLoop_1': 'pipeline',
                   **{c: '125' for c in data.RESOURCES}}
            raw.write_bytes(csv_bytes([row, dict(row, Version='v1', Latency_msec='nan')]))
            if snapshot:
                (root/'Data/manifest.json').write_text(json.dumps({'kernels': [
                    {'kernel': 'kernel', 'historical_snapshots': [snapshot]}]}))
            if cohort == 'test':
                # If any measurement row is inspected this malformed row must fail.
                raw.write_text(raw.read_text() + 'malformed,row\n')
            split = {n+'_kernels': ['kernel'] if n == cohort else [] for n in ('train', 'val', 'test')}
            with patch.object(data, 'ROOT', root), patch.object(data, 'applications', return_value={'kernel': source}), \
                    patch.object(data, 'split_identity', return_value=(split, {})), \
                    patch.object(data, 'graph_provenance', return_value={}):
                return data.canonical_index()

    def test_unknown_source_action_is_not_imputed_or_silently_eligible(self):
        entry = self.index('train')['kernels'][0]
        self.assertEqual(entry['unproven_source_actions'], ['L2'])
        self.assertEqual(entry['source_labels_absent_metadata'], ['L2'])
        self.assertFalse(entry['complete_source_configuration_eligible'])
        self.assertNotIn('L2', entry['csv_to_action'].values())

    def test_overflow_is_retained_and_unknown_measurement_is_separate(self):
        counts = self.index('train')['kernels'][0]['measurement_counts']
        self.assertEqual(counts['raw_rows'], 2)
        self.assertEqual(counts['finite_positive_latency_and_known_resources'], 1)
        self.assertEqual(counts['measured_resource_at_or_above_capacity_rows'], 1)
        self.assertEqual(counts['invalid_or_unknown_measurement_rows'], 1)

    def test_locked_test_only_reads_headers_and_hashes(self):
        entry = self.index('test')['kernels'][0]
        self.assertEqual(entry['measurement_counts'], {})
        self.assertFalse(entry['test_rows_parsed'])

    def test_optional_frozen_snapshot_reference_survives_without_local_copy(self):
        snapshot = {'path': 'LLM_branch/Data/preprocessed_CSVS/preprocessed-kernel.csv',
                    'sha256': '0'*64, 'role': 'historical_frozen_derived_snapshot',
                    'mapped_columns_absent': []}
        entry = self.index('train', snapshot)['kernels'][0]
        self.assertEqual(entry['historical_snapshots'], [snapshot])


if __name__ == '__main__':
    unittest.main()
