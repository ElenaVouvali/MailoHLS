"""Source-version boundaries and the exact permitted compiler normalizations."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from Preprocessing import gnosis_recovery as old
from Preprocessing import gnosis_compilation_recovery as phase


class CompilerCorrespondenceTests(unittest.TestCase):
    def test_linkage_normalization_does_not_erase_types_operations_or_hls_annotations(self):
        before = '; ModuleID = a\nsource_filename = "a.cpp"\ndefine void @_Z1fPi(i32* %x) {\n %v = load i32, i32* %x\n}\n'
        after = before.replace('_Z1fPi','f').replace('a.cpp','b.cpp')
        self.assertEqual(phase.normalize_ir(before,('_Z1fPi','f')),phase.normalize_ir(after))
        for changed in [after.replace('i32','i512'),after.replace('load','store'),
                        after+'!hls = !{i32 1}\n']:
            with self.subTest(ir=changed):
                self.assertNotEqual(phase.normalize_ir(before,('_Z1fPi','f')),phase.normalize_ir(changed))

    def test_all_four_correspondences_recompute_from_sealed_compiler_output(self):
        for kernel in phase.DILATE | set(phase.SYMBOLS):
            with self.subTest(kernel=kernel):
                proof = phase.validate(kernel)
                self.assertTrue(proof['canonical_eligible'])
                row = next(old.complete_rows(kernel))
                self.assertEqual(row['contract'],phase.CONTRACT)
                self.assertEqual(set(row['source_action_states']),set(old.load_registry()[kernel]['source_actions']))

    def test_knn_canonical_rejected_while_original_source_has_all_13_measured_actions(self):
        with self.assertRaisesRegex(ValueError,'No authenticated complete canonical recovery'):
            next(old.complete_rows(phase.KNN))
        rows = list(phase.complete_original_rows(phase.KNN))
        self.assertEqual(len(rows),5676)
        self.assertTrue(all(len(row['source_action_states']) == 13 for row in rows))
        self.assertTrue(all(row['contract'] == phase.ORIGINAL_CONTRACT for row in rows))
        self.assertTrue(all('GNWSIS-8888db17/' in row['source_version'] for row in rows))
        self.assertNotIn('L12',rows[0]['measurement'])

    def test_canonical_drift_and_modified_ir_fail_even_when_new_fixture_hashes_match(self):
        # Only temporary fixtures are altered; historical receipts are immutable.
        kernel = 'spcl_example_03'
        receipt = json.loads((old.ROOT/phase.REGISTRY).read_text())
        for mode in ['source','ir','parent']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root = Path(d)
                obj = copy.deepcopy(receipt)
                entry = obj['kernels'][kernel]
                paths = list(obj['shared_files'])+list(entry['files'])+[str(old.REGISTRY)]
                for name in paths:
                    target = root/name;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(old.ROOT/name,target)
                meta = root/'Data/ApplicationDataset'/kernel/'kernel_info.txt'
                meta.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(old.ROOT/meta.relative_to(root),meta)
                if mode == 'source':
                    source = root/entry['original_bundle']/'Example3.cpp'
                    source.write_text(source.read_text().replace('0.3333','0.7'))
                    entry['files'][str(source.relative_to(root))] = old.sha(source)
                elif mode == 'ir':
                    path = root/entry['evidence']['canonical']['llvm']
                    text = phase.read_text(root,entry['evidence']['canonical']['llvm']).replace('fmul float','fadd float')
                    path.write_bytes(gzip.compress(text.encode(),mtime=0))
                    entry['files'][str(path.relative_to(root))] = old.sha(path)
                else:
                    with (root/old.REGISTRY).open('a') as f:f.write('\n')
                target = root/phase.REGISTRY;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(json.dumps(obj))
                with self.assertRaises(ValueError):phase.validate(kernel,root)


if __name__ == '__main__':unittest.main()
