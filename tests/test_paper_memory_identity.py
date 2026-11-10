"""Regression checks against accidental reuse of same-width fusion memory."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import torch
from LLM_branch.common.structural_memory import load_memory_bank


class MemoryIdentityTests(unittest.TestCase):
    def test_new_representation_requires_explicit_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema='salt-paper-rc3-transformer4-jkmax-conv1-jk-v1'
            pack={'schema':'salt-paper-action-memory-v1','representation_schema':schema,
                'node_embs':torch.zeros(64,128),'node_embs_mask':torch.tensor([True]+[False]*63),
                'labels':torch.tensor([1]+[-1]*63),'slot_ids':torch.arange(1,65),'gnn_dim':128,
                'disable_pragma_injection':True}
            normalization=b'{}\n'
            (Path(tmp)/'normalization.json').write_bytes(normalization)
            pack.update(gnn_checkpoint_sha256='1'*64,feature_schema_sha256='2'*64,
                        normalization_sha256=hashlib.sha256(normalization).hexdigest())
            torch.save(pack,Path(tmp)/'kernel.memory.pt')
            manifest={k:pack[k] for k in ('representation_schema','gnn_checkpoint_sha256','feature_schema_sha256','normalization_sha256')}
            manifest['files']={'kernel.memory.pt':hashlib.sha256((Path(tmp)/'kernel.memory.pt').read_bytes()).hexdigest()}
            (Path(tmp)/'memory_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'explicit representation identity'):
                load_memory_bank(tmp,expected_mem_dim=128)
            with self.assertRaisesRegex(ValueError,'incompatible structural representation'):
                load_memory_bank(tmp,expected_mem_dim=128,expected_representation_schema='historical-v4b')
            bank,dim=load_memory_bank(tmp,expected_mem_dim=128,require_pragma_free_memory=True,
                expected_representation_schema=schema)
            self.assertEqual(dim,128)
            self.assertEqual(bank['kernel']['mask'].nonzero().flatten().tolist(),[0])
            (Path(tmp)/'normalization.json').write_bytes(b'changed normalization')
            with self.assertRaisesRegex(ValueError,'normalization hash'):
                load_memory_bank(tmp,expected_representation_schema=schema)
            (Path(tmp)/'normalization.json').write_bytes(normalization)
            pack['feature_schema_sha256']='3'*64
            torch.save(pack,Path(tmp)/'kernel.memory.pt')
            with self.assertRaisesRegex(ValueError,'mixed feature_schema_sha256'):
                load_memory_bank(tmp,expected_representation_schema=schema)


if __name__=='__main__':unittest.main()
