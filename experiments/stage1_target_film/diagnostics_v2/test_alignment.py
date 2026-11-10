from pathlib import Path
import pytest
from transformers import AutoTokenizer
from .alignment import value_token_positions,causal_kinds

@pytest.fixture(scope='module')
def tok():return AutoTokenizer.from_pretrained('/home/elvouvali/mailohls_runs/stage1_quality_sft_latency_v4/multi_reference/run',local_files_only=True,trust_remote_code=True)

@pytest.mark.parametrize('kind,value',[(1,'0'),(1,'1'),(2,'16'),(2,'1024'),(3,'block'),(3,'cyclic'),(3,'complete'),(4,'2'),(4,'1024'),(5,'1'),(5,'0')])
def test_real_value_delimiter(tok,kind,value):
 text=value+'\n';ids,pos=value_token_positions(tok,text);mask=causal_kinds(tok,text,7,kind,'cpu');assert mask[0,:6].sum()==0
 assert [j for j in range(len(ids)) if mask[0,6+j]]==pos
 assert all(mask[0,6+j]==0 for j in range(len(ids)) if j not in pos)
 assert tok.decode([ids[j] for j in pos])==value
