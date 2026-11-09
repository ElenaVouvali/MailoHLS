import pytest
from experiments.final_architecture_20261109.prepare_checkpoint_synthesis import evaluated_request

def test_selection_uses_line_position_not_upstream_metadata_index():
 row={'_jsonl_idx':999,'_quality_split':'validation','kernel_name':'k','device':'d','resource_budget_id':'half','selected_clock_period':10}
 selected={'jsonl_idx':0,'kernel_name':'k','device':'d','resource_budget_id':'half','selected_clock_period':10}
 assert evaluated_request({0:row},selected) is row
 with pytest.raises(ValueError):evaluated_request({0:row},{**selected,'resource_budget_id':'full'})
 with pytest.raises(ValueError):evaluated_request({0:{**row,'_quality_split':'train'}},selected)
