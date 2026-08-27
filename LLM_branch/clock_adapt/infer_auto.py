"""Deployment boundary for AUTO: select a clock, then call specified decode once."""
import argparse, json, torch
from .model import ClockSelector
from .extract_features import pooled_structural_memory
from LLM_branch.common.mailohls_contract import supported_clock_periods, DEVICE_RESOURCES

def select_clock(selector, memory_pack, device, fractions):
    memory=memory_pack['node_embs'].float(); mask=memory_pack['node_embs_mask'].bool(); caps=DEVICE_RESOURCES[device]
    rows=[]
    for clock in supported_clock_periods(device):
        ctx=torch.tensor(list(fractions)+[__import__('math').log1p(caps[k]) for k in ('BRAM_18K','DSP','FF','LUT')]+[__import__('math').log2(clock/5)],dtype=torch.float32)
        rows.append(ctx)
    context=torch.stack(rows); selector.eval()
    with torch.inference_mode(): logits=selector(memory,mask,context)
    return float(supported_clock_periods(device)[int(logits.argmax())]), logits

def auto_to_specified_request(base_request, selector, memory_pack):
    request=dict(base_request)
    if str(request.get('frequency_mode','specified')).lower() != 'auto':
        return request
    budget=request['resource_budget']; caps=DEVICE_RESOURCES[request['device']]
    fractions=[budget['BRAM_18K']/caps['BRAM_18K'],budget['DSP']/caps['DSP'],budget['FF']/caps['FF'],budget['LUT']/caps['LUT']]
    c,_=select_clock(selector,memory_pack,request['device'],fractions)
    request.update({'selected_clock_period':c,'selected_clock_period_ns':c,'frequency_mode':'specified'})
    return request

def main():
    p=argparse.ArgumentParser();p.add_argument('--selector',required=True);p.add_argument('--memory_pack',required=True);p.add_argument('--device',required=True);p.add_argument('--budget_fractions',required=True);a=p.parse_args(); ck=torch.load(a.selector,map_location='cpu',weights_only=False); m=ClockSelector(ck['mem_dim'],ck['context_dim'],ck['hidden_dim'],ck['dropout']);m.load_state_dict(ck['model']); pack=torch.load(a.memory_pack,map_location='cpu',weights_only=False); fr=[float(x) for x in a.budget_fractions.split(',')]; c,_=select_clock(m,pack,a.device,fr); print(json.dumps({'selected_clock_period':c,'selected_clock_period_ns':c,'frequency_mode':'specified'}))
if __name__=='__main__':main()
