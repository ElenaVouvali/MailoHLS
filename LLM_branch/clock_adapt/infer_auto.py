"""Deployment boundary for AUTO: select a clock, then call specified decode once."""
import argparse, json, torch
from .model import ClockSelector, AutoClockOverrideSelector, AutoClockOverrideSelectorV5, AutoClockOverrideSelectorV6, AutoClockOverrideSelectorV7
from .extract_features import pooled_structural_memory
from LLM_branch.common.mailohls_contract import supported_clock_periods, DEVICE_RESOURCES

def select_clock(selector, memory_pack, device, fractions, objective="PARETO_ADP", switch_threshold=0.05):
    memory=memory_pack['node_embs'].float(); mask=memory_pack['node_embs_mask'].bool(); caps=DEVICE_RESOURCES[device]
    rows=[]
    for clock in supported_clock_periods(device):
        onehot=[float(objective.upper()==name) for name in ('PARETO_LATENCY','PARETO_AREA','PARETO_ADP')]
        ctx=torch.tensor(list(fractions)+[__import__('math').log1p(caps[k]) for k in ('BRAM_18K','DSP','FF','LUT')]+[__import__('math').log2(clock/5)]+onehot,dtype=torch.float32)
        rows.append(ctx)
    context=torch.stack(rows); selector.eval()
    fast_idx=int(torch.tensor(supported_clock_periods(device)).argmin())
    with torch.inference_mode():
        raw=selector(memory,mask,context)
        if isinstance(raw, tuple):
            override_logits, clock_logits = raw[:2]
            if len(raw) == 3:
                pfeas = torch.sigmoid(raw[2])
                clock_logits = clock_logits + 2.0 * torch.log(pfeas.clamp_min(1e-6))
            clocks = torch.tensor(supported_clock_periods(device), device=clock_logits.device)
            slow_logits = clock_logits.masked_fill(
                ~(clocks > clocks[fast_idx]), float('-inf')
            )
            candidate = int(slow_logits.argmax())
            selected = (candidate if float(torch.sigmoid(override_logits)) >= float(switch_threshold) else fast_idx)
            predicted = clock_logits - clock_logits[fast_idx]
        else:
            predicted=raw-raw[fast_idx]
    if getattr(selector, "force_fastest", False):
        return float(supported_clock_periods(device)[fast_idx]), predicted
    if isinstance(raw, tuple):
        return float(supported_clock_periods(device)[selected]), predicted
    candidate=int(predicted.argmin())
    selected=candidate if float(predicted[candidate]) < -float(switch_threshold) else fast_idx
    return float(supported_clock_periods(device)[selected]), predicted

def auto_to_specified_request(base_request, selector, memory_pack):
    request=dict(base_request)
    if str(request.get('frequency_mode','specified')).lower() != 'auto':
        return request
    budget=request['resource_budget']; caps=DEVICE_RESOURCES[request['device']]
    if {"BRAM_18K", "DSP", "FF", "LUT"} <= set(budget):
        fractions=[budget['BRAM_18K']/caps['BRAM_18K'],budget['DSP']/caps['DSP'],budget['FF']/caps['FF'],budget['LUT']/caps['LUT']]
    elif {"bram", "dsp", "ff", "lut"} <= set(budget):
        fractions=[float(budget[k]) for k in ('bram','dsp','ff','lut')]
    else:
        raise ValueError("Unsupported AUTO resource_budget schema")
    switch_threshold=float(request.get('switch_threshold',getattr(selector,'switch_threshold',0.05)))
    try:
        c,_=select_clock(selector,memory_pack,request['device'],fractions,
                         request.get('objective','PARETO_ADP'), switch_threshold)
    except TypeError:  # compatibility with old test/adapters exposing 4 args
        c,_=select_clock(selector,memory_pack,request['device'],fractions)
    request.update({'clock_period':c,'selected_clock_period':c,'selected_clock_period_ns':c,'frequency_mode':'specified'})
    return request

def auto_select_then_decode(base_request, selector, memory_pack, build_prompt, constrained_decode):
    """Convert AUTO, then use the exact specified-clock Stage-2 path once."""
    specified_request=auto_to_specified_request(base_request,selector,memory_pack)
    prompt=build_prompt(specified_request)
    decoded=constrained_decode(prompt,specified_request)
    return specified_request,prompt,decoded

def main():
    p=argparse.ArgumentParser();p.add_argument('--selector',required=True);p.add_argument('--memory_pack',required=True);p.add_argument('--device',required=True);p.add_argument('--budget_fractions',required=True);a=p.parse_args(); ck=torch.load(a.selector,map_location='cpu',weights_only=False); cls=AutoClockOverrideSelectorV7 if ck.get('architecture') == 'override_v7' else AutoClockOverrideSelectorV6 if ck.get('architecture') == 'override_v6' else AutoClockOverrideSelectorV5 if ck.get('architecture') == 'override_v5' else AutoClockOverrideSelector if ck.get('architecture') == 'override_v4' else ClockSelector; m=(cls(ck['mem_dim'],ck['context_dim'],ck['hidden_dim'],ck.get('n_clocks',len(ck['clock_menu'])),ck['dropout']) if cls in (AutoClockOverrideSelector,AutoClockOverrideSelectorV5,AutoClockOverrideSelectorV6,AutoClockOverrideSelectorV7) else cls(ck['mem_dim'],ck['context_dim'],ck['hidden_dim'],ck['dropout']));m.load_state_dict(ck['model']); m.switch_threshold=float(ck.get('switch_threshold',0.05)); m.force_fastest=bool(ck.get('force_fastest',False)); pack=torch.load(a.memory_pack,map_location='cpu',weights_only=False); fr=[float(x) for x in a.budget_fractions.split(',')]; c,_=select_clock(m,pack,a.device,fr,switch_threshold=m.switch_threshold); print(json.dumps({'selected_clock_period':c,'selected_clock_period_ns':c,'frequency_mode':'specified'}))
if __name__=='__main__':main()
