import argparse, json, hashlib
from pathlib import Path
import torch
from torch import nn
from .model import (ClockResidualSelector, AutoClockOverrideSelector,
                    AutoClockOverrideSelectorV5, AutoClockOverrideSelectorV6,
                    AutoClockOverrideSelectorV7, select_auto_clock_decision)
from LLM_branch.common.mailohls_contract import DEVICE_RESOURCES, supported_clock_periods

INFEASIBLE_REGRET = 10.0

def select_switch_threshold(oof_predictions):
    """Select a switch threshold from out-of-fold *predictions*.

    Ground-truth QoR is used only to score the selected clock, never to choose
    the candidate clock during calibration.
    """
    candidates = sorted({0.0, .01, .02, .05, .10, .20, .30, .50})
    best = (float('inf'), .05)
    for threshold in candidates:
        losses=[]
        for e, predicted_delta in oof_predictions:
            _, fast = baseline_target(e)
            candidate = int(predicted_delta.argmin())
            selected = candidate if float(predicted_delta[candidate]) < -threshold else fast
            losses.append(selected_regret(e, selected))
        score=sum(losses)/max(1,len(losses))
        if score < best[0]: best=(score,threshold)
    return best[1]

def _sha256(path):
    if not path: return None
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def _qor(e):
    """Canonical objective QoR vector (AUTO is trained on the same scalar)."""
    case = e['case']; values = case.get('qor_by_clock', case.get('adp_by_clock', {}))
    return torch.tensor([float(values.get(str(c), float('nan'))) for c in e['clocks']], dtype=torch.float32)


def _override_qor(e, infeasible_regret=INFEASIBLE_REGRET):
    raw = _qor(e)
    feasible_meta = e["case"].get("clock_feasible", {})
    flags = []
    for i, c in enumerate(e["clocks"]):
        finite_positive = bool(torch.isfinite(raw[i]).item()) and bool((raw[i] > 0).item())
        declared = bool(feasible_meta.get(str(c), finite_positive))
        flags.append(finite_positive and declared)
    feasible = torch.tensor(flags, dtype=torch.bool)
    if not feasible.any():
        raise ValueError(f"No feasible AUTO clock for {e['case'].get('kernel')} / {e['case'].get('device')}")
    best_qor = raw[feasible].min()
    out = raw.clone()
    out[~feasible] = best_qor * (1.0 + float(infeasible_regret))
    if not torch.isfinite(out).all():
        raise RuntimeError(f"Non-finite AUTO override QoR: {out.tolist()}")
    return out

def has_complete_qor(e):
    qor = _qor(e)
    return bool(torch.isfinite(qor).all() and (qor > 0).all())

def baseline_target(e):
    qor = _qor(e)
    if not torch.isfinite(qor).all():
        raise ValueError(f"Incomplete AUTO QoR vector: {qor.tolist()}")
    if not (qor > 0).all():
        raise ValueError(f"Non-positive AUTO QoR vector: {qor.tolist()}")
    fast_idx = int(torch.tensor(e['clocks']).argmin())
    return torch.log(qor / qor[fast_idx]), fast_idx

def selected_regret(e, selected):
    qor = _override_qor(e)
    gold = int(qor.argmin())
    return max(0.0, float(qor[selected] / qor[gold] - 1.0))


def build_override_targets(adp, available, clock_values, min_override_regret=0.0):
    inf = torch.tensor(float("inf"), dtype=adp.dtype, device=adp.device)
    valid_adp = adp.masked_fill(~available, inf)
    batch = torch.arange(adp.shape[0], device=adp.device)
    best_idx = valid_adp.argmin(dim=-1)
    clocks = clock_values.unsqueeze(0).expand_as(adp)
    fastest_idx = clocks.masked_fill(~available, inf).argmin(dim=-1)
    best_adp = valid_adp[batch, best_idx]
    fastest_adp = valid_adp[batch, fastest_idx]
    fn_regret = ((fastest_adp - best_adp) / best_adp.clamp_min(1e-12)).clamp_min(0.0)
    y_override = best_idx.ne(fastest_idx) & (fn_regret >= float(min_override_regret))
    fastest_period = clock_values[fastest_idx]
    slower = available & (clocks > fastest_period.unsqueeze(1))
    cheapest_slower = valid_adp.masked_fill(~slower, inf).min(dim=-1).values
    fp_regret = ((cheapest_slower - best_adp) / best_adp.clamp_min(1e-12)).clamp_min(0.0)
    fp_regret = torch.where(torch.isfinite(fp_regret), fp_regret, torch.zeros_like(fp_regret))
    decision_regret = torch.where(y_override, fn_regret, fp_regret)
    weight = 1.0 + 1.5 * torch.log1p(decision_regret).clamp(max=3.0)
    return fastest_idx, best_idx, y_override, weight, fn_regret


def auto_clock_override_loss(override_logits, clock_logits, adp, available,
                             clock_values, slow_loss_weight=0.35):
    fastest_idx, best_idx, y_override, weight, _ = build_override_targets(adp, available, clock_values)
    gate = torch.nn.functional.binary_cross_entropy_with_logits(
        override_logits, y_override.float(), reduction="none")
    gate_loss = (gate * weight).sum() / weight.sum().clamp_min(1e-12)
    if y_override.any():
        masked = clock_logits.masked_fill(~available, float("-inf"))
        batch = torch.arange(clock_logits.shape[0], device=clock_logits.device)
        masked[batch, fastest_idx] = float("-inf")
        slow_loss = torch.nn.functional.cross_entropy(masked[y_override], best_idx[y_override])
    else:
        slow_loss = override_logits.sum() * 0.0
    return {"loss": gate_loss + slow_loss_weight * slow_loss,
            "gate_loss": gate_loss.detach(), "slow_loss": slow_loss.detach(),
            "override_fraction": y_override.float().mean().detach()}


def auto_clock_override_v5_loss(override_logits, clock_logits, adp, available,
                               clock_values, min_override_regret=0.02,
                               slow_loss_weight=0.20):
    fastest_idx, best_idx, y_override, weight, _ = build_override_targets(
        adp, available, clock_values, min_override_regret
    )
    gate_per = torch.nn.functional.binary_cross_entropy_with_logits(
        override_logits, y_override.float(), reduction="none"
    )
    gate_loss = (gate_per * weight).sum() / weight.sum().clamp_min(1e-12)
    masked = clock_logits.masked_fill(~available, float("-inf"))
    batch = torch.arange(clock_logits.shape[0], device=clock_logits.device)
    slow_mask = available.clone(); slow_mask[batch, fastest_idx] = False
    slow_logits = masked.clone(); slow_logits[batch, fastest_idx] = float("-inf")
    slow_loss = (torch.nn.functional.cross_entropy(slow_logits[y_override], best_idx[y_override])
                 if y_override.any() else override_logits.sum() * 0.0)
    best_adp = adp.masked_fill(~available, float("inf")).min(dim=-1, keepdim=True).values
    target_utility = -torch.log(adp / best_adp.clamp_min(1e-12))
    target_utility = target_utility - target_utility.mean(dim=-1, keepdim=True)
    pred_utility = clock_logits - clock_logits.mean(dim=-1, keepdim=True)
    value_loss = torch.nn.functional.smooth_l1_loss(pred_utility, target_utility)
    return {"loss": gate_loss + 0.50 * value_loss + slow_loss_weight * slow_loss,
            "gate_loss": gate_loss.detach(), "slow_loss": slow_loss.detach(),
            "value_loss": value_loss.detach(), "override_fraction": y_override.float().mean().detach()}


def build_feasibility_targets(e):
    return torch.tensor([float(e["case"].get("clock_feasible", {}).get(str(c), True)) for c in e["clocks"]], dtype=torch.float32)


def auto_clock_override_v6_loss(override_logits, clock_logits, feasibility_logits,
                               adp, available, clock_values,
                               min_override_regret=0.02, slow_loss_weight=0.20,
                               feasibility_loss_weight=0.25, feasibility_target=None):
    base = auto_clock_override_v5_loss(override_logits, clock_logits, adp, available, clock_values, min_override_regret, slow_loss_weight)
    target = (available.float() if feasibility_target is None else feasibility_target.float())
    raw = torch.nn.functional.binary_cross_entropy_with_logits(feasibility_logits, target, reduction="none")
    weight = torch.where(target < 0.5, torch.full_like(target, 10.0), torch.ones_like(target))
    feas_loss = (raw * weight).sum() / weight.sum().clamp_min(1e-12)
    base.update(feasibility_loss=feas_loss.detach(), loss=base["loss"] + feasibility_loss_weight * feas_loss)
    return base


def select_override_threshold(oof_rows, max_false_override_rate=0.05):
    """Select a conservative gate threshold from group-aware OOF rows.

    Each row must provide ``override_probability``, ``y_override`` and
    ``false_override``.  The latter is the deployment decision cost indicator
    (1 only when a fastest-optimal case is overridden).  This helper is kept
    separate from fitting so callers can construct OOF predictions with
    GroupKFold and never tune the threshold on the validation family.
    """
    best = None
    for i in range(50, 100):
        tau = i / 100.0
        selected = [r for r in oof_rows if float(r["override_probability"]) >= tau]
        false_rate = (sum(float(r["false_override"]) for r in selected) / max(1, sum(1 for r in oof_rows if not r["y_override"])))
        if false_rate > max_false_override_rate:
            continue
        mean_regret = sum(float(r.get("regret", 0.0)) for r in oof_rows if float(r["override_probability"]) >= tau) / max(1, len(oof_rows))
        key = (mean_regret, tau)
        if best is None or key < best[0]:
            best = (key, tau, false_rate)
    if best is None:
        return 0.99
    return float(best[1])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train_features',required=True); ap.add_argument('--val_features',required=True); ap.add_argument('--hidden_dim',type=int,default=64); ap.add_argument('--dropout',type=float,default=.1); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight_decay',type=float,default=1e-4); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--patience',type=int,default=12); ap.add_argument('--seed',type=int,default=123); ap.add_argument('--output_dir',required=True); ap.add_argument('--switch_threshold',type=float,default=None); ap.add_argument('--regret_weight',type=float,default=.1); ap.add_argument('--temperature',type=float,default=1.0); ap.add_argument('--slow_loss_weight',type=float,default=.35); ap.add_argument('--min_override_regret',type=float,default=.02); ap.add_argument('--architecture',choices=('residual','override_v4','override_v5','override_v6','override_v7'),default='residual'); ap.add_argument('--split_json',required=True); ap.add_argument('--budget_bank',required=True); ap.add_argument('--memory_manifest',required=True); ap.add_argument('--cases_dir',required=True)
    a=ap.parse_args(); a.switch_threshold = (0.5 if a.architecture in ('override_v4','override_v5','override_v6','override_v7') else 0.05) if a.switch_threshold is None else a.switch_threshold; torch.manual_seed(a.seed); train=torch.load(a.train_features,weights_only=False); val=torch.load(a.val_features,weights_only=False)
    if not train or not val: raise ValueError('Training and validation feature sets must both be non-empty')
    dim=int(train[0]['candidate_context'].shape[-1]); mem_dim=int(train[0]['memory'].shape[-1]); n_clocks=len(train[0]['clocks']); model=(AutoClockOverrideSelectorV7(mem_dim,dim,a.hidden_dim,n_clocks,a.dropout) if a.architecture == 'override_v7' else AutoClockOverrideSelectorV6(mem_dim,dim,a.hidden_dim,n_clocks,a.dropout) if a.architecture == 'override_v6' else AutoClockOverrideSelector(mem_dim,dim,a.hidden_dim,n_clocks,a.dropout) if a.architecture == 'override_v4' else AutoClockOverrideSelectorV5(mem_dim,dim,a.hidden_dim,n_clocks,a.dropout) if a.architecture == 'override_v5' else ClockResidualSelector(mem_dim,dim,a.hidden_dim,a.dropout)); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); best=float('inf'); bad=0; generator=torch.Generator().manual_seed(a.seed)
    for epoch in range(a.epochs):
        model.train(); total=0
        for index in torch.randperm(len(train),generator=generator).tolist():
            e=train[index]
            if a.architecture in ('override_v4','override_v5','override_v6','override_v7'):
                outputs = model(e['memory'],e['memory_mask'],e['candidate_context'])
                override, clock_logits = outputs[:2]
                qor=_override_qor(e).to(override.device).unsqueeze(0); available=e.get('available', torch.ones(n_clocks,dtype=torch.bool)).to(override.device).unsqueeze(0)
                clocks=torch.tensor(e['clocks'],dtype=qor.dtype,device=override.device)
                if a.architecture in ('override_v6','override_v7'):
                    feas = build_feasibility_targets(e).to(override.device).unsqueeze(0)
                    losses=auto_clock_override_v6_loss(override.unsqueeze(0),clock_logits.unsqueeze(0),outputs[2].unsqueeze(0),qor,available,clocks,a.min_override_regret,a.slow_loss_weight,feasibility_target=feas if a.architecture == 'override_v7' else None)
                else:
                    losses=(auto_clock_override_v5_loss(override.unsqueeze(0),clock_logits.unsqueeze(0),qor,available,clocks,a.min_override_regret,a.slow_loss_weight) if a.architecture == 'override_v5' else auto_clock_override_loss(override.unsqueeze(0),clock_logits.unsqueeze(0),qor,available,clocks,a.slow_loss_weight))
                loss=losses['loss']
                if (not torch.isfinite(override).all()
                        or not torch.isfinite(clock_logits).all()
                        or (a.architecture in ('override_v6','override_v7')
                            and not torch.isfinite(outputs[2]).all())):
                    raise RuntimeError(f"Non-finite AUTO outputs at train index {index}")
            else:
                raw=model(e['memory'],e['memory_mask'],e['candidate_context']); target, fast_idx=baseline_target(e); target=target.to(raw.device); pred=raw-raw[fast_idx]
                qor=_qor(e); regret=qor/qor.min()-1.0; probability=torch.softmax(-pred/max(float(a.temperature),1e-6),dim=-1)
                loss=nn.functional.smooth_l1_loss(pred,target)+float(a.regret_weight)*(probability*regret.to(pred.device)).sum()
            if not torch.isfinite(loss): raise RuntimeError(f"Non-finite AUTO loss at train index {index}; qor={qor.detach().cpu().tolist()}")
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total += float(loss)
        model.eval(); hard_regrets=[]
        with torch.no_grad():
            for e in val:
                if a.architecture in ('override_v4','override_v5','override_v6','override_v7'):
                    outputs=model(e['memory'],e['memory_mask'],e['candidate_context']); override, clock_logits=outputs[:2]; clocks=torch.tensor(e['clocks']); fast=int(clocks.argmin()); available=e.get('available',torch.ones(len(e['clocks']),dtype=torch.bool));
                    decision=select_auto_clock_decision(
                        override, clock_logits, clocks, a.switch_threshold,
                        feasibility_logits=(outputs[2] if a.architecture in ('override_v6', 'override_v7') else None),
                    )
                    selected=decision['selected_idx']
                else:
                    raw=model(e['memory'],e['memory_mask'],e['candidate_context']).reshape(-1); target, fast_idx=baseline_target(e); pred=raw-raw[fast_idx]; candidate=int(pred.argmin()); selected=candidate if float(pred[candidate]) < -a.switch_threshold else fast_idx
                hard_regrets.append(selected_regret(e, selected))
        hard_regrets=torch.tensor(hard_regrets)
        score=hard_regrets.mean().item() + 0.10*hard_regrets.quantile(0.90).item()
        if score < best: best=score; bad=0; Path(a.output_dir).mkdir(parents=True,exist_ok=True); schema='memory_attention+budget_bram_dsp_ff_lut+capacity_log+clock_log2+objective_onehot3+baseline_delta_v4'; clock_menus={device:list(supported_clock_periods(device)) for device in sorted(DEVICE_RESOURCES)}; torch.save({'model':model.state_dict(),'architecture':a.architecture,'feature_dim':dim,'mem_dim':mem_dim,'context_dim':dim,'hidden_dim':a.hidden_dim,'n_clocks':n_clocks,'dropout':a.dropout,'clock_menu':train[0]['clocks'],'clock_menus_by_device':clock_menus,'feature_schema':schema,'feature_schema_sha256':hashlib.sha256(schema.encode()).hexdigest(),'best_val_loss':best,'selection_metric':('override_gate_cost_weighted' if a.architecture in ('override_v4','override_v5') else 'baseline_anchored_delta_switch'),'switch_threshold':a.switch_threshold,'seed':a.seed,'optimizer':{'name':'AdamW','learning_rate':a.lr,'weight_decay':a.weight_decay},'training':{'epochs_requested':a.epochs,'patience':a.patience,'slow_loss_weight':a.slow_loss_weight,'min_override_regret':a.min_override_regret},'provenance':{'train_features_sha256':_sha256(a.train_features),'val_features_sha256':_sha256(a.val_features),'split_sha256':_sha256(a.split_json),'budget_bank_sha256':_sha256(a.budget_bank),'memory_manifest_sha256':_sha256(a.memory_manifest)}},Path(a.output_dir)/'selector.pt')
        else: bad += 1
        if bad >= a.patience: break
    print(json.dumps({'best_val_loss':best,'epochs':epoch+1}))
if __name__=='__main__': main()
