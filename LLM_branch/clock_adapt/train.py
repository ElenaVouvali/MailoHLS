import argparse, json, hashlib
from pathlib import Path
import torch
from torch import nn
from .model import ClockResidualSelector
from LLM_branch.common.mailohls_contract import DEVICE_RESOURCES, supported_clock_periods

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
    qor = _qor(e)
    gold = int(qor.argmin())
    return max(0.0, float(qor[selected] / qor[gold] - 1.0))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train_features',required=True); ap.add_argument('--val_features',required=True); ap.add_argument('--hidden_dim',type=int,default=64); ap.add_argument('--dropout',type=float,default=.1); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight_decay',type=float,default=1e-4); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--patience',type=int,default=12); ap.add_argument('--seed',type=int,default=123); ap.add_argument('--output_dir',required=True); ap.add_argument('--switch_threshold',type=float,default=.05); ap.add_argument('--regret_weight',type=float,default=.1); ap.add_argument('--temperature',type=float,default=1.0); ap.add_argument('--split_json',required=True); ap.add_argument('--budget_bank',required=True); ap.add_argument('--memory_manifest',required=True); ap.add_argument('--cases_dir',required=True)
    a=ap.parse_args(); torch.manual_seed(a.seed); train=torch.load(a.train_features,weights_only=False); val=torch.load(a.val_features,weights_only=False)
    if not train or not val: raise ValueError('Training and validation feature sets must both be non-empty')
    dim=int(train[0]['candidate_context'].shape[-1]); mem_dim=int(train[0]['memory'].shape[-1]); model=ClockResidualSelector(mem_dim,dim,a.hidden_dim,a.dropout); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); best=float('inf'); bad=0; generator=torch.Generator().manual_seed(a.seed)
    for epoch in range(a.epochs):
        model.train(); total=0
        for index in torch.randperm(len(train),generator=generator).tolist():
            e=train[index]
            raw=model(e['memory'],e['memory_mask'],e['candidate_context']); target, fast_idx=baseline_target(e); target=target.to(raw.device); pred=raw-raw[fast_idx]
            qor=_qor(e); regret=qor/qor.min()-1.0; probability=torch.softmax(-pred/max(float(a.temperature),1e-6),dim=-1)
            loss=nn.functional.smooth_l1_loss(pred,target)+float(a.regret_weight)*(probability*regret.to(pred.device)).sum()
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss)
        model.eval(); hard_regrets=[]
        with torch.no_grad():
            for e in val:
                raw=model(e['memory'],e['memory_mask'],e['candidate_context']).reshape(-1); target, fast_idx=baseline_target(e); pred=raw-raw[fast_idx]; candidate=int(pred.argmin()); selected=candidate if float(pred[candidate]) < -a.switch_threshold else fast_idx
                hard_regrets.append(selected_regret(e, selected))
        hard_regrets=torch.tensor(hard_regrets)
        score=hard_regrets.mean().item() + 0.10*hard_regrets.quantile(0.90).item()
        if score < best: best=score; bad=0; Path(a.output_dir).mkdir(parents=True,exist_ok=True); schema='memory_attention+budget_bram_dsp_ff_lut+capacity_log+clock_log2+objective_onehot3+baseline_delta_v4'; clock_menus={device:list(supported_clock_periods(device)) for device in sorted(DEVICE_RESOURCES)}; torch.save({'model':model.state_dict(),'feature_dim':dim,'mem_dim':mem_dim,'context_dim':dim,'hidden_dim':a.hidden_dim,'dropout':a.dropout,'clock_menu':train[0]['clocks'],'clock_menus_by_device':clock_menus,'feature_schema':schema,'feature_schema_sha256':hashlib.sha256(schema.encode()).hexdigest(),'best_val_loss':best,'selection_metric':'baseline_anchored_delta_switch','switch_threshold':a.switch_threshold,'seed':a.seed,'optimizer':{'name':'AdamW','learning_rate':a.lr,'weight_decay':a.weight_decay},'training':{'epochs_requested':a.epochs,'patience':a.patience},'provenance':{'train_features_sha256':_sha256(a.train_features),'val_features_sha256':_sha256(a.val_features),'split_sha256':_sha256(a.split_json),'budget_bank_sha256':_sha256(a.budget_bank),'memory_manifest_sha256':_sha256(a.memory_manifest)}},Path(a.output_dir)/'selector.pt')
        else: bad += 1
        if bad >= a.patience: break
    print(json.dumps({'best_val_loss':best,'epochs':epoch+1}))
if __name__=='__main__': main()
