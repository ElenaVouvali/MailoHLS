import argparse, json, hashlib
from pathlib import Path
import torch
from torch import nn
from .model import ClockResidualSelector
from LLM_branch.common.mailohls_contract import DEVICE_RESOURCES, supported_clock_periods

def _sha256(path):
    if not path: return None
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def _regrets(e):
    adps=e['case'].get('adp_by_clock',{}); gold=float(e['case'].get('gold_adp',1.0))
    return torch.tensor([min(10.,max(0.,float(adps.get(str(c),gold*11))/max(gold,1e-9)-1.)) for c in e['clocks']],dtype=torch.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train_features',required=True); ap.add_argument('--val_features',required=True); ap.add_argument('--hidden_dim',type=int,default=64); ap.add_argument('--dropout',type=float,default=.1); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight_decay',type=float,default=1e-4); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--patience',type=int,default=12); ap.add_argument('--seed',type=int,default=123); ap.add_argument('--output_dir',required=True); ap.add_argument('--clock_class_weighting',choices=('uniform','inverse_sqrt'),default='inverse_sqrt'); ap.add_argument('--regret_weight',type=float,default=.2); ap.add_argument('--split_json',required=True); ap.add_argument('--budget_bank',required=True); ap.add_argument('--memory_manifest',required=True); ap.add_argument('--cases_dir',required=True)
    a=ap.parse_args(); torch.manual_seed(a.seed); train=torch.load(a.train_features,weights_only=False); val=torch.load(a.val_features,weights_only=False)
    if not train or not val: raise ValueError('Training and validation feature sets must both be non-empty')
    dim=int(train[0]['candidate_context'].shape[-1]); mem_dim=int(train[0]['memory'].shape[-1]); labels=torch.tensor([e['label'] for e in train],dtype=torch.long); num_clocks=len(train[0]['clocks']); class_weights=torch.ones(num_clocks) if a.clock_class_weighting=='uniform' else (labels.numel()/torch.bincount(labels,minlength=num_clocks).clamp_min(1).float()).sqrt().clamp_(0.5,4.0); class_weights=class_weights/class_weights.mean(); model=ClockResidualSelector(mem_dim,dim,a.hidden_dim,a.dropout); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); best=float('inf'); bad=0; generator=torch.Generator().manual_seed(a.seed)
    for epoch in range(a.epochs):
        model.train(); total=0
        for index in torch.randperm(len(train),generator=generator).tolist():
            e=train[index]
            logits=[]
            logits=model(e['memory'],e['memory_mask'],e['candidate_context']); label=torch.tensor(e['label']); base_ce=nn.functional.cross_entropy(logits.view(1,-1),label.view(1),reduction='none')[0]; ce=base_ce*class_weights[label]
            regrets=_regrets(e).to(logits.device)
            loss=ce + a.regret_weight*torch.softmax(logits,dim=-1).dot(regrets)
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss)
        model.eval(); hard_regrets=[]
        with torch.no_grad():
            for e in val:
                logits=model(e['memory'],e['memory_mask'],e['candidate_context']).reshape(-1)
                regrets=_regrets(e).to(logits.device)
                predicted=int(logits.argmax())
                hard_regrets.append(float(regrets[predicted]))
        hard_regrets=torch.tensor(hard_regrets)
        score=hard_regrets.mean().item() + 0.10*hard_regrets.quantile(0.90).item()
        if score < best: best=score; bad=0; Path(a.output_dir).mkdir(parents=True,exist_ok=True); schema='memory_attention+budget_bram_dsp_ff_lut+capacity_log+clock_log2_v3'; clock_menus={device:list(supported_clock_periods(device)) for device in sorted(DEVICE_RESOURCES)}; torch.save({'model':model.state_dict(),'feature_dim':dim,'mem_dim':mem_dim,'context_dim':dim,'hidden_dim':a.hidden_dim,'dropout':a.dropout,'clock_menu':train[0]['clocks'],'clock_menus_by_device':clock_menus,'class_weights':class_weights.tolist(),'feature_schema':schema,'feature_schema_sha256':hashlib.sha256(schema.encode()).hexdigest(),'best_val_loss':best,'selection_metric':'mean_hard_argmax_regret_plus_0.10_p90_hard_argmax_regret','regret_weight':a.regret_weight,'seed':a.seed,'optimizer':{'name':'AdamW','learning_rate':a.lr,'weight_decay':a.weight_decay,'betas':[0.9,0.999],'epsilon':1e-8},'training':{'epochs_requested':a.epochs,'patience':a.patience,'clock_class_weighting':a.clock_class_weighting},'provenance':{'train_features_sha256':_sha256(a.train_features),'val_features_sha256':_sha256(a.val_features),'split_sha256':_sha256(a.split_json),'budget_bank_sha256':_sha256(a.budget_bank),'memory_manifest_sha256':_sha256(a.memory_manifest),'cases':{n:_sha256(str(Path(a.cases_dir)/n)) if (Path(a.cases_dir)/n).is_file() else None for n in ('train.jsonl','val.jsonl','test.jsonl')}}},Path(a.output_dir)/'selector.pt')
        else: bad += 1
        if bad >= a.patience: break
    print(json.dumps({'best_val_loss':best,'epochs':epoch+1}))
if __name__=='__main__': main()
