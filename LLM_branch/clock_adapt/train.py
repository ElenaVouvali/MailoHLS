import argparse, json
from pathlib import Path
import torch
from torch import nn
from .model import ClockResidualSelector

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train_features',required=True); ap.add_argument('--val_features',required=True); ap.add_argument('--hidden_dim',type=int,default=64); ap.add_argument('--dropout',type=float,default=.1); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight_decay',type=float,default=1e-4); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--patience',type=int,default=12); ap.add_argument('--seed',type=int,default=123); ap.add_argument('--output_dir',required=True); ap.add_argument('--clock_class_weighting',choices=('uniform','inverse_sqrt'),default='inverse_sqrt'); ap.add_argument('--regret_weight',type=float,default=.2)
    a=ap.parse_args(); torch.manual_seed(a.seed); train=torch.load(a.train_features,weights_only=False); val=torch.load(a.val_features,weights_only=False)
    if not train or not val: raise ValueError('Training and validation feature sets must both be non-empty')
    dim=int(train[0]['candidate_context'].shape[-1]); labels=torch.tensor([e['label'] for e in train],dtype=torch.long); num_clocks=len(train[0]['clocks']); class_weights=torch.ones(num_clocks) if a.clock_class_weighting=='uniform' else (labels.numel()/torch.bincount(labels,minlength=num_clocks).clamp_min(1).float()).sqrt().clamp_(0.5,4.0); class_weights=class_weights/class_weights.mean(); model=ClockResidualSelector(128,dim,a.hidden_dim,a.dropout); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); best=float('inf'); bad=0; generator=torch.Generator().manual_seed(a.seed)
    for epoch in range(a.epochs):
        model.train(); total=0
        for index in torch.randperm(len(train),generator=generator).tolist():
            e=train[index]
            logits=[]
            logits=model(e['memory'],e['memory_mask'],e['candidate_context']); label=torch.tensor(e['label']); base_ce=nn.functional.cross_entropy(logits.view(1,-1),label.view(1),reduction='none')[0]; ce=base_ce*class_weights[label]
            adps=e['case'].get('adp_by_clock',{}); gold=float(e['case'].get('gold_adp',1.0)); regrets=torch.tensor([min(10.0,max(0.0,float(adps.get(str(c),gold*11))/max(gold,1e-9)-1.0)) for c in e['clocks']],dtype=logits.dtype)
            loss=ce + a.regret_weight*torch.softmax(logits,dim=-1).dot(regrets)
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss)
        model.eval(); losses=[]
        with torch.no_grad():
            for e in val:
                logits=model(e['memory'],e['memory_mask'],e['candidate_context']).view(1,-1)
                losses.append(float(nn.functional.cross_entropy(logits,torch.tensor([e['label']]))))
        score=sum(losses)/max(1,len(losses))
        if score < best: best=score; bad=0; Path(a.output_dir).mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'feature_dim':dim,'mem_dim':128,'context_dim':dim,'hidden_dim':a.hidden_dim,'dropout':a.dropout,'clock_menu':train[0]['clocks'],'class_weights':class_weights.tolist(),'feature_schema':'memory_attention+budget_bram_dsp_ff_lut+capacity_log+clock_log2_v3','best_val_loss':best},Path(a.output_dir)/'selector.pt')
        else: bad += 1
        if bad >= a.patience: break
    print(json.dumps({'best_val_loss':best,'epochs':epoch+1}))
if __name__=='__main__': main()
