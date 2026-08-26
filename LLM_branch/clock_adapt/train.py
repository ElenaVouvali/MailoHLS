import argparse, json
from pathlib import Path
import torch
from torch import nn
from .model import ClockResidualSelector

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train_features',required=True); ap.add_argument('--val_features',required=True); ap.add_argument('--hidden_dim',type=int,default=64); ap.add_argument('--dropout',type=float,default=.1); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight_decay',type=float,default=1e-4); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--patience',type=int,default=12); ap.add_argument('--seed',type=int,default=123); ap.add_argument('--output_dir',required=True); ap.add_argument('--clock_class_weighting',default='inverse_sqrt')
    a=ap.parse_args(); torch.manual_seed(a.seed); train=torch.load(a.train_features,weights_only=False); val=torch.load(a.val_features,weights_only=False)
    if not train or not val: raise ValueError('Training and validation feature sets must both be non-empty')
    dim=int(train[0]['features'].shape[-1]); labels=torch.tensor([e['label'] for e in train],dtype=torch.long); num_clocks=len(train[0]['clocks']); class_weights=(labels.numel()/torch.bincount(labels,minlength=num_clocks).clamp_min(1).float()).sqrt().clamp_(0.5,4.0); model=ClockResidualSelector(dim,a.hidden_dim,a.dropout); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); best=float('inf'); bad=0
    for epoch in range(a.epochs):
        model.train(); total=0
        for e in train:
            logits=[]
            logits.append(model(e['features']))
            logits=logits[0]; ce=nn.functional.cross_entropy(logits.view(1,-1),torch.tensor([e['label']]),weight=class_weights)
            adps=e['case'].get('adp_by_clock',{}); gold=float(e['case'].get('gold_adp',1.0)); regrets=torch.tensor([min(10.0,max(0.0,float(adps.get(str(c),gold*11))/max(gold,1e-9)-1.0)) for c in e['clocks']],dtype=logits.dtype)
            loss=ce + 0.2*torch.softmax(logits,dim=-1).dot(regrets)
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss)
        model.eval(); losses=[]
        with torch.no_grad():
            for e in val:
                logits=model(e['features']).view(1,-1)
                losses.append(float(nn.functional.cross_entropy(logits,torch.tensor([e['label']]))))
        score=sum(losses)/max(1,len(losses))
        if score < best: best=score; bad=0; Path(a.output_dir).mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'feature_dim':dim,'hidden_dim':a.hidden_dim,'dropout':a.dropout,'clock_menu':train[0]['clocks'],'class_weights':class_weights.tolist(),'feature_schema':'pooled_memory_128+budget_4+capacity_4+clock_log2_v1','best_val_loss':best},Path(a.output_dir)/'selector.pt')
        else: bad += 1
        if bad >= a.patience: break
    print(json.dumps({'best_val_loss':best,'epochs':epoch+1}))
if __name__=='__main__': main()
