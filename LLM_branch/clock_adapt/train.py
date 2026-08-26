import argparse, json
from pathlib import Path
import torch
from torch import nn
from .model import ClockResidualSelector

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train_features',required=True); ap.add_argument('--val_features',required=True); ap.add_argument('--hidden_dim',type=int,default=64); ap.add_argument('--dropout',type=float,default=.1); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight_decay',type=float,default=1e-4); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--patience',type=int,default=12); ap.add_argument('--seed',type=int,default=123); ap.add_argument('--output_dir',required=True); ap.add_argument('--clock_class_weighting',default='inverse_sqrt')
    a=ap.parse_args(); torch.manual_seed(a.seed); train=torch.load(a.train_features,weights_only=False); val=torch.load(a.val_features,weights_only=False); dim=train[0]['features'].numel(); model=ClockResidualSelector(dim,a.hidden_dim,a.dropout); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); best=float('inf'); bad=0
    for epoch in range(a.epochs):
        model.train(); total=0
        for e in train:
            logits=[]
            for i in range(len(e['clocks'])): logits.append(model(e['features'][i].unsqueeze(0),e['scores'][i].view(1)))
            loss=nn.functional.cross_entropy(torch.cat(logits).view(1,-1),torch.tensor([e['label']]))
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss)
        model.eval(); losses=[]
        with torch.no_grad():
            for e in val:
                logits=torch.cat([model(e['features'][i].unsqueeze(0),e['scores'][i].view(1)) for i in range(len(e['clocks']))]).view(1,-1)
                losses.append(float(nn.functional.cross_entropy(logits,torch.tensor([e['label']]))))
        score=sum(losses)/max(1,len(losses))
        if score < best: best=score; bad=0; Path(a.output_dir).mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'feature_dim':dim,'best_val_loss':best},Path(a.output_dir)/'selector.pt')
        else: bad += 1
        if bad >= a.patience: break
    print(json.dumps({'best_val_loss':best,'epochs':epoch+1}))
if __name__=='__main__': main()
