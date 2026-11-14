"""Exercise unchanged checkpoint functions on a CPU toy, never model weights."""
import json,random,sys
from pathlib import Path
from types import SimpleNamespace
import torch
deployment=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(deployment/'code'))
from experiments.film_gmt_stage2_final.train import save_checkpoint,restore_checkpoint
out=deployment/'runs/checkpoint_roundtrip'
out.mkdir(parents=True,exist_ok=False)
torch.manual_seed(123);random.seed(123)
model=torch.nn.Module()
model.gated_cross_attn_layer=torch.nn.Linear(2,2)
opt=torch.optim.AdamW(model.parameters(),lr=.001)
scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1/(1+step))
scaler=torch.amp.GradScaler('cuda',enabled=False)
model.gated_cross_attn_layer(torch.ones(1,2)).sum().backward();opt.step();scheduler.step()
expected={k:v.detach().clone() for k,v in model.state_dict().items()}
a=SimpleNamespace(output=out)
checkpoint=save_checkpoint(a,model,None,opt,scaler,scheduler,150,{},'frozen-test','protocol-test')
expected_torch=torch.rand(4);expected_python=random.random();expected_lr=opt.param_groups[0]['lr']
with torch.no_grad():
    for p in model.parameters():p.zero_()
opt.param_groups[0]['lr']=100
torch.manual_seed(999);random.seed(999)
a.resume_from=checkpoint
step=restore_checkpoint(a,model,opt,scaler,scheduler,'protocol-test','frozen-test')
assert step==150 and all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
assert torch.equal(torch.rand(4),expected_torch) and random.random()==expected_python
assert opt.param_groups[0]['lr']==expected_lr
rejected=False
try:restore_checkpoint(a,model,opt,scaler,scheduler,'wrong-protocol','frozen-test')
except ValueError:rejected=True
assert rejected
print(json.dumps({'status':'PASS','scope':'CPU toy; exact committed save/restore functions',
 'restored_step':step,'weights_optimizer_scheduler_rng_restored':True,
 'protocol_mismatch_rejected':True,'protected_models_modified':False},indent=2))
