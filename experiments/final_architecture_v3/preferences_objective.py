"""Standard reference DPO on summed nonforced values; SFT is a separate arm."""
import torch
from torch.nn import functional as F

def selected_logps(logits,labels,weights):
 mask=labels[1:].ne(-100)
 if not mask.any():raise ValueError('No nonforced value tokens')
 selected=logits[0,:-1][mask].float();targets=labels[1:][mask]
 logps=F.log_softmax(selected,-1).gather(-1,targets[:,None]).squeeze(-1)
 masses=weights[1:][mask].float()
 return logps.sum(dtype=torch.float32),-(logps*masses).sum()/masses.sum().clamp_min(1e-8),int(mask.sum())

def dpo_loss(chosen,rejected,ref_chosen,ref_rejected,beta=.1):
 return -F.logsigmoid(float(beta)*((chosen.float()-rejected.float())-(ref_chosen.detach().float()-ref_rejected.detach().float())))
