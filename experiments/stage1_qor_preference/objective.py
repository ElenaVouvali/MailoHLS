"""Standard summed, masked completion DPO plus the existing action-normalized SFT anchor."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def selected_logps(logits, labels, token_weights):
    """Causal value tokens only; labels/masses come unchanged from SFTDataset."""
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError('This bounded pilot scores one complete sequence per forward')
    mask = labels[1:].ne(-100)
    if not mask.any(): raise ValueError('No nonforced value tokens')
    selected = logits[0, :-1][mask].float()
    targets = labels[1:][mask]
    lp = F.log_softmax(selected, dim=-1).gather(-1, targets[:,None]).squeeze(-1)
    masses = token_weights[1:][mask].float()
    sequence_sum = lp.sum(dtype=torch.float32)
    sft = -(lp * masses).sum(dtype=torch.float32) / masses.sum().clamp_min(1.)
    correct = selected.argmax(-1).eq(targets).sum()
    return sequence_sum, sft, int(mask.sum()), correct


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=.1):
    margin = (chosen.float() - rejected.float()) - (reference_chosen.detach().float() - reference_rejected.detach().float())
    return -F.logsigmoid(float(beta) * margin)


def pair_loss(chosen, rejected, reference_chosen, reference_rejected, sft, beta=.1, weight=.2):
    dpo = dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta)
    return sft.float() + weight * dpo, dpo
