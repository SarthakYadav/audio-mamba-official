import torch
import torch.nn as nn
import torch.nn.functional as F


def mae_loss(pred, target, mask=None, norm_pix_loss:bool=False):
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var+1e-6) ** 0.5
    loss = (pred - target) ** 2
    if mask is not None:
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
    else:
        loss = loss.mean()
    return loss
