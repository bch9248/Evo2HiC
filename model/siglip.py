import torch
from torch import nn
import torch.utils.checkpoint
import numpy as np

class Siglip(nn.Module):
    def __init__(self, init_t = 10, init_b = -10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = nn.Parameter(torch.log(torch.tensor(init_t, dtype=float)))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=float))
    
    def forward(self, x):
        return x * self.t.exp() + self.b

def siglip(embeds0:torch.Tensor, embeds1:torch.Tensor, siglip:Siglip, avg_hic:bool = False, return_rank = False):    
    if avg_hic:
        embeds0 = embeds0.mean(dim=2)
        embeds1 = embeds1.mean(dim=2)
    else:
        assert embeds0.shape[2] == 1 and embeds1.shape[2] == 1

    embeds0 = nn.functional.normalize(embeds0, dim=-1)
    embeds1 = nn.functional.normalize(embeds1, dim=-1)

    G = 3
    n = embeds0.shape[-2]

    groups = []
    current_group0 = []
    current_group1 = []
    S = 0

    for i in range(0, n):
        if S + (n-i) > n*(n-1)//2//G:
            groups.append((current_group0, current_group1))
            current_group0 = []
            current_group1 = []
            S = 0

        current_group0.append(embeds0.diagonal(i, -2, -1))
        current_group1.append(embeds1.diagonal(i, -2, -1))

        S+=n-i

    if S > 0:
        if len(groups) > 0:
            last_group0, last_group1 = groups[-1]
            groups[-1] = (last_group0 + current_group0, last_group1 + current_group1)
        else:
            groups.append((current_group0, current_group1))

    if return_rank:
        ranks = []

    losses = []

    for group0, group1 in groups:
        group_embeds0 = torch.concat(group0, dim=-1)
        group_embeds1 = torch.concat(group1, dim=-1)

        group_embeds0 = group_embeds0.transpose(-1, -2)
        group_embeds1 = group_embeds1.transpose(-1, -2)

        group_embeds0 = group_embeds0.flatten(0, -2)
        group_embeds1 = group_embeds1.flatten(0, -2)

        labels = 2*torch.eye(group_embeds0.shape[0], device=embeds0.device) - 1

        logits = torch.inner(group_embeds0, group_embeds1)

        if return_rank:
            rank = (logits > logits.diag()[:, None]).sum(dim=-1)
            ranks.append(rank)

        logits = siglip(logits) * labels
        loss = nn.functional.logsigmoid(logits)
        loss = - loss.sum(dim=-1)

        losses.append(loss)

    if return_rank:
        rank = torch.concat(ranks)

    loss = torch.concat(losses).mean()

    if return_rank:
        return loss, rank

    return loss

def siglip_HiC_DNA(HiC_embeds:torch.Tensor, DNA_embeds:torch.Tensor, siglip:Siglip, avg_hic_emb:bool = False):    
    if avg_hic_emb:
        assert DNA_embeds.shape[2] == 1
        HiC_embeds = HiC_embeds.mean(dim=2)
        DNA_embeds = DNA_embeds.squeeze(dim=2)
    else:
        raise NotImplementedError

    HiC_embeds = HiC_embeds.permute((0,1,3,4,2))
    DNA_embeds = DNA_embeds.permute((0,1,3,4,2))

    HiC_embeds = nn.functional.normalize(HiC_embeds, dim=-1)
    DNA_embeds = nn.functional.normalize(DNA_embeds, dim=-1)

    HiC_embeds = HiC_embeds.flatten(0, -2)
    DNA_embeds = DNA_embeds.flatten(0, -2)

    labels = 2*torch.eye(HiC_embeds.shape[0], device=HiC_embeds.device) - 1

    logits = torch.inner(HiC_embeds, DNA_embeds)
    logits = siglip(logits) * labels.unsqueeze(0)
    loss = nn.functional.logsigmoid(logits)
    loss = - loss.sum(dim=-1)
    loss = loss.mean()

    return loss

def siglip_DNA_evo2(DNA_embeds:torch.Tensor, evo2_embeds:torch.Tensor, siglip:Siglip, negative_evo2_embeds:torch.Tensor = None):    
    DNA_embeds = DNA_embeds.flatten(0, -2)
    evo2_embeds = evo2_embeds.flatten(0, -2)

    labels = 2*torch.eye(DNA_embeds.shape[0], device=DNA_embeds.device) - 1

    if negative_evo2_embeds is not None:
        negative_evo2_embeds = negative_evo2_embeds.flatten(0,-2)
        negative_labels = -torch.ones((DNA_embeds.shape[0], negative_evo2_embeds.shape[0]), device=DNA_embeds.device)
        evo2_embeds = torch.concat([evo2_embeds, negative_evo2_embeds], dim=0)
        labels = torch.concat([labels, negative_labels], dim=1)

    DNA_embeds = nn.functional.normalize(DNA_embeds, dim=-1)
    evo2_embeds = nn.functional.normalize(evo2_embeds, dim=-1)

    logits = torch.inner(DNA_embeds, evo2_embeds)
    logits = siglip(logits) * labels.unsqueeze(0)
    loss = nn.functional.logsigmoid(logits)
    loss = - loss.sum(dim=-1)
    loss = loss.mean()

    return loss