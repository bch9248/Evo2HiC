import torch
from torch import Tensor
from beartype.typing import List, Union, Optional
from torch.nn.functional import mse_loss, huber_loss

from dataset.normalizer import Normalizer
from functools import partial, wraps
import numpy as np

from model.multiresolution_block import restore_size

def multiresolution_loss(
    input: Tensor,
    target: Tensor,
    img_normalize: bool,
    loss_fn,
    normalizer: Normalizer, 
    relative_resolutions: list[int],
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean"
):
    losses = []

    base_loss = loss_fn(input, target, reduction='none')
    losses.append(base_loss)

    if img_normalize:
        input = (input+1) * 0.5
        target = (target+1) * 0.5

    # don't include negative in multi-resolution 
    zero = normalizer.normalize(0).item()
    non_negative = input >= zero
    input = input.clamp(min=zero)

    # for small pixel calculate mse for different resolution
    for r in relative_resolutions:
        if r==1: continue

        i = torch.nn.functional.avg_pool2d(input, r ,r)
        t = torch.nn.functional.avg_pool2d(target, r ,r)

        i = restore_size(i, r, input)
        t = restore_size(t, r, target)

        upbound = normalizer.normalize(5).item()
        mask = (t<upbound) & non_negative

        loss_r = loss_fn(i, t, reduction='none')
        loss_r = torch.where(mask, loss_r, base_loss)

        losses.append(loss_r)

    loss = torch.stack(losses, dim=-1).mean(dim=-1)

    assert not loss.isnan().any()
    assert not loss.isinf().any()

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise NotImplementedError
