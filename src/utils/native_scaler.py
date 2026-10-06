# --------------------------------------------------------
# Based on timm and MAE-priv code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------

import math

import numpy as np
import torch
import torch.distributed as dist
from torch.amp.grad_scaler import GradScaler
from torch import inf


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = GradScaler('cuda', enabled=enabled)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        skip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        # 1. Backpropagate scaled loss
        self._scaler.scale(loss).backward(create_graph=create_graph)

        norm = None
        if update_grad:
            # 2. Unscale once
            self._scaler.unscale_(optimizer)

            # 3. Get norm (clip_grad_norm_ returns the norm even if it clips)
            if clip_grad is not None:
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                norm = get_grad_norm_(parameters)

            # 4. Global Sync: Check if we should skip (Finite + Skip Threshold)
            should_skip = (skip_grad is not None and norm >= skip_grad) or not torch.isfinite(norm)

            # Convert to tensor for all_reduce if using DistributedDataParallel
            if dist.is_initialized():
                skip_flag = torch.tensor([1.0 if should_skip else 0.0], device=loss.device)
                dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
                should_skip = skip_flag.item() > 0

            # 5. Step or Skip
            if not should_skip:
                self._scaler.step(optimizer)

            self._scaler.update()

        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule
