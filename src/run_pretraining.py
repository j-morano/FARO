import datetime
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List
import random
import re
import copy

import numpy as np
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from einops import rearrange
from torchvision import utils as vutils
import torch.distributed as dist

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler, is_main_process
from utils.architecture import get_model_architecture, WeightedLoss, DOMAIN_CONF
from utils.datasets import (
    build_multimae_pretraining_dataset,
    build_multimae_contrastive_pretraining_dataset,
    build_patient_balanced_multimae_pretraining_dataset
)
from utils.optim_factory import create_optimizer
from utils.task_balancing import (NoWeightingStrategy,
                                  UncertaintyWeightingStrategy)
from config import args



def replace_domain_names(key):
    mapping = {
        # old -> new
        'rgb': 'bscan',
        'depth': 'slo',
        'semseg': 'bscanlayermap',
    }
    if 'oct' in args.in_domains  and 'bscanlayermap' not in args.in_domains:
        mapping['semseg'] = 'oct'
    if 'octsmall' in args.in_domains  and 'bscanlayermap' not in args.in_domains:
        mapping['semseg'] = 'octsmall'
    for dn in mapping:
        key = key.replace(dn, mapping[dn])
    return key


def get_model(args):
    model = get_model_architecture(args)

    if args.weights and 'dinov2' in args.weights:
        print(f">> Loading weights from {args.weights}")
        weights = torch.load(args.weights, map_location=args.device, weights_only=False)
        def do_replacements(key):
            key = key.replace('blocks.', 'encoder.')
            return key
        weights = {do_replacements(k): v for k, v in weights.items()}
        try:
            model.load_state_dict(weights, strict=True)
        except RuntimeError as e:
            for line in str(e).split('\n')[1:]:
                if 'Missing key' in line:
                    # Get all missing keys
                    missing_keys = re.findall(r'"(.*?)"', line)
                    for mk in missing_keys:
                        if not (
                            'global_tokens' in mk
                            or 'input_adapter' in mk
                            or 'output_adapter' in mk
                        ):
                            raise ValueError(f'Missing key {mk} in weights')
                elif 'Unexpected key' in line:
                    pass
                elif 'size mismatch' in line:
                    raise ValueError(line)
            model.load_state_dict(weights, strict=False)
        print('>> Weights loaded successfully')
    elif args.weights and 'RETFound' in args.weights:
        print(f">> Loading weights from {args.weights}")
        weights = torch.load(args.weights, map_location=args.device, weights_only=False)
        weights = weights['model']
        print('Changing model keys')
        def do_replacements(key):
            key = key.replace('blocks.', 'encoder.')
            key = key.replace('cls_token', 'global_tokens')
            # key = key.replace('decoder_encoder.', 'encoder.')
            return key
        weights = {do_replacements(k): v for k, v in weights.items()}
        try:
            model.load_state_dict(weights, strict=True)
        except RuntimeError as e:
            print(e)
            model.load_state_dict(weights, strict=False)
    elif args.weights and 'VisionFM' in args.weights:
        print(">> Loading weights from VisionFM")
        state_dict = torch.load(args.weights)['teacher']
        for key in list(state_dict.keys()):
            if 'backbone.blocks.' in key:
                state_dict[key.replace('backbone.blocks.', 'encoder.')] = state_dict.pop(key)
            if 'backbone.cls_token' in key:
                state_dict[key.replace('backbone.cls_token', 'global_tokens')] = state_dict.pop(key)
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    elif args.weights and 'medsam' in args.weights:
        print('Changing checkpoints keys for MedSAM model')
        state_dict = torch.load(args.weights)
        # Remove all weights starting with 'mask_decoder'
        for k in list(state_dict.keys()):
            if 'mask_decoder' in k or 'prompt_encoder' in k:
                del state_dict[k]
        for k in list(state_dict.keys()):
            if 'image_encoder' in k:
                state_dict[
                    k.replace('image_encoder.blocks', 'encoder')
                    .replace('lin1', 'fc1')
                    .replace('lin2', 'fc2')
                ] = state_dict.pop(k)
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    elif args.weights:
        print(f">> Loading weights from {args.weights}")
        weights = torch.load(args.weights, map_location=args.device, weights_only=False)['model']
        if '_vit_large' in args.weights:
            def do_replacements(key):
                key = key.replace('blocks.', 'encoder.')
                return key
            weights = {do_replacements(k): v for k, v in weights.items()}
            model.load_state_dict(weights, strict=False)
        else:
            # Replace original domain names in weights by the ones used now.
            weights = {
                replace_domain_names(k): v
                for k, v in weights.items()
            }
            try:
                model.load_state_dict(weights, strict=True)
            except RuntimeError as e:
                if '"encoder.' in str(e) or '"decoder.' in str(e):
                    raise RuntimeError(f"Error loading model weights.\n{e}")
                else:
                    try:
                        model.load_state_dict(weights, strict=True)
                    except RuntimeError as e:
                        # If there is a size mismatch, remove those keys
                        #   from the weights to load and try again
                        if 'size mismatch' in str(e):
                            lines = str(e).split('\n')
                            for line in lines:
                                if 'size mismatch for ' in line:
                                    # get text between 'size mismatch for' and ':'
                                    start = line.find('size mismatch for ') + len('size mismatch for ')
                                    end = line.find(':', start)
                                    key = line[start:end]
                                    print(f"🚨 IMPORTANT: Removing key {key} from weights")
                                    weights[key] = model.state_dict()[key]
                        print('>> Loading weights with strict=False after size mismatch')
                        msg = model.load_state_dict(weights, strict=False)
                        print(msg.missing_keys)
                        # Check if all missing keys are for modules for
                        #   global tokens (starting with global_modules).
                        for mk in msg.missing_keys:
                            if 'global_modules' not in mk:
                                raise ValueError(f'Missing key {mk} in weights')

    return model


def seed_worker(worker_id):
    # Set a fixed seed for each worker to ensure reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def scan_data(args, log_writer=None):
    # Get dataset
    if args.contrastive or args.dino:
        print('🚨 IMPORTANT: Building CONTRASTIVE pretraining dataset')
        dataset_train = build_multimae_contrastive_pretraining_dataset(args)
    elif args.patient_balanced:
        print('🚨 IMPORTANT: Building PATIENT-BALANCED pretraining dataset')
        dataset_train = build_patient_balanced_multimae_pretraining_dataset(args)
    else:
        print('🚨 IMPORTANT: Building STANDARD pretraining dataset (random b-scan selection)')
        dataset_train = build_multimae_pretraining_dataset(args)

    if args.repeat_dataset is not None and args.repeat_dataset > 1:
        print(f'==> Repeating dataset {args.repeat_dataset} times for training')
        dataset_train = RepeatedDataset(dataset_train, args.repeat_dataset)

    # if args.distributed:
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=sampler_rank,
        shuffle=True,
        drop_last=True,
    )
    print("Sampler_train = %s" % str(sampler_train))
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if log_writer is None:
        if global_rank == 0 and args.log_wandb:
            log_writer = utils.WandbLogger(args)
        else:
            log_writer = None

    # print(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        # shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )
    return data_loader_train, num_training_steps_per_epoch, log_writer


def info_nce_loss(z1, z2, temperature=0.1):
    """
    Standard InfoNCE loss for contrastive learning.

    Args:
        z1: Tensor of shape [N, D] (View 1 embeddings)
        z2: Tensor of shape [N, D] (View 2 embeddings)
        temperature: Scalar to scale the logits (smaller = sharper focus)
    """
    # 1. Normalize embeddings to the unit hypersphere
    z1 = F.normalize(z1, dim=1, eps=1e-10)
    z2 = F.normalize(z2, dim=1, eps=1e-10)

    # 2. Compute similarity matrix [N, N]
    # Each row i contains the dot product of z1[i] with all z2[j]
    logits = torch.matmul(z1, z2.T) / temperature

    # 3. Create labels (the diagonal)
    # View1[i] should match View2[i] -> indices are 0, 1, 2, ..., N-1
    labels = torch.arange(z1.size(0), device=z1.device)

    # 4. Cross Entropy pushes the diagonal values to be high
    # and off-diagonal values to be low
    return F.cross_entropy(logits, labels)


def symmetric_info_nce_loss(z1, z2, temperature=0.1):
    """
    Symmetric InfoNCE loss for contrastive learning.

    Args:
        z1: Tensor of shape [N, D] (View 1 embeddings)
        z2: Tensor of shape [N, D] (View 2 embeddings)
        temperature: Scalar to scale the logits (smaller = sharper focus)
    """
    loss1 = info_nce_loss(z1, z2, temperature)
    loss2 = info_nce_loss(z2, z1, temperature)
    return (loss1 + loss2) / 2.0


def full_gather(tensor):
    # Create a version that preserves the gradient connection
    tensor = tensor.contiguous()
    # Use a list to gather
    tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    # Detach all gathered tensors except the one that belongs to this rank
    # This prevents redundant gradient paths that often lead to 'inf'
    rank = torch.distributed.get_rank()
    tensors_gather[rank] = tensor

    output = torch.cat(tensors_gather, dim=0)
    return output


def symmetric_info_nce_loss_multi(z1, z2, temperature=0.1):
    # 1. Normalize local embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # 2. Gather all (keeping the gradient on local samples)
    z1_all = full_gather(z1) # [Total_N, D]
    z2_all = full_gather(z2) # [Total_N, D]

    # 3. Compute logits: local query vs. global candidates
    # Local z1 against all gathered z2
    logits_1 = torch.matmul(z1, z2_all.T) / temperature
    # Local z2 against all gathered z1
    logits_2 = torch.matmul(z2, z1_all.T) / temperature

    # 4. Correct Labels: Where is 'z1' inside 'z1_all'?
    # It's at [rank * batch_size : (rank + 1) * batch_size]
    rank = torch.distributed.get_rank()
    bs = z1.size(0)
    labels = torch.arange(rank * bs, (rank + 1) * bs, device=z1.device)

    # 5. Compute Symmetric Loss
    loss_1 = F.cross_entropy(logits_1, labels)
    loss_2 = F.cross_entropy(logits_2, labels)
    return (loss_1 + loss_2) / 2.0


class DINOLoss(nn.Module):
    # https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, interleaved=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        if interleaved and self.ncrops == 2:
            # In this case, student_output is already interleaved (v1, v2, v1, v2, ...)
            # We just need to separate the views
            student_out = [student_out[0::2], student_out[1::2]]
        else:
            student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        if interleaved and self.ncrops == 2:
            # Do the same
            teacher_out = teacher_out.detach()
            teacher_out = [teacher_out[0::2], teacher_out[1::2]]
        else:
            teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)



def train_one_epoch_constrastive(
    model: torch.nn.Module,
    data_loader: Iterable,
    tasks_loss_fn: Dict[str, torch.nn.Module],
    loss_balancer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = None,
    max_skip_norm: float = None,
    log_writer=None,
    lr_scheduler=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_encoded_tokens: int = 196,
    in_domains: List[str] = [],
    loss_on_unmasked: bool = True,
    alphas: float = 1.0,
    sample_tasks_uniformly: bool = False,
    standardize_depth: bool = True,
    extra_norm_pix_loss: bool = False,
    fp32_output_adapters: List[str] = [],
    contrastive = False,
    dino_model = None,
    model_without_ddp = None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for task in tasks_loss_fn:
        tasks_loss_fn[task].epoch = epoch  # type: ignore

    for step, (x, sample_info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Debug print
        # for task in x:
        #     print(task, x[task].shape, x[task].dtype, x[task].min().item(), x[task].max().item())
        # for task in sample_info:
        #     if type(sample_info[task]) is torch.Tensor:
        #         print(task, sample_info[task].shape, sample_info[task].dtype, sample_info[task].min().item(), sample_info[task].max().item())
        #     else:
        #         print(task, sample_info[task])
        # exit(0)
        if contrastive or dino_model is not None:
            # Combine batch and views dimensions for contrastive learning
            for task in x:
                x[task] = x[task].flatten(0, 1) # combine batch and views
            for task in sample_info:
                if type(sample_info[task]) is torch.Tensor:
                    sample_info[task] = sample_info[task].flatten(0, 1) # combine batch and views
        # print(sample_info)
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        # # Truncated depth standardization
        # if standardize_depth and 'depth' in tasks_dict:
        #     # Flatten depth and remove bottom and top 10% of values
        #     trunc_depth = torch.sort(rearrange(tasks_dict['depth'], 'b c h w -> b (c h w)'), dim=1)[0]
        #     trunc_depth = trunc_depth[:,int(0.1 * trunc_depth.shape[1]): int(0.9 * trunc_depth.shape[1])]
        #     tasks_dict['depth'] = (tasks_dict['depth'] - trunc_depth.mean(dim=1)[:,None,None,None]) / torch.sqrt(trunc_depth.var(dim=1)[:,None,None,None] + 1e-6)

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        with autocast('cuda'):
            preds, masks = model(
                input_dict,
                num_encoded_tokens=num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly,
                fp32_output_adapters=fp32_output_adapters
            )

            # Debugging (qualitative results)
            if args.output_dir and step == 0 and is_main_process():
                # Save images for debugging on first step
                print('Saving debug images')
                to_save = None
                tasks_str = ''
                predsc = {}
                masksc = {}
                input_dictc = {}
                for task in preds:
                    if task in tasks_dict:
                        if tasks_str == '':
                            tasks_str = task
                        else:
                            tasks_str += f'-{task}'
                        # Create a copy of the prediction and mask
                        predsc[task] = preds[task][:64].clone().detach()
                        masksc[task] = masks[task][:64].clone().detach()
                        input_dictc[task] = input_dict[task][:64].clone().detach()
                        # print(preds[task].shape, input_dict[task].shape)
                        # torch.Size([8, 1, 768, 768]) torch.Size([8, 1, 768, 768])
                        print(task, 'mask', masksc[task].dtype, masksc[task].shape,
                              masksc[task].sum().item(), masksc[task].numel())
                        print(task, 'preds', predsc[task].dtype, predsc[task].shape,
                            predsc[task].min().item(), predsc[task].max().item())
                        print(task, 'input', input_dictc[task].dtype, input_dictc[task].shape,
                              input_dictc[task].min().item(), input_dictc[task].max().item())
                        ## Replace unmasked patches in the prediction by the input
                        # Reshape mask to image. Original shape is (B, L),
                        #   where L is the number of tokens.
                        #   We need to reshape it to (B, C, [D,] H, W)
                        if task in ['layermaps', 'bscanlayermap']:
                            # Apply softmax and undo one-hot encoding
                            predsc[task] = F.softmax(predsc[task], dim=1)
                            predsc[task] = predsc[task].argmax(dim=1, keepdim=True).float()
                            # Add channel dimension to input
                            input_dictc[task] = input_dictc[task].unsqueeze(1)
                            # to 0-1 (for visualization)
                            predsc[task] = predsc[task] / predsc[task].max()
                            input_dictc[task] = input_dictc[task] / input_dictc[task].max()
                            c = 1

                        if isinstance(args.patch_size, int):
                            c = input_dictc[task].shape[1]
                            h = input_dictc[task].shape[2] // args.patch_size
                            w = input_dictc[task].shape[3] // args.patch_size
                        else:
                            c = input_dictc[task].shape[1]
                            h = input_dictc[task].shape[2] // args.patch_size[task][0]
                            w = input_dictc[task].shape[3] // args.patch_size[task][1]

                        mask = rearrange(masksc[task], 'b (c h w) -> b c h w', c=c, h=h, w=w).float()
                        # print(task, 'new_mask', mask.shape, mask.sum().item(), mask.numel())
                        fh, fw = input_dictc[task].shape[2:]
                        mask = F.interpolate(mask, size=(fh, fw), mode='nearest')
                        # 1 means masked (predicted), 0 means input
                        predsc[task] = predsc[task] * mask + input_dictc[task] * (1 - mask)
                        print(task, 'new_mask', mask.dtype, mask.shape, mask.sum().item(), mask.numel())
                        input_to_save = input_dictc[task]
                        pred_to_save = predsc[task]

                        # Fixed size for saving
                        input_to_save = F.interpolate(input_to_save, (128, 128), mode='nearest')
                        pred_to_save = F.interpolate(pred_to_save, (128, 128), mode='nearest')
                        mask_to_save = F.interpolate(mask, (128, 128), mode='nearest')
                        if to_save is None:
                            to_save = torch.cat([input_to_save, pred_to_save, mask_to_save], dim=2)
                        else:
                            # Add a white line between images
                            # white_line = torch.ones(
                            #     (input_to_save.shape[0], input_to_save.shape[1], 8, input_to_save.shape[3]),
                            #     device=to_save.device
                            # )
                            # to_save = torch.cat([to_save, white_line, input_to_save, pred_to_save, mask_to_save], dim=2)
                            to_save = torch.cat([to_save, input_to_save, pred_to_save, mask_to_save], dim=2)
                if to_save is not None:
                    # Maximum supported image dimension is 65500 pixels
                    # if to_save.shape[2] * to_save.shape[3] > 65500:
                    to_save = F.interpolate(to_save, size=(to_save.shape[2], to_save.shape[3]), mode='bilinear')
                    current_save_path = Path(args.output_dir) / 'debug' / f'{epoch}_{tasks_str}.jpg'
                    print(current_save_path)
                    vutils.save_image(to_save, current_save_path)

            # ADDED to avoid potential deadlock in distributed training
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if 'fft_bscan' in tasks_loss_fn:
                preds['fft_bscan'] = preds['bscan']
                tasks_dict['fft_bscan'] = tasks_dict['bscan']
                masks['fft_bscan'] = masks.get('bscan', None)

            task_losses = {}
            for task in preds:
                # print(task, 'pred', preds[task].shape, preds[task].dtype, preds[task].min().item(), preds[task].max().item())
                c_pred = preds[task].float()
                if task in tasks_dict:
                    target = tasks_dict[task]
                    # print(task, 'tgt', target.shape, target.dtype, target.min().item(), target.max().item())

                    if loss_on_unmasked:
                        task_losses[task] = tasks_loss_fn[task](c_pred, target)
                    else:
                        task_losses[task] = tasks_loss_fn[task](c_pred, target, mask=masks.get(task, None))
                elif task != 'cls':
                    target = sample_info[task].to(device, non_blocking=True)
                    # print(task, 'tgt', target.shape, target.dtype, target.min().item(), target.max().item())
                    task_losses[task] = tasks_loss_fn[task](c_pred, target)

            if 'cls' in preds and contrastive:
                cls_all = preds['cls'].float()

                # Slice the interleaved batch
                z1 = cls_all[0::2] # All View 1s
                z2 = cls_all[1::2] # All View 2s

                # # In your training loop:
                # if torch.distributed.is_initialized():
                #     z1_all = full_gather(z1)
                #     z2_all = full_gather(z2)
                # else:
                #     z1_all, z2_all = z1, z2

                # Calculate Contrastive Loss
                task_losses['contrastive'] = symmetric_info_nce_loss_multi(
                    z1,
                    z2,
                    temperature=0.2,
                ) * 0.1
            elif 'cls' in preds and dino_model is not None:
                # DINO loss
                cls_all_student = preds['cls'].float()
                # print('cls_all_student', cls_all_student.shape)
                # Get predictions for the teacher
                with torch.no_grad():
                    _, _, preds_teacher = dino_model.teacher(input_dict, mask_inputs=False)
                cls_all_teacher = preds_teacher['cls'].float()
                # print('cls_all_teacher', cls_all_teacher.shape)
                # Pass throuhg heads
                cls_all_student = dino_model.student_head(cls_all_student)
                with torch.no_grad():
                    cls_all_teacher = dino_model.teacher_head(cls_all_teacher)
                task_losses['dino'] = dino_model.loss_fn(
                    cls_all_student,
                    cls_all_teacher,
                    epoch,
                    interleaved=True,
                ) * 0.1

            # print(json.dumps({k: v.item() for k, v in task_losses.items()}, indent=2))

            weighted_task_losses = loss_balancer(task_losses)
            loss = sum(weighted_task_losses.values())

        loss_value = sum(task_losses.values()).item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        weighted_task_loss_values = {f'{task}_loss_weighted': l.item() for task, l in weighted_task_losses.items()}

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, skip_grad=max_skip_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        if dino_model is not None:
            dino_model.update_teacher(model_without_ddp, dino_model.teacher, it)
            dino_model.update_teacher(dino_model.student_head_without_ddp, dino_model.teacher_head, it)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        # metric_logger.update(grad_norm=grad_norm)
        if math.isfinite(grad_norm):
            metric_logger.update(grad_norm=grad_norm)
        else:
            # Optional: track how many times you skip
            metric_logger.update(grad_norm_inf_count=1)

        if log_writer is not None:
            log_writer.update({
                'loss': loss_value,
                'lr': max_lr,
                'weight_decay': weight_decay_value,
                'grad_norm': grad_norm,
            })
            log_writer.update(task_loss_values)
            log_writer.update(weighted_task_loss_values)
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


class AsymmetricLoss(nn.Module):
    '''https://github.com/Alibaba-MIIL/ASL/blob/8c9e0bd8d5d450cf19093363fc08aa7244ad4408/src/loss_functions/losses.py#L5
    '''
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        # return -loss.sum()
        return -loss.mean()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations
    https://github.com/Alibaba-MIIL/ASL/blob/8c9e0bd8d5d450cf19093363fc08aa7244ad4408/src/loss_functions/losses.py#L53
    '''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        # return -self.loss.sum()
        return -self.loss.mean()



class FFTLoss(nn.Module):
    def __init__(self, patch_size: int = 32):
        super().__init__()
        self.patch_size = patch_size

    def compute_loss(self, fft_tgt, fft_prd):
        return torch.nn.functional.mse_loss(torch.abs(fft_tgt), torch.abs(fft_prd))

    def forward(self, target, pred, mask):
        # Example shapes for 512x512 inputs and patch size 32
        # target/pred: [B, 1, 512, 512]
        # mask: [B, 256] (1 for masked/hidden patches)

        p = self.patch_size

        # 1. Use rearrange to group pixels into patches.
        # This keeps the 16x16 spatial structure inside each patch
        # but aligns the number of patches with the mask (256).
        tgt_patches = rearrange(target, 'b c (h p1) (w p2) -> b (h w) c p1 p2', p1=p, p2=p)
        prd_patches = rearrange(pred, 'b c (h p1) (w p2) -> b (h w) c p1 p2', p1=p, p2=p)

        # 2. Masking
        # Since mask is [B, 256] and tgt_patches is [B, 256, C, 16, 16],
        # we can index directly. Result is [Total_Masked_Patches, C, 16, 16]
        masked_tgt = tgt_patches[mask == 1]
        masked_prd = prd_patches[mask == 1]

        # 3. FFT on the 16x16 spatial dimensions
        fft_tgt = torch.fft.rfft2(masked_tgt, norm="ortho")
        fft_prd = torch.fft.rfft2(masked_prd, norm="ortho")

        # 4. Return MSE of the magnitudes
        return self.compute_loss(fft_tgt, fft_prd)


class FFTFullLoss(FFTLoss):
    def compute_loss(self, fft_tgt, fft_prd):
        return torch.view_as_real(fft_tgt - fft_prd).pow(2).mean()


class RepeatedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, repetitions=100):
        self.dataset = dataset
        self.repetitions = repetitions

    def __getitem__(self, index):
        # The modulo operator maps the large index back to your 45k patients
        return self.dataset[index % len(self.dataset)]

    def __len__(self):
        return len(self.dataset) * self.repetitions


class DINOHead(nn.Module):
    # https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOModel(nn.Module):
    def __init__(self, args, embed_dim, device, dataloader_len):
        super().__init__()
        self.student_head = DINOHead(
            in_dim=embed_dim,
            out_dim=65536,
            use_bn=False,
            norm_last_layer=True,
        ).to(device)
        self.student_head_without_ddp = None
        teacher_args = copy.deepcopy(args)
        teacher_args.in_domains = ['bscan']
        teacher_args.out_domains = []
        self.teacher = get_model(teacher_args).to(device)
        self.teacher.eval()  # set teacher to eval mode
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher_head = DINOHead(
            in_dim=embed_dim,
            out_dim=65536,
            use_bn=False,
            norm_last_layer=True,
        ).to(device)
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        self.teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)
        self.loss_fn = DINOLoss(
            out_dim=65536,
            # total number of crops = 2 global crops + local_crops_number
            # NOTE: Here we only use global crops
            ncrops=2,
            warmup_teacher_temp=0.04,
            teacher_temp=0.07,
            warmup_teacher_temp_epochs=args.warmup_epochs,
            nepochs=args.epochs,
        ).to(device)
        self.momentum_schedule = utils.cosine_scheduler(
            0.996,
            1,
            args.epochs,
            dataloader_len,
        )

    @torch.no_grad()
    def update_teacher(self, student_model, teacher_model, iteration):
        """
        Robust EMA update using named parameters to ensure shape matching.
        """
        momentum = self.momentum_schedule[iteration]
        student_dict = {n: p for n, p in student_model.named_parameters()}
        for n, p_t in teacher_model.named_parameters():
            # Teacher params should be always a subset of student params
            p_s = student_dict[n]
            p_t.data.mul_(momentum).add_((1 - momentum) * p_s.detach().data)


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    args.in_domains = args.in_domains.split('-')
    args.out_domains = args.out_domains.split('-')
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    model = get_model(args)
    # NOTE: This forces the BatchNorm statistics to be calculated across
    #   all GPUs, removing the "local signature" the model uses to
    #   cheat in contrastive learning.
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.task_balancer == 'uncertainty':
        loss_balancer = UncertaintyWeightingStrategy(tasks=args.out_domains)
    else:
        loss_balancer = NoWeightingStrategy()

    tasks_loss_fn = {
        domain: DOMAIN_CONF[domain]['loss'](
            patch_size=tuple(args.patch_size[domain]),
            stride=DOMAIN_CONF[domain]['stride_level']
        )
        for domain in args.out_domains
    }
    if 'bscan' in args.out_domains and not args.no_fft_loss:
        tasks_loss_fn['fft_bscan'] = WeightedLoss(
            base_loss=FFTFullLoss,
            weight=5.0,
        )

    loss_configs = {
        'position': (nn.MSELoss, 0.5),
        'laterality': (nn.CrossEntropyLoss, 0.5),
        'device': (nn.CrossEntropyLoss, 0.5),
        'lesion_presence': (AsymmetricLoss, 2.0),
        'width_mm': (nn.MSELoss, 0.5),
        'thicknesses': (nn.HuberLoss, 10.0),
    }
    try:
        global_keys = set(model.global_modules.keys())
        gk_set = set(global_keys) - {'cls'}
        for key in gk_set:
            fun, weight = loss_configs[key]
            tasks_loss_fn[key] = WeightedLoss(base_loss=fun, weight=weight)
    except AttributeError:
        print("Model does not have global modules, skipping global losses")


    # Add normalized pixel loss if specified
    if args.extra_norm_pix_loss:
        tasks_loss_fn['norm_bscan'] = DOMAIN_CONF['bscan']['loss'](patch_size=args.patch_size,
                                                               stride=DOMAIN_CONF['bscan']['stride_level'],
                                                               norm_pix=True)

    data_loader_train, num_training_steps_per_epoch, log_writer = scan_data(args)

    dino_model = None
    if args.dino:
        dino_model = DINOModel(args, model.dim_tokens, device, len(data_loader_train))

    model.to(device)
    loss_balancer.to(device)
    model_without_ddp = model
    loss_balancer_without_ddp = loss_balancer
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.print_model:
        print("Model = ", model_without_ddp)
    print(f"Number of params: {n_parameters / 1e6} M")

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.blr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    # if args.distributed:
    # model = DDP(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    model = DDP(model, device_ids=[args.gpu])
    if dino_model is not None:
        dino_model.student_head = DDP(dino_model.student_head, device_ids=[args.gpu])
        dino_model.student_head_without_ddp = dino_model.student_head.module
    model_without_ddp = model.module

    # if args.distributed and args.task_balancer != 'none':
    if args.task_balancer != 'none':
        loss_balancer = DDP(loss_balancer, device_ids=[args.gpu])
        loss_balancer_without_ddp = loss_balancer.module

    to_optimize = {
        'model': model_without_ddp,
        'balancer': loss_balancer_without_ddp,
    }
    if dino_model is not None:
        to_optimize['dino_student_head'] = dino_model.student_head_without_ddp
    optimizer = create_optimizer(args, to_optimize)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        dino_model=dino_model,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # if True: # args.distributed:
        # NOTE: This is needed for the sampler to reshuffle the
        #   images for the epoch, as it uses the epoch num for
        #   doing so.
        data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch_constrastive(
            model=model,
            data_loader=data_loader_train,
            tasks_loss_fn=tasks_loss_fn,
            loss_balancer=loss_balancer,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            max_skip_norm=args.skip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_encoded_tokens=args.num_encoded_tokens,
            in_domains=args.in_domains,
            loss_on_unmasked=args.loss_on_unmasked,
            alphas=args.alphas,
            sample_tasks_uniformly=args.sample_tasks_uniformly,
            standardize_depth=args.standardize_depth,
            extra_norm_pix_loss=args.extra_norm_pix_loss,
            fp32_output_adapters=args.fp32_output_adapters.split('-'),
            contrastive=args.contrastive,
            dino_model=dino_model,
            model_without_ddp=model_without_ddp,
        )
        if log_writer is not None:
            log_writer.update({**{k: v for k, v in train_stats.items()}, 'epoch': epoch})
        if args.output_dir:
            is_checkpoint = (
                (epoch + 1) % args.save_ckpt_freq == 0
                or epoch + 1 == args.epochs
            )
            utils.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                loss_balancer=loss_balancer_without_ddp,
                epoch=epoch,
                is_checkpoint=is_checkpoint,
                dino_model=dino_model,
            )

        log_stats = {**{k: v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    with open(Path(args.output_dir) / 'training_time.txt', 'w') as f:
        f.write(total_time_str)


if __name__ == '__main__':
    if args.empty_loop:
        while True:
            time.sleep(1)
    main(args)
