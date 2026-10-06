import io
import os
from pathlib import Path
import glob
from os.path import join as pjoin
import shutil
import torch.distributed as dist
import gc
import multiprocessing as mp

import torch

from .dist import save_on_master
from .model import get_state_dict


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def save_model(
    args,
    epoch,
    model,
    model_without_ddp,
    optimizer,
    loss_scaler,
    loss_balancer=None,
    model_ema=None,
    is_checkpoint=False,
    dino_model=None,
):
    if dist.is_initialized():
        dist.barrier() # Sync everyone at the start

    # Only the master performs the actual I/O
    if is_main_process():
        print('Rank 0: Saving checkpoint...')
        gc.collect() # Collect garbage to free up memory before saving

        output_dir = Path(args.output_dir)
        epoch_name = str(epoch)
        checkpoint_path = output_dir / 'checkpoint-last.pth'

        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args
        }
        if loss_balancer is not None:
            to_save['loss_balancer'] = loss_balancer.state_dict()
        if dino_model is not None:
            to_save['dino_head'] = dino_model.student_head_without_ddp.state_dict()
            to_save['dino_loss_center'] = dino_model.loss_fn.center

        gc.disable()
        try:
            # save_on_master(to_save, checkpoint_path)
            torch.save(to_save, checkpoint_path)
        finally:
            gc.enable()

        if is_checkpoint:
            shutil.copyfile(checkpoint_path, output_dir / f'checkpoint-{epoch_name.zfill(3)}.pth')

        print(f"Rank 0: Finished saving {checkpoint_path}")

    # Everyone (Master and Slaves) must wait here before exiting the function
    if dist.is_initialized():
        dist.barrier()

    # else:
    #     client_state = {'epoch': epoch}
    #     if model_ema is not None:
    #         client_state['model_ema'] = get_state_dict(model_ema)
    #     model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(
    args,
    model,
    model_without_ddp,
    optimizer,
    loss_scaler,
    model_ema=None,
    best=False,
    dino_model=None,
):
    output_dir = Path(args.output_dir)
    skip_optimizer = False
    if loss_scaler is not None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            if best:
                args.resume = pjoin(output_dir, 'checkpoint-best.pth')
                assert os.path.exists(args.resume), f"Best checkpoint not found at {args.resume}"
            elif os.path.isfile (last_ckpt := pjoin(output_dir, 'checkpoint-last.pth')):
                args.resume = last_ckpt
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, f'checkpoint-{latest_ckpt:03d}.pth')
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu')
            else:
                checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            try:
                msg = model_without_ddp.load_state_dict(checkpoint['model'])
                print(msg)
            except RuntimeError:
                # del checkpoint['model']['input_adapters.bscan.pos_emb']
                # del checkpoint['model']['input_adapters.bscanlayermap.pos_emb']
                # del checkpoint['model']['input_adapters.bscanlayermap.proj.weight']
                # del checkpoint['model']['output_adapters.bscan.pos_emb']
                # del checkpoint['model']['output_adapters.bscanlayermap.pos_emb']
                # del checkpoint['model']['output_adapters.bscanlayermap.out_proj.weight']
                # del checkpoint['model']['output_adapters.bscanlayermap.out_proj.bias']
                # <<UNCOMMENT THIS WHEN CHANGING MODEL
                # for key in list(checkpoint['model'].keys()):
                #     if key.startswith('global_modules.'):
                #         del checkpoint['model'][key]
                # if 'model' in checkpoint and 'global_tokens' in checkpoint['model']:
                #     del checkpoint['model']['global_tokens']
                # msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                # print(msg)
                # UNCOMMENT THIS WHEN CHANGING MODEL>>
                # <<COMMENT THIS WHEN CHANGING MODEL
                try:
                    msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                    print(msg)
                except RuntimeError as e:
                    if 'size mismatch for global_tokens' in str(e):
                        model_global = model_without_ddp.global_tokens
                        ckpt_global = checkpoint['model']['global_tokens']
                        # Remove all global_modules.cls.head.*
                        # for key in list(checkpoint['model'].keys()):
                        #     if key.startswith('global_modules.cls.head.'):
                        #         del checkpoint['model'][key]
                        if model_global.shape[1] > ckpt_global.shape[1]:
                            # Replicate the last weight
                            last_weight = ckpt_global[:, -1:, :].clone()
                            num_extra = model_global.shape[1] - ckpt_global.shape[1]
                            extra_weights = last_weight.repeat(1, num_extra, 1)
                            checkpoint['model']['global_tokens'] = torch.cat([ckpt_global, extra_weights], dim=1)
                        else:
                            checkpoint['model']['global_tokens'] = ckpt_global[:, :model_global.shape[1], :]
                        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                        print(msg)
                        skip_optimizer = True
                    else:
                        raise e
            if dino_model is not None and 'dino_head' in checkpoint:
                msg = dino_model.student_head_without_ddp.load_state_dict(checkpoint['dino_head'], strict=True)
                # Initialize the teacher and teacher head to same
                #   params as student.
                print('Student head missing keys:', msg.missing_keys)
                msg = dino_model.teacher.load_state_dict(checkpoint['model'], strict=False)
                print('Teacher missing keys:', msg.missing_keys)
                msg = dino_model.teacher_head.load_state_dict(checkpoint['dino_head'], strict=True)
                print('Teacher head missing keys:', msg.missing_keys)
                dino_model.loss_fn.center.copy_(checkpoint['dino_loss_center'])
                print('Loaded DINO head and teacher from checkpoint!')

                # COMMENT THIS WHEN CHANGING MODEL>>
            print("Resume checkpoint %s" % args.resume)
            if not best:
                if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                    if skip_optimizer:
                        print("WARNING: Skipping optimizer state load due to detected mismatch. Starting with fresh moments.")
                    else:
                        try:
                            optimizer.load_state_dict(checkpoint['optimizer'])
                        except ValueError:
                            print("WARNING: Optimizer mismatch detected. Skipping optimizer state load and starting with fresh moments.")
                        args.start_epoch = checkpoint['epoch'] + 1
                        if hasattr(args, 'model_ema') and args.model_ema:
                            _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                        print("With optim & sched!")
                    if 'scaler' in checkpoint:
                        loss_scaler.load_state_dict(checkpoint['scaler'])
                        print("With scaler!")
                    # pass
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])
