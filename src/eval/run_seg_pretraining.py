import tqdm
from argparse import Namespace
from pathlib import Path
import os
from os.path import join
import json
import argparse
import socket

import numpy as np
import torch
from torchvision.utils import save_image

from mutils.misc import fix_seeds
from seg_heads import adapter_factory
from fm_config import fm_config_factory
from mutils.gdice import CEGDiceLoss
from mutils.dataset_folder import MultiTaskDatasetFolder
from run_seg_tuning import (
    remap_range, remove_files_recursive, ModelWrapper, compute_mean_dice
)




def get_args():
    parser = argparse.ArgumentParser(description='Segmentation Tuning')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--input_size', default=None, type=int)
    parser.add_argument('--fill', default=0, type=int)
    parser.add_argument('--num_classes', default=19, type=int)
    parser.add_argument(
        '--base_output_dir',
        default='./__output/seg',
        help='Base output directory for saving results. (default: %(default)s)',
    )
    parser.add_argument('--adapter', default='convnext', type=str)
    parser.add_argument(
        '--pool', default=None, type=str,
        help='Pooling method before the final layer. (default: %(default)s)',
    )
    parser.add_argument('--include_bg', action='store_true')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    args = parser.parse_args()
    # Pretty print all the arguments
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if socket.gethostname() == "hemingway":
        args.batch_size = 4
        # args.input_size = 128
    print("-"*50)
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("-"*50)
    return args


def save_debug_images(images, preds, masks, _ids, save_path, epoch, min_val_lt, max_val_lt, subset):
    images = images.clone().detach().cpu()
    preds = preds.clone().detach().cpu()
    masks = masks.clone().detach().cpu()
    if images.shape[1] > 1:
        images = images[:, 0:1]
    images_s = remap_range(images, 0, 1)
    preds_s = remap_range(preds.argmax(dim=1, keepdim=True).float(), min_val_lt, max_val_lt)
    masks_s = remap_range(masks.float(), min_val_lt, max_val_lt)
    images_pred_gts = torch.cat([
        images_s, preds_s, masks_s
    ], dim=-2)
    save_image(
        images_pred_gts,
        join(save_path, f"{str(epoch).zfill(3)}__{subset}__{_ids[0]}.jpg"),
    )


def main():
    args = get_args()
    fix_seeds(7)

    model_config = None
    model_name = None
    for kw in fm_config_factory.keys():
        if kw.lower() == args.weights.lower():
            model_config = fm_config_factory[kw](args)
            model_name = kw
            break
    if model_config is None:
        raise ValueError(f"Unknown model: {args.weights}")
    print(f"Using model {model_name}")

    # Initialize the model
    base_model = model_config.model
    args = model_config.args

    print("-"*50)
    print("MODEL Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("-"*50)

    args.output_dir = join(
        args.base_output_dir,
        "decoder_pretraining",
        f"{model_name}__{args.adapter}",
    )
    debug_path = join(args.output_dir, '__debug')
    # Create output directory if it doesn't exist
    os.makedirs(debug_path, exist_ok=True)
    if args.overwrite:
        print(f"Overwriting output directory: {args.output_dir}")
        remove_files_recursive(args.output_dir)
        # empty log file
        with open(join(args.output_dir, 'train_log.txt'), 'w') as f:
            f.write('')

    # Save args as json
    with open(join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Get dimensionality of the tokens dynamically
    head = adapter_factory[args.adapter](
        args.num_classes,
        image_size=(args.input_size, args.input_size),
        patch_size=[model_config.patch_size, model_config.patch_size],
        dim_tokens_enc=model_config.embed_dim,
    ).to(args.device)

    model = ModelWrapper(base_model, head).to(args.device)

    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters())
    n_tr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters:,}')
    print(f'Number of trainable params: {n_tr_parameters:,}')

    data_args = Namespace(
        filelist='../cfgs/training_databank.joblib'
    )
    if socket.gethostname() == "hemingway":
        root = '/mnt/Data/SSHFS/msc_raid_n10/FullVIBES-v2/v2_bscanlayermap'
    else:
        root = '/raid/optima/FullVIBES-v2/v2_bscanlayermap'
    train_dataset = MultiTaskDatasetFolder(
        root=root,
        tasks=['bscan', 'bscanlayermap'],
        loader=None,  # type: ignore
        args=data_args,
        transform=None,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    criterion = CEGDiceLoss(include_background=args.include_bg)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Print number of trainable parameters in the head
    n_head_parameters = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f'Number of trainable params in head: {n_head_parameters:,}')

    ####### MAIN TRAIN LOOP
    train_errors = []
    train_values = []
    model.train()
    for iter, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        sample_dict, info = batch
        ids = info['fsid']
        optimizer.zero_grad()
        for k in sample_dict:
            sample_dict[k] = sample_dict[k].to(args.device)
        images = sample_dict['bscan'].unsqueeze(1)   # [B, C, H, W]
        masks = sample_dict['bscanlayermap'].unsqueeze(1)   # [B, 1, H, W]
        # print(f"Images: {images.shape}, Masks: {masks.shape}")
        preds = model(images)
        # print(f"Preds: {preds.shape}")
        error = criterion(preds, masks.squeeze(1))
        error.backward()
        # print(f"Error: {error.item()}")
        train_errors.append(error.item())
        train_value = compute_mean_dice(
            preds,
            masks.squeeze(1),
            args.num_classes,
            include_background=args.include_bg,
        )
        train_values.append(train_value)
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()
        if iter % 1000 == 0:
            print('     ~ images', images.shape, images.dtype, images.min(), images.max())
            print('     ~ preds', preds.shape, preds.dtype, preds.min(), preds.max())
            print('     ~ masks', masks.shape, masks.dtype, masks.min(), masks.max())
            save_debug_images(
                images, preds, masks, ids, debug_path, iter,
                0, args.num_classes, 'train'
            )
            train_error_mean = np.mean(train_errors[-100:])
            train_value_mean = np.mean(train_values[-100:])
            line = f"Iter {iter}: Train error (last 100): {train_error_mean:.4f}, Train value (last 100): {train_value_mean:.4f}"
            print(line)
            # And also save to a log file
            with open(join(args.output_dir, 'train_log.txt'), 'a') as f:
                f.write(line + '\n')
            # Remove all previous checkpoints
            prev_checkpoint_path = Path(args.output_dir).glob("checkpoint__*.pth")
            for prev_checkpoint in prev_checkpoint_path:
                os.remove(prev_checkpoint)
            # Save model checkpoint
            checkpoint_path = join(args.output_dir, f"checkpoint__{str(iter).zfill(10)}.pth")
            torch.save({
                'iter': iter,
                'head_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_error_mean': train_error_mean,
                'train_value_mean': train_value_mean,
            }, checkpoint_path)


if __name__ == '__main__':
    main()
