from argparse import Namespace
import io
from pathlib import Path
import os
from typing import Dict
from os.path import join
import json
import argparse
import socket
import random

import numpy as np
import torch
from torch import nn
from torch import autocast
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.utils import save_image
from skimage import io as skio
import matplotlib.pyplot as plt
import tqdm

from mutils.misc import fix_seeds
from seg_heads import adapter_factory
from fm_config import fm_config_factory
from mutils.gdice import CEGDiceLoss
from mutils.dataset_folder import MultiTaskImageFolder




def get_args():
    parser = argparse.ArgumentParser(description='Segmentation Tuning')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data_set', type=str, required=True)
    parser.add_argument('--version', type=str, default="v0")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input_size', default=None, type=int)
    parser.add_argument(
        '--fill', default=None, type=float,
        help='Fill value for affine transformations. (default: %(default)s)',
    )
    parser.add_argument(
        '--base_output_dir',
        default='./__output/seg',
        help='Base output directory for saving results. (default: %(default)s)',
    )
    parser.add_argument('--adapter', default='convnext', type=str)
    parser.add_argument('--all_domains', default='bscan-semseg')
    parser.add_argument(
        '--pool', default=None, type=str,
        help='Pooling method before the final layer. (default: %(default)s)',
    )
    parser.add_argument('--include_bg', action='store_true')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--infer_only', action='store_true')
    parser.add_argument('--test_data', default=None, type=str)
    parser.add_argument('--cache_features', action='store_true')
    parser.add_argument('--pretrained_weights', default=None, type=str)
    parser.add_argument('--skip_test', action='store_true')
    args = parser.parse_args()
    # Pretty print all the arguments
    with open(join(args.data_set, 'INFO.json'), 'r') as f:
        original_mapping = json.load(f)
    args.num_classes = len(original_mapping)
    mapping = {}
    for k, v in original_mapping.items():
        mapping[v['value']] = int(k)
    args.mapping = mapping
    args.inverse_mapping = {v: k for k, v in args.mapping.items()}
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if socket.gethostname() == "hemingway":
        args.batch_size = 4
        # args.input_size = 128
    fast_path = Path("/raid/optima/Segmentation") / Path(args.data_set).stem
    if fast_path.exists():
        print(f"Using fast path for dataset: {Path(args.data_set).stem}")
        args.data_set = str(fast_path)
    print("-"*50)
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("-"*50)
    return args


def get_lookup_table(mapping: Dict[int, int], device: torch.device) -> torch.Tensor:
    # Create a lookup table
    max_key = max(mapping.keys())  # Get the largest key in the dictionary
    lookup_table = torch.full((max_key + 1,), -1)  # Initialize with default values
    for key, value in mapping.items():
        lookup_table[key] = value  # Fill the lookup table
    lookup_table = lookup_table.to(device)
    return lookup_table


def get_output_dir(args, model_name):
    # Set output directory based on some arguments
    output_dir = args.base_output_dir
    if output_dir[-1] != '/':
        output_dir += '/'
    output_dir += f'{args.version}/'
    output_dir += f'{args.seed}/'
    output_dir += f'{Path(args.data_set).stem}/'
    output_dir += f'{model_name}'
    output_dir += f'_{args.adapter}'
    if args.weights is not None:
        output_dir += '_w'
    if args.pretrained_weights is not None:
        output_dir += '_pw'
    return output_dir


def build_simple_transform(train: bool, model_config, input_size: int = 512):
    tforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    img_only_tforms = v2.Compose(tforms)

    if train:
        init_size = int(input_size * 1.1)
        spatial_tforms = v2.Compose([
            v2.Resize(
                (init_size, init_size),
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomAffine(
            #     degrees=(-5, 5),          # small rotation
            #     translate=(0.05, 0.05),   # 5% translation
            #     scale=(0.9, 1.1),         # 10% scale variation
            #     shear=0,                  # no shear for OCT
            #     interpolation=v2.InterpolationMode.BILINEAR,
            #     fill=0,
            # ),
            v2.RandomCrop((input_size, input_size)),
        ])
        # Intensity augmentation — image only, not mask
        intensity_tforms = [
            v2.ColorJitter(
                brightness=0.2,   # ±20% brightness
                contrast=0.2,     # ±20% contrast
                saturation=0.0,   # grayscale, irrelevant
                hue=0.0,
            ),
        ]
    else:
        spatial_tforms = v2.Compose([
            v2.Resize(
                (input_size, input_size),
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            ),
        ])
        intensity_tforms = []
    intensity_tforms += model_config.get_color_norm()
    intensity_tforms += model_config.get_model_norm()
    intensity_tforms = v2.Compose(intensity_tforms)

    def transform(task_dict):
        bscan = task_dict['bscan']
        image = img_only_tforms(bscan)

        if 'semseg' in task_dict:
            mask = task_dict['semseg']
            image = tv_tensors.Image(image)
            mask = tv_tensors.Mask(
                torch.as_tensor(mask).unsqueeze(0).long()
            )
            # Spatial transforms applied consistently to both
            image, mask = spatial_tforms(image, mask)
            task_dict['semseg'] = mask.squeeze(0).long()
        else:
            image = spatial_tforms(image)

        # Intensity transforms applied to image only, after spatial
        if intensity_tforms is not None:
            image = intensity_tforms(image)

        task_dict['bscan'] = image.float()
        return task_dict

    return transform


def remap_range(x, src_min=None, src_max=None, tgt_min=0.0, tgt_max=1.0, eps=1e-8):
    """Remap x from [src_min, src_max] to [tgt_min, tgt_max]."""
    if src_min is None:
        src_min = x.min()
    if src_max is None:
        src_max = x.max()
    x = x.clone()  # Avoid modifying the original tensor
    x = (x - src_min) / (src_max - src_min + eps)  # normalize to [0, 1]
    x = x * (tgt_max - tgt_min) + tgt_min           # scale to [tgt_min, tgt_max]
    return x.clamp(tgt_min, tgt_max)


def save_debug_images(images, preds, masks, _ids, save_path, epoch, min_val_lt, max_val_lt, subset):
    images = images.clone().detach().cpu()
    preds = preds.clone().detach().cpu()
    masks = masks.clone().detach().cpu()
    if images.shape[1] > 1:
        images = images[:, 0:1]
    images_pred_gts = torch.cat([
        remap_range(images, 0, 1),
        remap_range(preds.argmax(dim=1, keepdim=True).float(), min_val_lt, max_val_lt),
        remap_range(masks.unsqueeze(1).float(), min_val_lt, max_val_lt),
    ], dim=-2)
    save_image(
        images_pred_gts,
        join(save_path, f"{str(epoch).zfill(3)}__{subset}__{_ids[0]}.png"),
    )


def compute_mean_dice(preds, masks, num_classes, include_background=True):
    """
    preds: [B, C, H, W] logits
    masks: [B, H, W] integer labels
    """
    preds = preds.argmax(dim=1)  # [B, H, W]
    dice_scores = []
    start = 0 if include_background else 1
    for c in range(start, num_classes):
        pred_c = (preds == c).float()
        mask_c = (masks == c).float()
        intersection = (pred_c * mask_c).sum()
        union = pred_c.sum() + mask_c.sum()
        if union == 0:
            # Class not present in this batch — skip
            continue
        dice_scores.append((2 * intersection / (union + 1e-8)).item())
    return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0


def remove_files_recursive(path):
    """Remove all the files (leaves of the tree), not the dirs under
    the given path.
    """
    for item in Path(path).rglob('*'):
        if item.is_file():
            item.unlink()



class ModelWrapper(nn.Module):
    def __init__(self, model, head, freeze_model=True):
        super().__init__()
        self.model = model
        self.head = head
        # Freeze the model parameters
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, return_features=False):
        with torch.no_grad(), autocast('cuda', enabled=True):
            features = self.model(x, return_features=True)
        if not isinstance(features, tuple):
            features = features.float()
        # print(f"Features: {features.shape}")
        if return_features:
            return features
        else:
            return self.head(features)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()  # backbone always stays in eval
        return self


def save_plots(plot_values, save_path, log_scale=True):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(plot_values.train_errors, label='Train Loss')
    plt.plot(plot_values.val_errors, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(plot_values.train_values, label='Train Mean Dice')
    plt.plot(plot_values.val_values, label='Val Mean Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Dice Score')
    plt.title('Training and Validation Mean Dice Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def cache_features(loader, model, device, flip=True):
    all_features, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for _i, batch in tqdm.tqdm(
            enumerate(loader),
            total=len(loader),
            desc='Caching features',
        ):
            sample_dict, _target, ids = batch
            for k in sample_dict:
                sample_dict[k] = sample_dict[k].to(device)
            images = sample_dict['bscan']   # [B, C, H, W]
            targets = sample_dict['semseg']   # [B, 1, H, W]
            # print(f"Images: {images.shape}, Masks: {masks.shape}")
            with autocast('cuda', enabled=True):
                features = []
                c_features = model(images, return_features=True).cpu()
                # print(f'Batch {_i}: features shape {c_features.shape}')
                features.append(c_features)
                if flip:
                    features.append(model(
                        torch.flip(images, dims=[-1]),
                        return_features=True,
                    ).cpu())
                all_features.append(features)
                all_targets.append(targets.cpu())
    return all_features, all_targets

def main():
    args = get_args()
    fix_seeds(args.seed)

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

    args.output_dir = get_output_dir(args, model_name)
    debug_path = join(args.output_dir, '__debug')
    # Create output directory if it doesn't exist
    os.makedirs(debug_path, exist_ok=True)
    if args.overwrite:
        print(f"Overwriting output directory: {args.output_dir}")
        remove_files_recursive(args.output_dir)
    else:
        # Search for the checkpoint; if there, skip, as it was already
        # trained and saved.
        if not args.test_only and any(Path(args.output_dir).glob('best_head__*.pth')):
            print(f"Checkpoint found in {args.output_dir}. Skipping training.")
            exit(0)

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
    if args.adapter.startswith('xattn'):
        pretrained_fn = '_weights/ablation/v5_multimae-b-pbnofftLeOnly_64_bscan-to-bscan-bscanlayermap_512-128--32-8_shuffle_014.pth'
        weights = torch.load(pretrained_fn, map_location=args.device, weights_only=False)
        pretrained_sd = {
            k.replace('output_adapters.bscanlayermap.', ''): v
            for k, v in weights['model'].items()
            if 'bscanlayermap' in k
        }
        # delete out_proj.weight and out_proj.bias
        for k in ['out_proj.weight', 'out_proj.bias']:
            if k in pretrained_sd:
                del pretrained_sd[k]
        msg = head.load_state_dict(pretrained_sd, strict=False)
        print(msg)  # should show only out_proj as missing/mismatched, everything else loaded

    model = ModelWrapper(base_model, head).to(args.device)

    if args.pretrained_weights is not None:
        print(f"Loading pretrained weights from {args.pretrained_weights}")
        pretrained_sd = torch.load(args.pretrained_weights, map_location=args.device, weights_only=False)['head_state_dict']
        for k in ['head.final_layer.weight', 'head.final_layer.bias']:
            if k in pretrained_sd:
                del pretrained_sd[k]
        msg = model.load_state_dict(pretrained_sd, strict=False)
        print(msg)  # should show only out_proj as missing/mismatched, everything else loaded

    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters())
    n_tr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters:,}')
    print(f'Number of trainable params: {n_tr_parameters:,}')

    if args.cache_features:
        shuffle = False
        c_batch_size = 1
        is_train = False
    else:
        shuffle = True
        is_train = True
        c_batch_size = args.batch_size

    train_dataset = MultiTaskImageFolder(
        root=join(args.data_set, 'train'),
        tasks=args.all_domains.split('-'),
        args=args,
        transform=build_simple_transform(
            train=is_train,
            model_config=model_config,
            input_size=args.input_size,
        )
    )
    val_dataset = MultiTaskImageFolder(
        root=join(args.data_set, 'val'),
        tasks=args.all_domains.split('-'),
        args=args,
        transform=build_simple_transform(
            train=False,
            model_config=model_config,
            input_size=args.input_size,
        )
    )
    tasks = args.all_domains.split('-')
    if args.test_data is not None:
        if Path(args.test_data).exists():
            print(f"Using test data from {args.test_data}")
            test_data_path = args.test_data
            predictions_path = join(args.output_dir, f'{Path(args.test_data).stem}_predictions')
            tasks = ['bscan']
        elif Path(join(args.data_set, args.test_data)).exists():
            print(f"Using test data from {join(args.data_set, args.test_data)}")
            test_data_path = join(args.data_set, args.test_data)
            predictions_path = join(args.output_dir, f'{args.test_data}_predictions')
        else:
            raise ValueError(f"Test data path {args.test_data} does not exist.")
    else:
        test_data_path = join(args.data_set, 'test')
        predictions_path = join(args.output_dir, 'test_predictions')
    test_dataset = MultiTaskImageFolder(
        root=test_data_path,
        tasks=tasks,
        args=args,
        transform=build_simple_transform(
            train=False,
            model_config=model_config,
            input_size=args.input_size,
        )
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=c_batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=c_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
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

    lookup_table = get_lookup_table(args.inverse_mapping, device=args.device)
    print(f"Lookup table: {lookup_table}")
    min_val_lt = 0
    max_val_lt = lookup_table.numel() - 1

    if args.cache_features:
        if (Path(args.output_dir) / 'cached_features.npz').exists() and not args.overwrite:
            try:
                with np.load(args.output_dir / 'cached_features.npz') as data:
                    data['features_train']
                    data['targets_train']
                    data['features_val']
                    data['targets_val']
                    data['features_test']
                    data['targets_test']
                    data['num_classes']
                print('Cached features already exist. Exiting.')
                exit(0)
            except Exception as e:
                print('Error loading cached features:', e)
                print('Recomputing cached features.')
        print('IMPORTANT: Caching features for train, val and test sets')
        features_train, targets_train = cache_features(train_loader, model, args.device, flip=True)
        print(len(features_train), len(targets_train))
        features_val, targets_val = cache_features(val_loader, model, args.device, flip=False)
        print(len(features_val), len(targets_val))
        features_test, targets_test = -1, -1
        if test_loader is not None:
            features_test, targets_test = cache_features(test_loader, model, args.device, flip=False)
            print(len(features_test), len(targets_test))
        with open(Path(args.output_dir) / 'cached_features.npz', 'wb') as fp:
            np.savez(
                fp,
                features_train=features_train,
                targets_train=targets_train,
                features_val=features_val,
                targets_val=targets_val,
                features_test=features_test,
                targets_test=targets_test,
                num_classes=args.num_classes,
                lookup_table=lookup_table.cpu().numpy(),
                patch_size=model_config.patch_size,
                input_size=args.input_size,
                dim_tokens_enc=model_config.embed_dim,
            )
        print(f'Cached features saved to {Path(args.output_dir) / "cached_features.npz"}')
        exit(0)

    ####### MAIN TRAIN LOOP
    if not args.test_only:
        best_val_value = -1000
        best_epoch = -1
        best_checkpoint = None
        patience_glass = 0
        plot_values = Namespace(
            train_errors=[],
            train_values=[],
            val_errors=[],
            val_values=[],
        )
        for epoch in range(0, args.epochs):
            print(f"# Epoch {epoch} ({model_name})")
            train_errors = []
            train_values = []
            model.train()
            for iter, batch in enumerate(train_loader):
                sample_dict, _target, ids = batch
                optimizer.zero_grad()
                for k in sample_dict:
                    sample_dict[k] = sample_dict[k].to(args.device)
                images = sample_dict['bscan']   # [B, C, H, W]
                masks = sample_dict['semseg']   # [B, 1, H, W]
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
                if epoch % 10 == 0 and iter == 0:
                    print('     ~ images', images.shape, images.dtype, images.min(), images.max())
                    print('     ~ preds', preds.shape, preds.dtype, preds.min(), preds.max())
                    print('     ~ masks', masks.shape, masks.dtype, masks.min(), masks.max())
                    save_debug_images(
                        images, preds, masks, ids, debug_path, epoch,
                        min_val_lt, max_val_lt, 'train'
                    )
            mean_train_error = np.mean(train_errors)
            mean_train_value = np.mean(train_values)
            print(f"  - Train error: {mean_train_error:.4f}")
            print(f"  - Train mean dice: {mean_train_value:.4f}")
            plot_values.train_errors.append(mean_train_error)
            plot_values.train_values.append(mean_train_value)


            #### VAL LOOP
            val_errors = []
            val_values = []
            debug_iter = random.randint(0, len(val_loader)-1)
            model.eval()
            with torch.no_grad():
                for iter, batch in enumerate(val_loader):
                    sample_dict, _target, ids = batch
                    for k in sample_dict:
                        sample_dict[k] = sample_dict[k].to(args.device)
                    images = sample_dict['bscan']   # [B, C, H, W]
                    masks = sample_dict['semseg']   # [B, 1, H, W]
                    preds = model(images)
                    # print(f"Val Preds: {preds.shape}")
                    val_error = criterion(preds, masks.squeeze(1))
                    # print(f"Val error: {val_error.item():.4f}")
                    val_errors.append(val_error.item())
                    val_value = compute_mean_dice(
                        preds,
                        masks.squeeze(1),
                        args.num_classes,
                        include_background=args.include_bg,
                    )
                    val_values.append(val_value)
                    if epoch % 10 == 0 and iter == debug_iter:
                        save_debug_images(
                            images, preds, masks, ids, debug_path, epoch,
                            min_val_lt, max_val_lt, 'val'
                        )
            mean_val_value = np.mean(val_values)
            if mean_val_value > best_val_value:
                print(f"  New best model found at epoch {epoch}.")
                # Save checkpoint to a temp file in memory
                patience_glass = 0
                best_val_value = mean_val_value
                best_epoch = epoch
                best_checkpoint = io.BytesIO()
                torch.save(head.state_dict(), best_checkpoint)
            else:
                patience_glass += 1
            print(f"  - Val error: {np.mean(val_errors):.4f}")
            print(f"  - Val mean dice: {np.mean(val_values):.4f}")
            mean_val_error = np.mean(val_errors)
            mean_val_value = np.mean(val_values)
            plot_values.val_errors.append(mean_val_error)
            plot_values.val_values.append(mean_val_value)
            save_plots(plot_values, join(args.output_dir, 'training_plots.png'))
            if patience_glass >= args.patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation.")
                break

        # Now load best head
        print("Training finished. Loading best head from checkpoint for testing...")
        if best_checkpoint is None:
            raise RuntimeError("No best checkpoint found. Training may have failed.")
        best_checkpoint.seek(0) # Reset the buffer position to the beginning
        head.load_state_dict(torch.load(best_checkpoint, map_location=args.device))
        torch.save(
            head.state_dict(),
            join(args.output_dir, f'best_head__{str(best_epoch).zfill(3)}__{best_val_value:.4f}.pth')
        )
    else:
        # Load the best head from the output directory
        checkpoint_files = list(Path(args.output_dir).glob('best_head__*.pth'))
        if not checkpoint_files:
            raise RuntimeError("No checkpoint found in output directory for testing.")
        # Sort by epoch number extracted from filename
        best_checkpoint_file = checkpoint_files[0]
        checkpoint_parts = best_checkpoint_file.stem.split('__')
        print(f"Loading best head from {best_checkpoint_file} for testing...")
        head.load_state_dict(torch.load(best_checkpoint_file, map_location=args.device))
        best_epoch = int(checkpoint_parts[1])
        best_val_value = float(checkpoint_parts[2])

    ####### TEST LOOP
    if not args.skip_test:
        test_errors = []
        test_values = []
        os.makedirs(predictions_path, exist_ok=True)
        predictions_debug_path = join(debug_path, 'test')
        os.makedirs(predictions_debug_path, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader, desc='Testing', total=len(test_loader)):
                sample_dict, _target, ids = batch
                for k in sample_dict:
                    sample_dict[k] = sample_dict[k].to(args.device)
                images = sample_dict['bscan']   # [B, C, H, W]
                preds = model(images)
                if not args.infer_only:
                    masks = sample_dict['semseg']   # [B, 1, H, W]
                    test_error = criterion(preds, masks.squeeze(1))
                    test_errors.append(test_error.item())
                    test_value = compute_mean_dice(
                        preds,
                        masks.squeeze(1),
                        args.num_classes,
                        include_background=args.include_bg,
                    )
                    test_values.append(test_value)
                # Save predictions
                pred = preds.argmax(dim=1).squeeze(0).cpu().numpy()  # [H, W]
                pred = lookup_table[pred].cpu().numpy().astype(np.uint8)
                skio.imsave(
                    join(predictions_path, f"{ids[0]}.png"),
                    pred,
                    check_contrast=False
                )
                if not args.infer_only:
                    save_debug_images(
                        images, preds, masks, ids, predictions_debug_path, best_epoch,
                        min_val_lt, max_val_lt, 'test'
                    )
        print(f"  - Test error: {np.mean(test_errors):.4f}")
        print(f"  - Test mean dice: {np.mean(test_values):.4f}")



if __name__ == '__main__':
    main()
