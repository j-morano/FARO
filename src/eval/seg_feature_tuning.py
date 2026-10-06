import io
from argparse import Namespace
from pathlib import Path
import os
from os.path import join
import json
import argparse
import random

import numpy as np
import torch
from torchvision.utils import save_image
from skimage import io as skio
import matplotlib.pyplot as plt

from mutils.misc import fix_seeds
from seg_heads import adapter_factory
from mutils.gdice import CEGDiceLoss




def get_args():
    parser = argparse.ArgumentParser(description='Segmentation Tuning')
    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--adapter', default='convnext', type=str)
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
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Pretty print all the arguments
    print("-"*50)
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("-"*50)
    return args


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


def save_debug_images(preds, masks, _ids, save_path, epoch, min_val_lt, max_val_lt, subset):
    preds = preds.clone().detach().cpu()
    masks = masks.clone().detach().cpu()
    preds_save = remap_range(preds.argmax(dim=1, keepdim=True).float(), min_val_lt, max_val_lt)
    masks_save = remap_range(masks.unsqueeze(1).float(), min_val_lt, max_val_lt)
    images_pred_gts = torch.cat([preds_save, masks_save], dim=-2)
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



class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        targets = self.targets[index].squeeze()
        num_augs = features.shape[0]
        aug_index = random.randint(0, num_augs - 1)
        features = features[aug_index].squeeze()
        sample_dict = {
            'bscan': features,
            'semseg': targets
        }
        return sample_dict, 0, 0


def main():
    args = get_args()
    fix_seeds(args.seed)

    model_name = Path(args.results_path).stem

    data = np.load(Path(args.results_path) / 'cached_features.npz')
    features_train = data['features_train']
    targets_train = data['targets_train']
    features_val = data['features_val']
    targets_val = data['targets_val']
    features_test = data['features_test']
    targets_test = data['targets_test']
    num_classes = int(data['num_classes'])
    lookup_table = torch.tensor(data['lookup_table']).to('cuda')
    patch_size = int(data['patch_size'])
    input_size = int(data['input_size'])
    dim_tokens_enc = int(data['dim_tokens_enc'])

    train_dataset = FeatureDataset(features_train, targets_train)
    val_dataset = FeatureDataset(features_val, targets_val)
    test_dataset = FeatureDataset(features_test, targets_test)

    args.output_dir = str(Path(args.results_path) / "feature_tuning" / args.adapter / str(args.seed))
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
    model = adapter_factory[args.adapter](
        num_classes,
        image_size=(input_size, input_size),
        patch_size=[patch_size, patch_size],
        dim_tokens_enc=dim_tokens_enc,
    ).to(args.device)

    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters())
    n_tr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters:,}')
    print(f'Number of trainable params: {n_tr_parameters:,}')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
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
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    min_val_lt = 0
    max_val_lt = lookup_table.numel() - 1

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
                    num_classes,
                    include_background=args.include_bg,
                )
                train_values.append(train_value)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if epoch % 10 == 0 and iter == 0:
                    print('     ~ images', images.shape, images.dtype, images.min(), images.max())
                    print('     ~ preds', preds.shape, preds.dtype, preds.min(), preds.max())
                    print('     ~ masks', masks.shape, masks.dtype, masks.min(), masks.max())
                    save_debug_images(
                        preds, masks, ids, debug_path, epoch,
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
                        num_classes,
                        include_background=args.include_bg,
                    )
                    val_values.append(val_value)
                    if epoch % 10 == 0 and iter == debug_iter:
                        save_debug_images(
                            preds, masks, ids, debug_path, epoch,
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
                torch.save(model.state_dict(), best_checkpoint)
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
        model.load_state_dict(torch.load(best_checkpoint, map_location=args.device))
        torch.save(
            model.state_dict(),
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
        model.load_state_dict(torch.load(best_checkpoint_file, map_location=args.device))
        best_epoch = int(checkpoint_parts[1])
        best_val_value = float(checkpoint_parts[2])

    ####### TEST LOOP
    test_errors = []
    test_values = []
    predictions_path = join(args.output_dir, 'test_predictions')
    os.makedirs(predictions_path, exist_ok=True)
    predictions_debug_path = join(debug_path, 'test')
    os.makedirs(predictions_debug_path, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
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
                    num_classes,
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
                    preds, masks, ids, predictions_debug_path, best_epoch,
                    min_val_lt, max_val_lt, 'test'
                )
    print(f"  - Test error: {np.mean(test_errors):.4f}")
    print(f"  - Test mean dice: {np.mean(test_values):.4f}")



if __name__ == '__main__':
    main()
