from typing import Callable, Optional, Tuple
from copy import deepcopy
import json
import os
import sys
import hashlib
from collections import Counter
import socket
import time
import argparse
import datetime
from pathlib import Path

import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.amp.autocast_mode import autocast

from timm.loss import LabelSmoothingCrossEntropy
from torchvision import datasets

from mutils import misc
from mutils.classification import train_1_epoch, evaluate, EarlyStopping
from mutils.misc import fix_seeds, SortingHelpFormatter
from fm_config import fm_config_factory


def get_args():
    parser = argparse.ArgumentParser(
        'Retinal image classification experiments',
        add_help=True,
        formatter_class=SortingHelpFormatter
    )

    # Model parameters
    parser.add_argument(
        '--input_size', default=None, type=int,
        help='Images input size. If None, it is automatically set to 512 for'
            ' MIRAGE (default: %(default)s)',
    )
    parser.add_argument(
        '--drop_path', type=float, default=0.1, metavar='PCT',
        help='Drop path rate. (default: %(default)s)',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.05,
        help='Weight decay. (default: %(default)s)',
    )

    # Optimizer parameters
    parser.add_argument(
        '--lr', type=float, default=1e-5, metavar='LR',
        help='Learning rate. (default: %(default)s)',
    )
    parser.add_argument(
        '--layer_decay', type=float, default=0.75,
        help='Layer-wise LR decay. (default: %(default)s)',
    )
    parser.add_argument(
        '--min_lr', type=float, default=1e-8, metavar='LR',
        help='Lower LR bound for cyclic schedulers that hit 0.'
            ' (default: %(default)s)'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=10, metavar='N',
        help='Epochs to warmup LR'
    )
    parser.add_argument(
        '--smoothing', type=float, default=0.1,
        help='Label smoothing. (default: %(default)s)',
    )
    parser.add_argument(
        '--balance', action='store_true',
        help='Whether to use class-balanced sampling. (default: %(default)s)',
    )
    parser.add_argument(
        '--accum_iter', default=1, type=int,
        help='Accumulate gradient iterations (for increasing the effective'
            ' batch size under memory constraints). (default: %(default)s)'
    )

    parser.add_argument(
        '--resume', default='',
        help='Checkpoint to resume from. (default: %(default)s)',
    )
    parser.add_argument(
        '--pool', default=None, type=str,
        help='Pooling method before the final layer. (default: %(default)s)',
    )
    parser.add_argument(
        '--base_output_dir',
        default='./__output/cls',
        help='Base output directory for saving results. (default: %(default)s)',
    )

    # Data parameters
    parser.add_argument(
        '--num_workers', default=8, type=int,
        help='Number of workers for data loading. (default: %(default)s)',
    )
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes)'
            ' transfer to GPU. (default: %(default)s)',
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Training parameters
    parser.add_argument(
        '--device', default='cuda',
        help='Device to use for training / testing. (default: %(default)s)',
    )
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Seed for reproducibility. (default: %(default)s)',
    )
    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N',
        help='Start epoch. (default: %(default)s)',
    )
    parser.add_argument(
        '--batch_size', default=None, type=int,
        help='Batch size per GPU (effective batch size is batch_size *'
            ' accum_iter * # gpus). "None" for automatic calculation.'
            ' (default: %(default)s)',
    )
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument(
        '--eval', action='store_true',
        help='Wether to run only the evaluation on the test set.'
            ' (default: %(default)s)',
    )
    parser.add_argument(
        '--early_stopping_epochs', default=20, type=int,
        help='Parameter to control how many epochs to wait for the validation'
            ' loss to improve before stopping. (default: %(default)s)',
    )
    parser.add_argument(
        '--early_stopping_delta', default=0.001, type=float,
        help='Parameter to specify the minimum change in the validation metric'
            ' required to consider it an improvement. (default: %(default)s)',
    )
    parser.add_argument(
        '--early_stopping_delta_two', default=0.001, type=float,
        help='Parameter to specify the minimum change in the validation metric two'
            ' required to consider it an improvement. (default: %(default)s)',
    )
    parser.add_argument(
        '--early_start_from', default=20, type=int,
        help='Parameter to specify the epoch to start taking into account'
            ' the early stopping criteria. (default: %(default)s)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Do not run the experiment, just build the model and print the'
            ' information. (default: %(default)s)',
    )
    parser.add_argument(
        '--version', default='v1',
        help='Version of the experiment. (default: %(default)s)',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite the output directory if it exists. (default: %(default)s)',
    )
    parser.add_argument(
        '--val_metric', default='bacc', type=str,
        help='Validation metric to monitor for early stopping. (default: %(default)s)',
    )
    parser.add_argument(
        '--val_metric_two', default='loss', type=str,
        help='Second validation metric to monitor for early stopping. (default: %(default)s)',
    )
    parser.add_argument(
        '--save_predictions', action='store_true',
        help='Save test predictions. (default: %(default)s)',
    )
    parser.add_argument(
        '--fill', default=None, type=float,
        help='Fill value for affine transformations. (default: %(default)s)',
    )
    parser.add_argument(
        '--strong_aug', action='store_true',
        help='Apply strong augmentation: random affine transformations and color jitter. (default: %(default)s)',
    )
    parser.add_argument('--no_strong_aug', action='store_false', dest='strong_aug')
    parser.set_defaults(strong_aug=False)
    parser.add_argument(
        '--three_d', action='store_true',
        help='Whether to use 3D models and datasets. (default: %(default)s)',
    )
    parser.add_argument(
        '--cache_features',
        action='store_true',
    )

    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Pre-trained weights to initialise the model with. (required)',
    )
    # Dataset parameters
    required_parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory for the classification datasets. (required)',
    )
    required_parser.add_argument(
        '--data_set',
        type=str,
        required=True,
        help='Dataset directory name. (required)',
    )

    return parser.parse_args()


def process_args(args):
    hostname = socket.gethostname()
    print(f'Running on {hostname}')

    if args.data_root[-1] != '/':
        args.data_root += '/'
    args.data_path = args.data_root + args.data_set

    # Automatic number of classes calculation
    train_data_path = args.data_path + '/train'
    num_classes = 0
    for class_dir in Path(train_data_path).iterdir():
        if class_dir.is_dir():
            num_classes += 1
    num_samples = 0
    for class_dir in Path(train_data_path).iterdir():
        if class_dir.is_dir():
            num_samples += len(list(class_dir.iterdir()))
    args.num_classes = num_classes
    print(f'Number of classes: {num_classes}')
    print(f'Number of training samples: {num_samples}')

    if args.batch_size is None:
        max_size = 64
        # Automatic batch size calculation
        # Batch size is closest power of 2 to 1/10 of the dataset, with a
        #   maximum of 64.
        args.batch_size = min(max_size, 2 ** (int(round(num_samples * 0.25)).bit_length() - 1))
        if args.batch_size < 1:
            args.batch_size = 8
    print(f'Batch size: {args.batch_size}')
    if args.three_d and args.batch_size > 1:
        print(
            f'⚠️ 3D mode: converting batch_size={args.batch_size} to '
            f'accum_iter={args.batch_size}, batch_size=1'
        )
        args.accum_iter = args.batch_size
        args.batch_size = 1
    return args


def get_output_dir(args, model_name):
    # Set output directory based on some arguments
    if args.three_d:
        output_dir = args.base_output_dir.replace('cls', 'cls_3d')
    else:
        output_dir = args.base_output_dir
    if output_dir[-1] != '/':
        output_dir += '/'
    output_dir += f'{args.version}/'
    output_dir += f'{args.seed}/'
    output_dir += f'{args.data_set}/'
    output_dir += f'{model_name}'
    output_dir += '_linear'
    if args.weights is not None:
        output_dir += '_w'
    return output_dir


def build_dataset(subset, args, build_transform: Callable, augment=False):
    transform = build_transform(subset, augment)
    root = os.path.join(args.data_path, subset)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


class ThreeDDataset:
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        subset: str = 'train',
        max_slices: int = 256,
    ) -> None:
        self.root = root
        self.subset = subset
        self.samples = self.get_samples()
        self.transform = transform
        self.targets = [target for _, target in self.samples]
        self.max_slices = max_slices

    def get_samples(self):
        samples = []
        classes = sorted(Path(self.root).iterdir())
        classes_map = {cls.name: i for i, cls in enumerate(classes)}
        for cls in classes:
            for vol in sorted(cls.iterdir()):
                if vol.suffix == '.npz':
                    samples.append((vol, classes_map[cls.name]))
        return samples

    def split_to_slices(self, data):
        num_slices = data.shape[0]
        tgt_num_slices = min(num_slices, self.max_slices)
        selected_indices = np.linspace(0, num_slices - 1, tgt_num_slices, dtype=int)
        return selected_indices

    def __getitem__(self, index: int) -> Tuple:
        path, target = self.samples[index]
        sample = np.load(path)['vol']
        if sample.shape[0] > self.max_slices:
            sample = sample[self.split_to_slices(sample)]
        sample = sample[None]  # Add channel dimension
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path.name, target

    def __len__(self) -> int:
        return len(self.samples)


def build_3d_dataset(subset, args, build_transform: Callable, augment=False):
    transform = build_transform(subset, augment)
    root = os.path.join(args.data_path, subset)
    dataset = ThreeDDataset(root, transform=transform, subset=subset)
    return dataset


def main():
    args = get_args()
    fix_seeds(args.seed)

    device = torch.device(args.device)

    args = process_args(args)

    model_config = None
    model_name = None
    for kw in fm_config_factory.keys():
        if kw.lower() == args.weights.lower():
            model_config = fm_config_factory[kw](args)
            model_name = kw
            break
    if model_config is None:
        raise ValueError(f"Unknown model: {args.weights}")

    # Initialize the model
    model = model_config.model
    args = model_config.args

    args.output_dir = get_output_dir(args, model_name)

    model_config.set_requires_grad()

    model.to(device)
    # print(model)

    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters())
    n_tr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters:,}')
    print(f'Number of trainable params: {n_tr_parameters:,}')
    if n_tr_parameters > 100_000 and not args.cache_features:
        print('ERROR: More than 100k trainable parameters for linear probing. Check the model config.')
        exit(1)

    # Save args in the name of the model as a checksum and in a json file
    args_vars = vars(args).copy()
    # Remove unnecessary keys
    model_config_keys = [
        'accum_iter', 'drop_path', 'early_start_from', 'early_stopping_delta',
        'early_stopping_delta_two', 'early_stopping_epochs', 'fill', 'weights',
        'input_size', 'layer_decay', 'lr', 'min_lr', 'model',
        'affine', 'pool', 'smoothing', 'start_epoch', 'val_metric',
        'val_metric_two', 'warmup_epochs', 'weight_decay',
    ]
    for key in list(args_vars.keys()):
        if key not in model_config_keys:
            args_vars.pop(key, None)
    args_str = json.dumps(args_vars, indent=2, sort_keys=True)
    if not args.cache_features:
        args_checksum = hashlib.md5(args_str.encode('utf-8')).hexdigest()[:8]
        print(f'Args checksum: {args_checksum}')
        args.output_dir += f'_{args.pool}_{args_checksum}/'
    else:
        args.output_dir += f'_{args.pool}/'

    output_dir = Path(args.output_dir)

    # Create output directory
    print(f'> Saving to {args.output_dir}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'args.json', 'w') as f:
        f.write(args_str)

    print(f'Args:\n{args_str}')

    if (
        (output_dir / 'test_eval.csv').exists()
        and not args.overwrite
        and not args.save_predictions
    ):
        print('Experiment already run. Exiting.')
        sys.exit(0)

    if (
        (output_dir / 'predictions.npz').exists()
        and args.save_predictions
        and not args.overwrite
    ):
        print('Predictions already saved. Exiting.')
        sys.exit(0)

    if args.dry_run:
        print('Dry run. Exiting.')
        sys.exit(0)

    optimizer = model_config.get_optimizer(model)

    if socket.gethostname() == "hemingway":
        args.batch_size = min(args.batch_size, 2)

    if args.three_d:
        build_dataset_func = build_3d_dataset
    else:
        build_dataset_func = build_dataset

    dataset_train = None
    dataset_val = None
    if args.cache_features:
        c_batch_size = 1
        shuffle = False
    else:
        c_batch_size = args.batch_size
        shuffle = True
    if not args.eval:
        dataset_train = build_dataset_func(
            subset='train',
            args=args,
            build_transform=model_config.build_transform,
            augment=True
        )
        try:
            print(dataset_train.class_to_idx)
        except AttributeError:
            pass
        train_loader = DataLoader(
            dataset_train,
            shuffle=shuffle,
            batch_size=c_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of training samples: {len(dataset_train)}')

        dataset_val = build_dataset_func(
            subset='val',
            args=args,
            build_transform=model_config.build_transform,
            augment=False,
        )
        valid_loader = DataLoader(
            dataset_val,
            shuffle=False,
            batch_size=c_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of validation samples: {len(dataset_val)}')
    else:
        train_loader = None
        valid_loader = None

    if 'cross_train' not in args.data_set.lower():
        dataset_test = build_dataset_func(
            subset='test',
            args=args,
            build_transform=model_config.build_transform,
            augment=False
        )
        test_loader = DataLoader(
            dataset_test,
            shuffle=False,
            batch_size=c_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of test samples: {len(dataset_test)}')
    else:
        test_loader = None

    # if args.save_predictions:
    #     assert test_loader is not None
    #     print('Getting predictions for the best checkpoint')
    #     args.resume = f'{args.output_dir}/checkpoint-best-model.pth'
    #     misc.load_model(args=args, model=model, optimizer=None)
    #     save_path = args.output_dir
    #     test_stats = evaluate(
    #         model, test_loader, 'Best', device, args.num_classes, mode='Test',
    #         save_predictions=True, save_path=save_path
    #     )
    #     exit(0)

    def cache_features(loader, model, device, flip=True):
        all_features, all_targets = [], []
        model.eval()
        with torch.no_grad():
            for _i, batch in tqdm.tqdm(
                enumerate(loader),
                total=len(loader),
                desc='Caching features',
            ):
                images, targets = batch[0], batch[-1]
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
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
        all_sample_ids = [s[0].stem for s in loader.dataset.samples]
        return all_features, all_targets, all_sample_ids

    if args.cache_features:
        if (output_dir / 'cached_features.npz').exists() and not args.overwrite:
            try:
                with np.load(output_dir / 'cached_features.npz') as data:
                    data['features_train']
                    data['targets_train']
                    data['features_val']
                    data['targets_val']
                    data['features_test']
                    data['targets_test']
                    data['num_classes']
                print('Cached features already exist. Exiting.')
                sys.exit(0)
            except Exception as e:
                print('Error loading cached features:', e)
                print('Recomputing cached features.')
        print('IMPORTANT: Caching features for train, val and test sets')
        features_train, targets_train, ids_train = cache_features(train_loader, model, device)
        print(len(features_train), len(targets_train), len(ids_train))
        features_val, targets_val, ids_val = cache_features(valid_loader, model, device, flip=False)
        print(len(features_val), len(targets_val), len(ids_val))
        features_test, targets_test, ids_test = -1, -1, -1
        if test_loader is not None:
            features_test, targets_test, ids_test = cache_features(test_loader, model, device, flip=False)
            print(len(features_test), len(targets_test), len(ids_test))
        with open(output_dir / 'cached_features.npz', 'wb') as fp:
            np.savez(
                fp,
                features_train=features_train,
                targets_train=targets_train,
                ids_train=ids_train,
                features_val=features_val,
                targets_val=targets_val,
                ids_val=ids_val,
                features_test=features_test,
                targets_test=targets_test,
                ids_test=ids_test,
                num_classes=args.num_classes,
            )
        print(f'Cached features saved to {output_dir / "cached_features.npz"}')
        exit(0)

    if not args.eval:
        if args.balance:
            print('ℹ️ Using class-balanced loss')
            class_counts = Counter(dataset_train.targets)  # type: ignore
            total_samples = len(dataset_train)  # type: ignore
            class_weights = [
                total_samples / (args.num_classes * class_counts[i])
                for i in range(args.num_classes)
            ]
            criterion = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float).to(device)
            )
        else:
            if args.smoothing > 0.0:
                criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            else:
                criterion = torch.nn.CrossEntropyLoss()

        greater_is_better = args.val_metric != 'loss'
        greater_is_better_two = args.val_metric_two != 'loss'

        # Initialize early stopping object
        early_stopping = EarlyStopping(
            patience=args.early_stopping_epochs,
            delta=args.early_stopping_delta,
            delta_two=args.early_stopping_delta_two,
            greater_is_better=greater_is_better,
            greater_is_better_two=greater_is_better_two,
            start_from=args.early_start_from,
        )

        start_time = time.time()
        train_stats_all, val_stats_all = [], []
        best_model = argparse.Namespace()
        assert train_loader is not None
        assert valid_loader is not None
        for epoch in range(args.start_epoch, args.epochs):
            # try:
            train_stats = train_1_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                device,
                epoch,
                args=args,
            )
            # except ValueError as e:
            #     print('Early stopping')
            #     print(e)
            #     break

            train_stats_all.append(train_stats.values())

            val_stats = evaluate(
                model, valid_loader, epoch, device, args.num_classes,
                mode='Valid', args=args
            )
            assert val_stats is not None
            val_stats_all.append(val_stats.values())

            # If the validation loss has improved, save checkpoint
            # Check if early stopping criterion is met
            is_best = early_stopping(val_stats[args.val_metric], val_stats[args.val_metric_two], epoch)
            if early_stopping.early_stop:
                print(f'Early stopping @ epoch {epoch}')
                break
            else:
                if is_best and args.output_dir:
                    # Save in memory to avoid writing to disk all the time
                    best_model= argparse.Namespace(
                        # NOTE: Pass model and optimizer state_dicts as
                        #   values (copies), not as references.
                        model=deepcopy(model.state_dict()),
                        optimizer=deepcopy(optimizer.state_dict()),
                        epoch=epoch,
                    )
                    # misc.save_model(args, epoch, model, optimizer)
                    print(
                        f'New best {model_config.__class__.__name__} model'
                        f' on {args.data_set} with seed {args.seed}'
                        f' @ epoch {epoch}'
                        f'\n\t({early_stopping.best_value}, {early_stopping.best_value_two})'
                    )

        trainable_param_names = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        misc.save_model(
            args,
            epoch=best_model.epoch,
            model=best_model.model,
            optimizer=best_model.optimizer,
            trainable_param_names=trainable_param_names,
        )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # Save evaluation results
        pd.DataFrame(
            data=train_stats_all,
            columns=['Epoch', 'Loss', 'BAcc', 'F1-score']  # type: ignore
        ).to_csv(f'{args.output_dir}/train_eval.csv', index=False)


        pd.DataFrame(
            data=val_stats_all,
            columns=['Epoch', 'Loss', 'BAcc', 'AUROC', 'AP', 'F1-score', 'MCC'],  # type: ignore
        ).to_csv(f'{args.output_dir}/valid_eval.csv', index=False)

    if test_loader is not None:
        # Evaluate on the best checkpoint
        args.resume = f'{args.output_dir}/checkpoint-best-model.pth'
        # Load model from the best_model in-memory checkpoint
        model.load_state_dict(best_model.model)
        test_stats = evaluate(
            model, test_loader, 'Best', device, args.num_classes, mode='Test',
            save_predictions=True, save_path=args.output_dir
        )
        assert test_stats is not None
        pd.DataFrame(
            data=[test_stats.values()],
            columns=['Epoch', 'Loss', 'BAcc', 'AUROC', 'AP', 'F1-score', 'MCC'],  # type: ignore
        ).to_csv(f'{args.output_dir}/test_eval.csv', index=False)



if __name__ == '__main__':
    main()
