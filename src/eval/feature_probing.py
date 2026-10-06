from pathlib import Path
import argparse
import random
from collections import Counter
import time
import datetime
from copy import deepcopy

import numpy as np
from torch import nn
import torch
import pandas as pd

from mutils.classification import train_1_epoch, evaluate, EarlyStopping
from mutils import misc




class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets, ids):
        self.features = features
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        targets = self.targets[index].squeeze()
        ids = self.ids[index]
        num_augs = features.shape[0]
        aug_index = random.randint(0, num_augs - 1)
        features = features[aug_index].squeeze()
        return features, targets, ids


class AttentionPool(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        # Adding a Tanh activation increases capacity
        self.attention = nn.Linear(dim, 1)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x shape: [B, N, D]
        scores = self.attention(x)            # [B, N, 1]
        weights = torch.softmax(scores, dim=1) # [B, N, 1]
        out = (x * weights).sum(dim=1)         # [B, D]
        return self.head(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True)
    # parser.add_argument("-m", "--model", type=str, required=True)
    # parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--no_balance", action="store_false", dest="balance")
    parser.set_defaults(balance=True)
    parser.add_argument("--val_metric", type=str, default='mcc')
    parser.add_argument("--val_metric_two", type=str, default='loss')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        '--accum_iter', default=1, type=int,
        help='Accumulate gradient iterations (for increasing the effective'
            ' batch size under memory constraints). (default: %(default)s)'
    )
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3,)
    parser.add_argument(
        '--min_lr', type=float, default=1e-8, metavar='LR',
        help='Lower LR bound for cyclic schedulers that hit 0.'
            ' (default: %(default)s)'
    )
    parser.add_argument(
        '--attn', action='store_true',
        help='Use attention-based pooling instead of Linear.'
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results.')
    parser.add_argument(
        '--n_per_class', type=int, default=None,
        help='Number of training examples to sample per class (for '
             'few-shot data efficiency analysis). If None, uses all data.',
    )
    args = parser.parse_args()

    misc.fix_seeds(args.seed)

    dataset_name = Path(args.root).parent.stem
    model_name = Path(args.root).stem
    features_fn = (
        Path(args.root) / "cached_features.npz"
    )
    assert features_fn.exists(), features_fn

    args.output_dir = features_fn.parent / "feature_probing" / str(args.seed)
    if args.n_per_class is not None:
        args.output_dir = args.output_dir / f"nshot_{args.n_per_class}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = f"{model_name}|{dataset_name}|{args.seed}"
    if len(list(args.output_dir.iterdir())) == 5:
        if args.overwrite:
            print(f"Overwriting experiment '{experiment_name}'.")
            for file in args.output_dir.iterdir():
                file.unlink()
        else:
            print(f"Skipping experiment '{experiment_name}' because it has already been run.")
            exit(0)
    else:
        print(f"Running experiment '{experiment_name}'.")

    features = np.load(features_fn)
    features_train = torch.tensor(features["features_train"])
    targets_train = torch.tensor(features["targets_train"])
    ids_train = features["ids_train"]
    features_val = torch.tensor(features["features_val"])
    targets_val = torch.tensor(features["targets_val"])
    ids_val = features["ids_val"]
    features_test = torch.tensor(features["features_test"])
    targets_test = torch.tensor(features["targets_test"])
    ids_test = features["ids_test"]
    num_classes = features["num_classes"].item()
    # print(features_train.shape, targets_train.shape)
    # print(features_val.shape, targets_val.shape)
    # print(features_test.shape, targets_test.shape)
    # print(f"Number of classes: {num_classes}")


    if args.n_per_class is not None:
        rng = np.random.default_rng(args.seed)
        targets_flat = targets_train.squeeze(-1).numpy()
        selected_indices = []
        for c in range(num_classes):
            class_indices = np.where(targets_flat == c)[0]
            n_available = len(class_indices)
            n_select = min(args.n_per_class, n_available)
            if n_select < args.n_per_class:
                print(f'WARNING: class {c} has only {n_available} samples, '
                      f'fewer than requested n_per_class={args.n_per_class}.')
            chosen = rng.choice(class_indices, size=n_select, replace=False)
            selected_indices.extend(chosen.tolist())
        selected_indices = np.array(selected_indices)
        rng.shuffle(selected_indices)  # avoid class-ordered batches

        features_train = features_train[selected_indices]
        targets_train = targets_train[selected_indices]
        ids_train = ids_train[selected_indices]
        print(f'Using {args.n_per_class} examples/class '
              f'({len(selected_indices)} total) for few-shot analysis.')



    train_ds = FeatureDataset(features_train, targets_train, ids_train)
    val_ds = FeatureDataset(features_val, targets_val, ids_val)
    test_ds = FeatureDataset(features_test, targets_test, ids_test)
    num_samples = len(train_ds)
    max_size = 64
    # Automatic batch size calculation
    # Batch size is closest power of 2 to 1/10 of the dataset, with a
    #   maximum of 64.
    train_batch_size = min(max_size, 2 ** (int(round(num_samples * 0.25)).bit_length() - 1))
    train_batch_size = max(1, min(train_batch_size, num_samples))  # clamp to [1, num_samples]
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
    )
    # Get first item of the training loader
    feature_shape = next(iter(train_loader))[0].shape
    print(f"Feature shape: {feature_shape}")
    # exit(0)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if args.attn:
        model = AttentionPool(feature_shape[-1], num_classes)
    else:
        model = nn.Linear(feature_shape[-1], num_classes)
    # Print number of parameters of the model
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2,
    )

    if args.balance:
        print('ℹ️ Using class-balanced loss')
        class_counts = Counter(train_ds.targets.squeeze(-1).numpy().tolist())
        print(class_counts)
        total_samples = len(train_ds)
        class_weights = [
            total_samples / (num_classes * class_counts[i])
            for i in range(num_classes)
        ]
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(device)
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()

    greater_is_better = args.val_metric != 'loss'
    greater_is_better_two = args.val_metric_two != 'loss'
    early_stopping = EarlyStopping(
        patience=20,
        delta=0.001,
        delta_two=0.001,
        greater_is_better=greater_is_better,
        greater_is_better_two=greater_is_better_two,
        start_from=20,
    )

    start_time = time.time()
    train_stats_all, val_stats_all = [], []
    best_model = argparse.Namespace()
    assert train_loader is not None
    assert val_loader is not None
    for epoch in range(0, args.epochs):
        # try:
        train_stats = train_1_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            args=args,
            debug=False,
        )
        # except ValueError as e:
        #     print('Early stopping')
        #     print(e)
        #     break

        train_stats_all.append(train_stats.values())

        val_stats = evaluate(
            model, val_loader, epoch, device, num_classes,
            mode='Valid', args=args, debug=False,
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
                    f'New best {model_name} model'
                    f' on {dataset_name} with seed {args.seed}'
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
        columns=['Epoch', 'Loss', 'BAcc', 'F1-score']
    ).to_csv(f'{args.output_dir}/train_eval.csv', index=False)


    pd.DataFrame(
        data=val_stats_all,
        columns=['Epoch', 'Loss', 'BAcc', 'AUROC', 'AP', 'F1-score', 'MCC'],
    ).to_csv(f'{args.output_dir}/valid_eval.csv', index=False)

    if test_loader is not None:
        # Evaluate on the best checkpoint
        args.resume = f'{args.output_dir}/checkpoint-best-model.pth'
        # Load model from the best_model in-memory checkpoint
        model.load_state_dict(best_model.model)
        test_stats = evaluate(
            model, test_loader, 'Best', device, num_classes, mode='Test',
            save_predictions=True, save_path=args.output_dir
        )
        assert test_stats is not None
        pd.DataFrame(
            data=[test_stats.values()],
            columns=['Epoch', 'Loss', 'BAcc', 'AUROC', 'AP', 'F1-score', 'MCC'],
        ).to_csv(f'{args.output_dir}/test_eval.csv', index=False)



if __name__ == "__main__":
    main()
