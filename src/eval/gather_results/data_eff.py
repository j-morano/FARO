"""
Gathers raw few-shot data-efficiency results across datasets, models,
and seeds into a single CSV, without any plotting/aggregation.
Expected directory structure:
    <root>/<dataset>/<model>/feature_probing/<seed>/nshot_<num>/test_eval.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

MODEL_DISPLAY_NAMES = {
    'miragefm3d_xattn_xattn_linear_w_2avg': 'CLOSER',
    'octcube3d_60_linear_w_cls_patch': 'OCTCube',
    # add other model-key -> display-name mappings as needed
}


def get_args():
    parser = argparse.ArgumentParser(description='Gather few-shot data-efficiency results')
    parser.add_argument('--root', type=str,
        default='/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/cls_3d/SOTA_comparison_features_analysis/0',
        help='Root directory containing <dataset>/<model>/feature_probing/...'
    )
    parser.add_argument('-m', '--metrics', type=str, nargs='+', default=['MCC', 'AUROC'])
    parser.add_argument('--output_dir', type=str,
        default='/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/complete_results/data_efficiency'
    )
    return parser.parse_args()


def parse_results(root, metrics):
    root = Path(root)
    rows = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            fp_dir = model_dir / 'feature_probing'
            if not fp_dir.exists():
                continue
            for seed_dir in sorted(fp_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                for nshot_dir in sorted(seed_dir.iterdir()):
                    if not nshot_dir.is_dir():
                        continue
                    match = re.match(r'nshot_?(-?\d+)', nshot_dir.name)
                    if not match:
                        continue
                    n_per_class = int(match.group(1))
                    csv_path = nshot_dir / 'test_eval.csv'
                    if not csv_path.exists():
                        continue
                    df = pd.read_csv(csv_path)
                    best_row = df[df['Epoch'] == 'Best']
                    if len(best_row) == 0:
                        continue
                    model_label = MODEL_DISPLAY_NAMES.get(model_dir.name, model_dir.name)
                    row = {
                        'Dataset': dataset_dir.name,
                        'Model': model_label,
                        'Seed': seed_dir.name,
                        'n_per_class': n_per_class,
                    }
                    for metric in metrics:
                        if metric in best_row.columns:
                            row[metric] = best_row[metric].values[0]
                    rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = get_args()
    all_df = parse_results(args.root, args.metrics)

    if all_df.empty:
        print("No results found. Check --root path and directory structure.")
        return

    print(f"Loaded {len(all_df)} result rows.")
    for model in sorted(all_df['Model'].unique()):
        count = (all_df['Model'] == model).sum()
        print(f"Model: {model}, Rows: {count}")

    save_fn = Path(args.output_dir) / 'raw_data_efficiency_results.csv'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(save_fn, index=False)
    print(f"\nSaved to {save_fn}")


if __name__ == '__main__':
    main()
