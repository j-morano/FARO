"""
Stratify B-scan-level Dice and HD952D by ground-truth lesion size.
"""
import argparse
import numpy as np
import pandas as pd
from mutils.misc import SortingHelpFormatter


def get_args():
    parser = argparse.ArgumentParser(
        'Stratify B-scan Dice/HD952D by GT lesion size',
        formatter_class=SortingHelpFormatter,
    )
    parser.add_argument('-r', '--results_csvs', type=str, nargs='+', required=True)
    parser.add_argument('-n', '--names', type=str, nargs='+', required=True)
    parser.add_argument('--n_bins', type=int, default=16)
    parser.add_argument('--min_n', type=int, default=5)
    parser.add_argument('--out', type=str, default='lesion_size_stratified.csv')
    parser.add_argument(
        '--bin_strategy', type=str, default='quantile', choices=['quantile', 'log'],
        help='"quantile": equal sample count per bin (current default).'
             ' "log": equal width in log-space, concentrating more bins at'
             ' small lesion sizes. (default: %(default)s)'
    )
    return parser.parse_args()


def get_shared_bin_edges(df, n_bins, strategy='quantile'):
    nonzero = df[df['GT_Area'] > 0]
    if strategy == 'quantile':
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(nonzero['GT_Area'].quantile(quantiles).values)
    elif strategy == 'log':
        lo = nonzero['GT_Area'].min()
        hi = nonzero['GT_Area'].max()
        edges = np.unique(np.geomspace(lo, hi, n_bins + 1))
    else:
        raise ValueError(f'Unknown bin strategy: "{strategy}"')
    return edges


def summarize(group):
    return pd.Series({
        'N': len(group),
        'Dice_mean': group['Dice'].mean(),
        'Dice_std': group['Dice'].std(),
        'HD952D_mean': np.nanmean(group['HD952D']),
        'HD952D_std': np.nanstd(group['HD952D']),
        'HD952D_N': group['HD952D'].notna().sum(),  # HD95 can be NaN more often than Dice
    })


def stratify_one_model(df, model_name, bin_edges, min_n):
    no_lesion = df[df['GT_Area'] == 0]
    has_lesion = df[df['GT_Area'] > 0].copy()
    has_lesion['SizeBin'] = pd.cut(has_lesion['GT_Area'], bins=bin_edges, include_lowest=True)

    rows = [dict(Model=model_name, SizeBin='No lesion', **summarize(no_lesion))]
    for bin_val, group in has_lesion.groupby('SizeBin', observed=True):
        rows.append(dict(Model=model_name, SizeBin=str(bin_val), **summarize(group)))

    summary = pd.DataFrame(rows)
    dropped = summary[summary['N'] < min_n]
    if len(dropped):
        print(f'[{model_name}] Dropping bins with N < {min_n}:')
        print(dropped[['SizeBin', 'N']].to_string(index=False))
    return summary[summary['N'] >= min_n].reset_index(drop=True)


def main():
    args = get_args()
    dfs = [pd.read_csv(p) for p in args.results_csvs]
    bin_edges = get_shared_bin_edges(dfs[0], args.n_bins, strategy=args.bin_strategy)
    print(f'Shared bin edges: {bin_edges}')

    all_summaries = [
        stratify_one_model(df, name, bin_edges, args.min_n)
        for df, name in zip(dfs, args.names)
    ]
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(args.out, index=False)
    print(combined.to_string(index=False))


if __name__ == '__main__':
    main()
