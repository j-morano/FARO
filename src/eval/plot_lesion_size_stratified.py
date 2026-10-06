"""
Plot Dice as a function of ground-truth lesion size, comparing models, from
a lesion-size-stratified results CSV (as produced by stratify_lesion_size.py).

Produces a 2-panel figure. The "No lesion" bin is excluded, since this
analysis is specifically about sensitivity to lesion size:
    Panel 1: absolute Dice per bin (mean +/- SEM), plotted at each bin's
             geometric-mean lesion size on a log x-axis -- this correctly
             represents that bins are quantile-based and vary hugely in
             width, so the small-lesion bins (where the effect is
             concentrated) are not visually compressed into a single tick.
    Panel 2: relative advantage of a focal model (default: first in
             --model_order) over the best competing model, per bin -- the
             number that directly visualizes "how much better, in relative
             terms, at this lesion size", rather than requiring the reader
             to infer it from overlapping absolute-value lines.

Expected input columns:
    Model, SizeBin, N, Dice_mean, Dice_std

Usage:
    python plot_lesion_size_stratified.py -i octave_stratified.csv \
        -o octave_lesion_size.pdf --title "OCTAVE" \
        --model_order CLOSER iBOT MAE --focal_model CLOSER
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(
        description='Plot Dice vs. lesion size bin, comparing models.'
    )
    parser.add_argument('-i', '--input_csv', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--model_order', type=str, nargs='+', default=None)
    parser.add_argument('--focal_model', type=str, default=None,
                         help='Model whose relative advantage is plotted in panel 2.'
                              ' Default: first model in --model_order.')
    parser.add_argument('--colors', type=str, nargs='+', default=None)
    parser.add_argument('--min_n', type=int, default=0)
    parser.add_argument('--log_x', action='store_true', default=False,
                         help='Use a log-scaled x-axis (bin geometric-mean size)'
                              ' for the absolute Dice panel. (default: %(default)s)')
    parser.add_argument('--no_log_x', dest='log_x', action='store_false')
    parser.add_argument('--figsize', type=float, nargs=2, default=(11, 4.2))
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def _bin_edges(size_bin):
    """Parse a pandas interval string like '(129.0, 317.333]' -> (lo, hi)."""
    inner = size_bin.strip('()[]')
    lo, hi = [float(s.strip()) for s in inner.split(',')]
    return lo, hi


def _bin_sort_key(size_bin):
    lo, _ = _bin_edges(size_bin)
    return lo


def _bin_geomean(size_bin):
    lo, hi = _bin_edges(size_bin)
    lo = max(lo, 1e-6)  # guard against a zero/negative lower edge
    return np.sqrt(lo * hi)


def _short_bin_label(size_bin):
    lo, hi = _bin_edges(size_bin)

    def _fmt(v):
        if v >= 1000:
            return f'{v / 1000:.1f}k'.replace('.0k', 'k')
        return f'{v:.0f}'

    return f'{_fmt(lo)}\u2013{_fmt(hi)}'


def load_and_prepare(input_csv, model_order, min_n):
    df = pd.read_csv(input_csv)

    required = {'Model', 'SizeBin', 'N', 'Dice_mean', 'Dice_std'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Input CSV is missing required column(s): {missing}')

    # This analysis is specifically about lesion-size sensitivity, so the
    # "No lesion" bin (which has no size to plot against) is excluded.
    df = df[df['SizeBin'] != 'No lesion'].copy()

    if min_n > 0:
        df = df[df['N'] >= min_n].copy()

    if model_order is None:
        model_order = list(dict.fromkeys(df['Model']))
    else:
        missing_models = set(model_order) - set(df['Model'].unique())
        if missing_models:
            raise ValueError(f'--model_order contains models not in the CSV: {missing_models}')
        df = df[df['Model'].isin(model_order)]

    # SEM, not raw std -- with N in the hundreds, std vastly overstates
    # uncertainty in the *mean*, which is what we're comparing across models.
    df['Dice_sem'] = df['Dice_std'] / np.sqrt(df['N'].clip(lower=1))

    bin_order = sorted(df['SizeBin'].unique(), key=_bin_sort_key)
    df['_x_geomean'] = df['SizeBin'].apply(_bin_geomean)
    df['_x_index'] = df['SizeBin'].apply(lambda b: bin_order.index(b))

    return df, model_order, bin_order


def plot_absolute(ax, df, model_order, bin_order, colors, log_x):
    x_col = '_x_geomean' if log_x else '_x_index'
    for i, model in enumerate(model_order):
        sub = df[df['Model'] == model].sort_values(x_col)
        x = sub[x_col].to_numpy()
        y = sub['Dice_mean'].to_numpy()
        sem = sub['Dice_sem'].to_numpy()
        color = colors[i] if colors is not None else None

        line, = ax.plot(
            x, y, marker='o', markersize=5, linewidth=1.8, label=model, color=color
        )
        ax.fill_between(x, y - sem, y + sem, alpha=0.25, color=line.get_color())

    if log_x:
        ax.set_xscale('log')
        ax.set_xticks([_bin_geomean(b) for b in bin_order])
        ax.set_xticklabels([_short_bin_label(b) for b in bin_order], rotation=45, ha='right')
        ax.xaxis.set_minor_locator(plt.NullLocator())
    else:
        ax.set_xticks(range(len(bin_order)))
        ax.set_xticklabels([_short_bin_label(b) for b in bin_order], rotation=45, ha='right')

    ax.set_xlabel('Ground-truth lesion size (px)')
    ax.set_ylabel('Dice (mean \u00b1 SEM)')
    ax.grid(True, alpha=0.3)

    # Sample-size annotations, from the first model (bins/N are shared across models).
    # First half placed near the top of the axis, second half near the bottom,
    # so labels don't get covered by the lines/markers wherever the curve happens
    # to sit (e.g. a rising curve tends to crowd the top-right, so those labels
    # move to the bottom instead).
    ref = df[df['Model'] == model_order[0]].sort_values(x_col)
    ymin, ymax = ax.get_ylim()
    y_top = ymax - 0.04 * (ymax - ymin)
    y_bottom = ymin + 0.04 * (ymax - ymin)
    n_rows = len(ref)
    for i, (_, row) in enumerate(ref.iterrows()):
        top_half = i < n_rows / 2
        y_text = y_top if top_half else y_bottom
        va = 'top' if top_half else 'bottom'
        ax.text(
            row[x_col], y_text, f"n={int(row['N'])}",
            ha='center', va=va, fontsize=9, color='gray', rotation=90,
        )


def plot_relative_advantage(ax, df, model_order, bin_order, focal_model, color):
    """
    Relative Dice advantage of focal_model over the best competing model,
    per bin: (focal - best_other) / best_other * 100.
    Positive = focal model is better, at that bin, in relative terms.
    """
    other_models = [m for m in model_order if m != focal_model]
    rel_x, rel_y = [], []

    for b in bin_order:
        focal_row = df[(df['Model'] == focal_model) & (df['SizeBin'] == b)]
        other_rows = df[(df['Model'].isin(other_models)) & (df['SizeBin'] == b)]
        if focal_row.empty or other_rows.empty:
            continue
        focal_val = focal_row['Dice_mean'].values[0]
        best_other = other_rows['Dice_mean'].max()
        if best_other == 0:
            continue
        rel = (focal_val - best_other) / best_other * 100
        rel_x.append(bin_order.index(b))
        rel_y.append(rel)

    ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.7)
    ax.bar(rel_x, rel_y, color=color, alpha=0.85, width=0.6)
    ax.set_xticks(range(len(bin_order)))
    ax.set_xticklabels([_short_bin_label(b) for b in bin_order], rotation=45, ha='right')
    ax.set_xlabel('Ground-truth lesion size (px)')
    ax.set_ylabel(f'{focal_model} relative Dice gain (%)')
    ax.grid(True, alpha=0.3, axis='y')


def main():
    args = get_args()
    input_csv = Path(args.input_csv)
    output = Path(args.output) if args.output else input_csv.with_suffix('.pdf')

    df, model_order, bin_order = load_and_prepare(input_csv, args.model_order, args.min_n)
    focal_model = args.focal_model or ('CLOSER' if 'CLOSER' in model_order else model_order[0])

    colors = args.colors
    if colors is not None and len(colors) != len(model_order):
        raise ValueError('--colors must have the same length as --model_order')
    focal_color = colors[model_order.index(focal_model)] if colors is not None else '#7B5EA7'

    fig, axes = plt.subplots(1, 2, figsize=args.figsize)

    plot_absolute(axes[0], df, model_order, bin_order, colors, args.log_x)
    plot_relative_advantage(axes[1], df, model_order, bin_order, focal_model, focal_color)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='upper center', ncol=len(model_order),
        bbox_to_anchor=(0.5, 0.98), frameon=False,
    )

    if args.title:
        fig.suptitle(args.title, y=1.0, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.subplots_adjust(top=0.88)
    fig.savefig(output, dpi=args.dpi, bbox_inches='tight')
    print(f'Saved figure to "{output}"')


if __name__ == '__main__':
    main()
