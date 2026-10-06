"""
Aggregates few-shot data-efficiency results across datasets, models,
and seeds, producing a line plot (metric vs. n_per_class) and a LaTeX
table.

Expected directory structure:
    <root>/<dataset>/<model>/feature_probing/<seed>/nshot_<num>/test_eval.csv
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visualization.plot_utils import get_palette


MODEL_DISPLAY_NAMES = {
    'miragefm3d_xattn_xattn_linear_w_2avg': 'CLOSER',
    'octcube3d_60_linear_w_cls_patch': 'OCTCube',
    # add other model-key -> display-name mappings as needed
}


def get_args():
    parser = argparse.ArgumentParser(description='Aggregate few-shot data-efficiency results')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing <dataset>/<model>/feature_probing/...')
    parser.add_argument('-m', '--metrics', type=str, nargs='+', default=['MCC', 'AUROC'])
    parser.add_argument('--output_dir', type=str, default='/home/morano/SW/Documents/My_Papers/MIRAGEv2/results/data_efficiency')
    return parser.parse_args()


def compute_seed_level_stats(all_df, metric):
    """
    Averaging scheme: first average across datasets WITHIN each seed
    (giving one value per (Model, n_per_class, Seed)), then compute
    mean/std ACROSS seeds. This treats seed as the unit of replication,
    consistent with the paper's other statistical analyses, rather than
    conflating dataset-to-dataset and seed-to-seed variance together.
    """
    # Step 1: average across datasets, within each seed
    per_seed = all_df.groupby(['Model', 'n_per_class', 'Seed'], observed=True)[metric].mean().reset_index()

    # Step 2: mean and std across seeds
    stats = per_seed.groupby(['Model', 'n_per_class'], observed=True)[metric].agg(['mean', 'std']).reset_index()

    return per_seed, stats

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


def plot_data_efficiency(all_df, metric, output_dir):
    per_seed, _ = compute_seed_level_stats(all_df, metric)

    # For plotting only: remap the large "all data" sentinel value to
    # a closer position (50) on the x-axis, labeled "All" instead of
    # its literal number, so it doesn't stretch the log-scale axis.
    max_real_shot = per_seed[per_seed['n_per_class'] < 1000]['n_per_class'].max()
    all_sentinel = per_seed['n_per_class'].max()
    plot_position_for_all = 70

    per_seed_plot = per_seed.copy()
    per_seed_plot['n_per_class_plot'] = per_seed_plot['n_per_class'].replace(
        all_sentinel, plot_position_for_all
    )

    plt.figure(figsize=(5, 5))
    sns.set_style("whitegrid")

    models = sorted(per_seed_plot['Model'].unique())
    palette = get_palette(models)

    ax = sns.lineplot(
        data=per_seed_plot,
        x='n_per_class_plot',
        y=metric,
        hue='Model',
        hue_order=models,
        palette=palette,
        marker='o',
        errorbar='sd',
    )
    ax.set_xscale('log')

    shot_counts_plot = sorted(per_seed_plot['n_per_class_plot'].unique())
    ax.set_xticks(shot_counts_plot)
    tick_labels = [
        'All' if n == plot_position_for_all else str(int(n))
        for n in shot_counts_plot
    ]
    ax.set_xticklabels(tick_labels)
    ax.minorticks_off()

    char = "b"
    if metric == "MCC":
        char = "a"
    ax.text(
        -0.15, 1.025, char,
        transform=ax.transAxes,
        fontsize=16,
        fontweight='bold',
        va='top',
        ha='right',
    )

    ax.set_xlabel('Number of training examples per class')
    ax.set_ylabel(metric)
    # ax.set_title(f'Data Efficiency: {metric} vs. Training Examples per Class')
    plt.tight_layout()

    save_fn = Path(output_dir) / f'data_efficiency_{metric}.pdf'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {save_fn}')


def build_table(all_df, metric, output_dir):
    _, stats = compute_seed_level_stats(all_df, metric)

    models = sorted(stats['Model'].unique())
    n_shots = sorted(stats['n_per_class'].unique())

    print("\n" + "=" * 80)
    print(f"DATA EFFICIENCY TABLE ({metric})")
    print("=" * 80)
    wide = stats.pivot(index='n_per_class', columns='Model', values=['mean', 'std'])
    print(wide.round(4))

    # LaTeX table
    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('\\centering')
    lines.append(f'\\caption{{Data efficiency comparison ({metric}, mean $\\pm$ std across datasets).}}')
    lines.append('\\label{tab:data_efficiency}')
    lines.append(f'\\begin{{tabular}}{{l{"l" * len(models)}}}')
    lines.append('\\toprule')
    lines.append('\\textbf{N per class} & ' + ' & '.join(f'\\textbf{{{m}}}' for m in models) + ' \\\\')
    lines.append('\\midrule')
    for n in n_shots:
        cells = [str(n)]
        for m in models:
            row = stats[(stats['n_per_class'] == n) & (stats['Model'] == m)]
            if len(row) == 0 or pd.isna(row['mean'].values[0]):
                cells.append('--')
            else:
                mean_val = row['mean'].values[0]
                std_val = row['std'].values[0]
                cells.append(f'{mean_val:.3f}$\\pm${std_val:.3f}')
        lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    latex_str = '\n'.join(lines)
    print('\n' + '=' * 80)
    print('LATEX TABLE')
    print('=' * 80)
    print(latex_str)

    save_fn = Path(output_dir) / f'data_efficiency_{metric}_table.tex'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    with open(save_fn, 'w') as f:
        f.write(latex_str)
    print(f'\nSaved table to {save_fn}')


def build_combined_table(all_df, metrics, output_dir):
    """
    Builds a single table: rows = n_per_class, columns = (model, metric)
    pairs, e.g. CLOSER-MCC, CLOSER-AUROC, OCTCube-MCC, OCTCube-AUROC.
    """
    all_stats = {}
    for metric in metrics:
        per_seed = all_df.groupby(['Model', 'n_per_class', 'Seed'], observed=True)[metric].mean().reset_index()
        stats = per_seed.groupby(['Model', 'n_per_class'], observed=True)[metric].agg(['mean', 'std']).reset_index()
        all_stats[metric] = stats

    models = sorted(all_df['Model'].unique())
    n_shots = sorted(all_df['n_per_class'].unique())

    # Console table
    print("\n" + "=" * 100)
    print(f"DATA EFFICIENCY TABLE ({' & '.join(metrics)})")
    print("=" * 100)
    for n in n_shots:
        row_str = f"n_per_class={n}: "
        for model in models:
            parts = []
            for metric in metrics:
                stats = all_stats[metric]
                match = stats[(stats['n_per_class'] == n) & (stats['Model'] == model)]
                if len(match) > 0 and not pd.isna(match['mean'].values[0]):
                    parts.append(f"{metric}={match['mean'].values[0]:.3f}±{match['std'].values[0]:.3f}")
                else:
                    parts.append(f"{metric}=--")
            row_str += f"[{model}: {', '.join(parts)}] "
        print(row_str)

    # LaTeX table: grouped column headers, one group per model, with
    # one sub-column per metric within each group
    n_models = len(models)
    n_metrics = len(metrics)
    col_spec = 'l' + ('l' * n_metrics) * n_models

    header_top = ' & ' + ' & '.join(
        f'\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{{model}}}}}' for model in models
    ) + ' \\\\'
    cmidrules = ' '.join(
        f'\\cmidrule(lr){{{2 + n_metrics*i}-{1 + n_metrics*(i+1)}}}' for i in range(n_models)
    )
    header_bottom = '\\textbf{N per class}' + ' & ' + ' & '.join(
        ' & '.join(f'\\textbf{{{m}}}' for m in metrics) for _ in models
    ) + ' \\\\'

    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('\\centering')
    lines.append(f'\\caption{{Data efficiency comparison ({", ".join(metrics)}; mean $\\pm$ std across seeds, averaged over datasets).}}')
    lines.append('\\label{tab:data_efficiency}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    lines.append(header_top)
    lines.append(cmidrules)
    lines.append(header_bottom)
    lines.append('\\midrule')

    for n in n_shots:
        cells = [str(n)]
        for model in models:
            for metric in metrics:
                stats = all_stats[metric]
                match = stats[(stats['n_per_class'] == n) & (stats['Model'] == model)]
                if len(match) == 0 or pd.isna(match['mean'].values[0]):
                    cells.append('--')
                else:
                    mean_val = match['mean'].values[0]
                    std_val = match['std'].values[0]
                    cells.append(f'{mean_val:.3f}$\\pm${std_val:.3f}')
        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    latex_str = '\n'.join(lines)
    print('\n' + '=' * 100)
    print('LATEX TABLE')
    print('=' * 100)
    print(latex_str)

    save_fn = Path(output_dir) / 'data_efficiency_table.tex'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    with open(save_fn, 'w') as f:
        f.write(latex_str)
    print(f'\nSaved table to {save_fn}')


def main():
    args = get_args()
    all_df = parse_results(args.root, args.metrics)

    if all_df.empty:
        print("No results found. Check --root path and directory structure.")
        return

    print(f"Loaded {len(all_df)} result rows...")

    for metric in args.metrics:
        if metric in all_df.columns:
            plot_data_efficiency(all_df, metric, args.output_dir)  # keep separate plots per metric

    build_combined_table(all_df, args.metrics, args.output_dir)  # one combined table


if __name__ == '__main__':
    main()
