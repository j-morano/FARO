"""
Generates a combined table (console + LaTeX) from computational
efficiency benchmark CSVs (one per model, produced by
computational_efficiency.py).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
import seaborn as sns

from visualization.plot_utils import get_palette


MODEL_DISPLAY_NAMES = {
    'miragefm3d_xattn_xattn': 'CLOSER',
    'octcube3d_60': 'OCTCube',
}


def get_args():
    parser = argparse.ArgumentParser(description='Build efficiency comparison table')
    parser.add_argument('--res_dir', type=str, default='./__output/efficiency')
    parser.add_argument('--output_dir', type=str, default='/home/morano/SW/Documents/My_Papers/MIRAGEv2/results/comp_efficiency')
    parser.add_argument(
        '--reference_model', type=str, default='miragefm3d_xattn_xattn',
        help='Model key to use as the reference (denominator) for '
             'speedup/memory-ratio columns.',
    )
    return parser.parse_args()


def fit_scaling_exponent(df):
    """Fits time ~ n_bscans^b via log-log linear regression."""
    df = df.dropna(subset=['mean_time_s']).sort_values('n_bscans')
    log_n = np.log(df['n_bscans'])
    log_t = np.log(df['mean_time_s'])
    b, _ = np.polyfit(log_n, log_t, 1)
    return b


def build_wide_table(dfs_by_model, reference_key):
    """
    Builds a wide table: rows = n_bscans, columns = (model, metric)
    pairs for time and memory, plus speedup/memory-ratio columns for
    every other model relative to the reference model.
    """
    all_bscans = sorted(set().union(*[set(df['n_bscans']) for df in dfs_by_model.values()]))
    other_keys = [k for k in dfs_by_model if k != reference_key]

    rows = []
    for n in all_bscans:
        row = {'n_bscans': n}
        values = {}
        for model_key, df in dfs_by_model.items():
            display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
            match = df[df['n_bscans'] == n]
            if len(match) == 0 or pd.isna(match['mean_time_s'].values[0]):
                t, m = None, None
            else:
                t = match['mean_time_s'].values[0]
                m = match['peak_mem_gb'].values[0]
            row[f'{display_name}_time'] = t
            row[f'{display_name}_mem'] = m
            values[model_key] = (t, m)

        ref_t, ref_m = values.get(reference_key, (None, None))
        for other_key in other_keys:
            other_t, other_m = values.get(other_key, (None, None))
            other_name = MODEL_DISPLAY_NAMES.get(other_key, other_key)
            if ref_t is not None and other_t is not None and ref_t > 0:
                row[f'{other_name}_speedup'] = other_t / ref_t
            else:
                row[f'{other_name}_speedup'] = None
            if ref_m is not None and other_m is not None and ref_m > 0:
                row[f'{other_name}_mem_ratio'] = other_m / ref_m
            else:
                row[f'{other_name}_mem_ratio'] = None

        rows.append(row)
    return pd.DataFrame(rows)


def print_console_table(wide_df, model_names, other_names):
    print("\n" + "=" * 100)
    print("COMPUTATIONAL EFFICIENCY COMPARISON")
    print("=" * 100)
    display_cols = ['n_bscans']
    for name in model_names:
        display_cols += [f'{name}_time', f'{name}_mem']
    for other_name in other_names:
        display_cols += [f'{other_name}_speedup', f'{other_name}_mem_ratio']
    print(wide_df[display_cols].round(3).to_string(index=False))


def print_latex_table(wide_df, model_names, reference_name, other_names, save_fn):
    n_models = len(model_names)

    col_spec = 'l' + 'll' * n_models + 'll' * len(other_names)

    header_top_parts = []
    for name in model_names:
        header_top_parts.append(f'\\multicolumn{{2}}{{c}}{{\\textbf{{{name}}}}}')
    for other_name in other_names:
        header_top_parts.append(
            f'\\multicolumn{{2}}{{c}}{{\\textbf{{{other_name}}} / \\textbf{{{reference_name}}}}}'
        )
    header_top = ' & ' + ' & '.join(header_top_parts) + ' \\\\'

    n_groups = n_models + len(other_names)
    cmidrules = ' '.join(
        f'\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}' for i in range(n_groups)
    )

    header_bottom_parts = ['\\textbf{B-scans}']
    for _ in model_names:
        header_bottom_parts.append('\\textbf{Time (s)} & \\textbf{Mem. (GB)}')
    for _ in other_names:
        header_bottom_parts.append('\\textbf{Time ratio} & \\textbf{Mem. ratio}')
    header_bottom = ' & '.join(header_bottom_parts) + ' \\\\'

    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('\\centering')
    lines.append(f'\\caption{{Inference time and peak GPU memory usage as a function of the '
                 f'number of B-scans per volume. Ratio columns show the relative time and '
                 f'memory usage of each model with respect to {reference_name}.}}')
    lines.append('\\label{tab:efficiency}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    lines.append(header_top)
    lines.append(cmidrules)
    lines.append(header_bottom)
    lines.append('\\midrule')

    for _, row in wide_df.iterrows():
        cells = [str(int(row['n_bscans']))]
        for name in model_names:
            t = row.get(f'{name}_time')
            m = row.get(f'{name}_mem')
            cells += (['--', '--'] if (t is None or pd.isna(t)) else [f'{t:.3f}', f'{m:.2f}'])
        for other_name in other_names:
            sp = row.get(f'{other_name}_speedup')
            mr = row.get(f'{other_name}_mem_ratio')
            cells += (['--', '--'] if (sp is None or pd.isna(sp)) else [f'{sp:.2f}$\\times$', f'{mr:.2f}$\\times$'])
        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    latex_str = '\n'.join(lines)
    print('\n' + '=' * 100)
    print('LATEX TABLE')
    print('=' * 100)
    print(latex_str)

    with open(save_fn, 'w') as f:
        f.write(latex_str)
    print(f'\nSaved to {save_fn}')



def plot_efficiency(dfs_by_model, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.set_style("whitegrid")

    models = sorted(dfs_by_model.keys())
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    palette = get_palette(display_names)

    all_bscans = set()
    for model_key in models:
        df = dfs_by_model[model_key].dropna(subset=['mean_time_s']).sort_values('n_bscans')
        display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        color = palette[display_name] if isinstance(palette, dict) else None
        all_bscans.update(df['n_bscans'].tolist())

        axes[0].plot(df['n_bscans'], df['mean_time_s'], marker='o', label=display_name, color=color)
        axes[1].plot(df['n_bscans'], df['peak_mem_gb'], marker='o', label=display_name, color=color)

    bscan_ticks = sorted(all_bscans)

    def fmt(x, pos):
        if x >= 1:
            return f'{x:.0f}' if x == int(x) else f'{x:.1f}'
        else:
            return f'{x:.2f}'

    for ax, ylabel, title in zip(
        axes,
        ['Inference time (s)', 'Peak GPU memory (GB)'],
        ['Inference Time vs. B-scans', 'Peak Memory vs. B-scans'],
    ):
        ax.set_xscale('log')
        ax.set_yscale('log')

        # X-axis: explicit ticks at the actual tested B-scan counts
        ax.set_xticks(bscan_ticks)
        ax.set_xticklabels([str(n) for n in bscan_ticks])
        ax.minorticks_off()

        # Y-axis: more ticks (major + minor), plain numeric labels
        # instead of default log powers-of-10
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(1, 10))))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax.yaxis.set_minor_formatter(NullFormatter())  # keep minor ticks unlabeled, just visual gridlines

        # Grid lines: major (solid) on both axes, subtle style
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)

        char = "b"
        if ylabel == 'Inference time (s)':
            char = "a"
        ax.text(
            -0.15, 1.025, char,
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top',
            ha='right',
        )

        ax.set_xlabel('Number of B-scans')
        ax.set_ylabel(ylabel)
        # ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    save_fn = Path(output_dir) / 'efficiency_plot.pdf'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {save_fn}')


def main():
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs_by_model = {}
    for csv_path in Path(args.res_dir).glob('*.csv'):
        if csv_path.name in ('efficiency_comparison.csv',):
            continue  # skip our own previous output
        df = pd.read_csv(csv_path)
        if 'model' not in df.columns:
            print(f"WARNING: skipping {csv_path} — no 'model' column found.")
            continue
        model_key = df['model'].iloc[0]
        dfs_by_model[model_key] = df

    if args.reference_model not in dfs_by_model:
        raise ValueError(f"Reference model '{args.reference_model}' not found among "
                          f"loaded CSVs: {list(dfs_by_model.keys())}")

    model_names = [MODEL_DISPLAY_NAMES.get(k, k) for k in dfs_by_model.keys()]
    reference_name = MODEL_DISPLAY_NAMES.get(args.reference_model, args.reference_model)
    other_names = [
        MODEL_DISPLAY_NAMES.get(k, k) for k in dfs_by_model if k != args.reference_model
    ]

    print("\nFitted scaling exponents (time ~ n_bscans^b):")
    for model_key, df in dfs_by_model.items():
        b = fit_scaling_exponent(df)
        display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        print(f"  {display_name}: b = {b:.3f}")

    wide_df = build_wide_table(dfs_by_model, args.reference_model)
    print_console_table(wide_df, model_names, other_names)

    save_fn = output_dir / 'efficiency_latex_table.tex'
    print_latex_table(wide_df, model_names, reference_name, other_names, save_fn)

    csv_save_fn = output_dir / 'efficiency_comparison.csv'
    wide_df.to_csv(csv_save_fn, index=False)
    print(f'Saved comparison CSV to {csv_save_fn}')

    plot_efficiency(dfs_by_model, args.output_dir)


if __name__ == '__main__':
    main()
