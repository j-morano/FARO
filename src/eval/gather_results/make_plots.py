"""
Reusable plotting/table script that reads from the pre-gathered CSVs
(2d_raw_results.csv, 3d_raw_results.csv, ablation_raw_seg_results.csv,
retrieval_raw_results.csv, raw_data_efficiency_results.csv, ...) instead
of rescanning raw eval folders. Styling (colors, longtable, bold/underline,
significance stars) matches the original per-experiment scripts.
Usage:
    python make_plots.py --kind 2d
    python make_plots.py --kind 3d
    python make_plots.py --kind chd
    python make_plots.py --kind seg_sota --level volume
    python make_plots.py --kind seg_ablation --level dataset
    python make_plots.py --kind retrieval
    python make_plots.py --kind data_efficiency
Only two real arguments:
    --kind   which results to process (2d, 3d, 3dabl, chd, abl, prets,
             seg_sota, seg_ablation, retrieval, data_efficiency)
    --level  significance-test level, used only for segmentation kinds:
             'dataset' (Wilcoxon on per-dataset means, default),
             'volume' (mixed-effects model on patient-level data, dataset
             as random intercept, falling back to Wilcoxon on per-dataset
             means if the model fails to converge),
             'volume_wilcoxon' (Wilcoxon pooled over all patients, paired
             by (Dataset, ID))
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
from scipy.stats import wilcoxon, ttest_rel
import statsmodels.formula.api as smf
from visualization.plot_utils import get_palette, get_model_colors

BASE_PATH = Path('/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/gen_results/')
CSV_DIR = Path('/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/raw_results/')

# Set from --format at the bottom; 'svg' by default so figures are easy
# to edit/compose in third-party vector tools, 'pdf' available via flag.
FIG_FORMAT = 'svg'

# ============================================================
# Per-kind configuration — everything that used to be a CLI flag
# or hardcoded at the bottom of the old scripts lives here.
# ============================================================
KINDS = {
    '2d': dict(
        type='classification', csv='2d_raw_results.csv', experiment='2d',
        order=['RETFound', 'VisionFM', 'UrFound', 'OCTCube', 'MIRAGE', 'CLOSER'],
        metrics=['MCC', 'BAcc', 'AUROC', 'AP'], single_dataset=False, wilcoxon=False,
    ),
    '3d': dict(
        type='classification', csv='3d_raw_results.csv', experiment='3d',
        order=['OCTCube (60)', 'CLOSER (60)', 'CLOSER (all)'],
        rename={'OCTCube (60 B-scans)': 'OCTCube (60)',
                'CLOSER (60 B-scans)': 'CLOSER (60)',
                'CLOSER (all B-scans)': 'CLOSER (all)'},
        metrics=['MCC', 'BAcc', 'AUROC', 'AP'], single_dataset=False, wilcoxon=False,
        fig_width=4, second_vs_third=True,
    ),
    '3dabl': dict(
        type='classification', csv='3dabl_raw_results.csv', experiment='3dabl',
        order=None, metrics=['MCC', 'BAcc', 'AUROC', 'AP'], single_dataset=False, wilcoxon=False,
    ),
    'chd': dict(
        type='classification', csv='chd_raw_results.csv', experiment='chd',
        order=['OCTCube', 'MIRAGE', 'RETFound', 'UrFound', 'VisionFM', 'CLOSER'],
        metrics=['MCC', 'BAcc', 'AUROC', 'AP'], single_dataset=True, wilcoxon=False,
    ),
    'abl': dict(
        type='classification', csv='abl_components_raw_results.csv', experiment='abl',
        order=None, metrics=['MCC', 'BAcc', 'AUROC', 'AP'], single_dataset=False, wilcoxon=False,
    ),
    'prets': dict(
        type='classification', csv='prets_raw_results.csv', experiment='prets',
        order=None, metrics=['MCC', 'BAcc', 'AUROC', 'AP'], single_dataset=False, wilcoxon=False,
    ),
    'seg_sota': dict(
        type='segmentation', csv='sota_raw_seg_results.csv', experiment='sota',
        order=['VisionFM', 'UrFound', 'OCTCube', 'RETFound', 'MIRAGE', 'CLOSER'],
        metrics=['Dice', 'HD952D'],
    ),
    'seg_ablation': dict(
        type='segmentation', csv='ablation_raw_seg_results.csv', experiment='ablation',
        order=None, metrics=['Dice', 'HD952D'],
    ),
    'retrieval': dict(
        type='retrieval', csv='retrieval_raw_results.csv',
        order=['OCTCube', 'CLOSER'],
    ),
    'data_efficiency': dict(
        type='data_efficiency', csv='raw_data_efficiency_results.csv',
        order=None, metrics=['MCC', 'AUROC'],
    ),
    'comp_efficiency': dict(
        type='comp_efficiency', csv='comp_efficiency_comparison.csv',
        reference_model='CLOSER',
    ),
}


# ============================================================
# Shared helpers
# ============================================================
def get_sig_stars(p_value):
    if p_value is None or pd.isna(p_value):
        return ''
    if p_value < 0.001:
        return '$^{***}$'
    elif p_value < 0.01:
        return '$^{**}$'
    elif p_value < 0.05:
        return '$^{*}$'
    return ''


def longtable_header(lines, n_cols, header_row, caption, label):
    """Appends the longtable preamble (caption, first/repeat heads, feet)."""
    lines.append(caption + f' \\label{{{label}}} \\\\')
    lines.append('\\toprule')
    lines.append(header_row)
    lines.append('\\midrule')
    lines.append('\\endfirsthead')
    lines.append(f'\\multicolumn{{{n_cols}}}{{c}}{{\\tablename\\ \\thetable{{}} -- continued from previous page}} \\\\')
    lines.append('\\toprule')
    lines.append(header_row)
    lines.append('\\midrule')
    lines.append('\\endhead')
    lines.append('\\midrule')
    lines.append(f'\\multicolumn{{{n_cols}}}{{r}}{{\\textit{{continued on next page}}}} \\\\')
    lines.append('\\endfoot')
    lines.append('\\bottomrule')
    lines.append('\\endlastfoot')


def save_latex(lines, save_fn):
    latex_str = '\n'.join(lines)
    print('\n' + '=' * 60)
    print(f'LATEX TABLE -> {save_fn}')
    print('=' * 60)
    print(latex_str)
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    with open(save_fn, 'w') as f:
        f.write(latex_str)
    print(f'Saved to {save_fn}')


def mixed_effects_test(full_df, best_model, other_model, metric):
    """
    Linear mixed-effects model treating dataset as random intercept, to
    properly account for within-dataset correlation (seeds or patients)
    when only a few datasets are available (avoids pseudoreplication).
    """
    subset = full_df[full_df['Model'].isin([best_model, other_model])].copy()
    subset['is_best'] = (subset['Model'] == best_model).astype(int)
    subset[metric] = pd.to_numeric(subset[metric], errors='coerce')
    subset = subset.dropna(subset=[metric])
    try:
        model = smf.mixedlm(f"{metric} ~ is_best", data=subset, groups=subset["Dataset"])
        result = model.fit()
        return result.params['is_best'], result.pvalues['is_best']
    except Exception as e:
        print(f"Mixed effects model failed: {e}")
        return None, None


def sig_str_from_pvalue(p_value):
    if p_value < 0.001:
        return '$p<0.001$'
    elif p_value < 0.01:
        return '$p<0.01$'
    elif p_value < 0.05:
        return '$p<0.05$'
    return f'$p={p_value:.3f}$'


def wrap_for_plot(label):
    """Insert a line break before a trailing '(...)' for cleaner plot
    x-axis labels, without needing '\n' stored in the data itself."""
    if ' (' in label:
        return label.replace(' (', '\n(', 1)
    return label


def draw_mean_annotation(ax, x, mean_val, value_fmt='{:.3f}', mean_style='marker', box_half_width=0.38,
                          dot_spacing_px=6):
    """
    Draws the mean marker/label for a single box at x position `x`.
    mean_style options:
      - 'marker' (default): standard seaborn-style circle marker + boxed
        number placed just above it (caller must pass showmeans=True to
        sns.boxplot for the marker itself to render).
      - 'dots': no marker; a row of small white squares (thin black
        border) spans the box width at the mean height, spaced roughly
        `dot_spacing_px` pixels apart regardless of figure size, with the
        value centered directly on top (outlined text for legibility).
      - 'diamond': a white diamond marker with a black border sits at
        the mean height (shape deliberately distinct from the circular
        outlier/stripplot points), with the value labeled just above it
        — close by, but not touching the marker.
    """
    if mean_style == 'dots':
        p_left = ax.transData.transform((x - box_half_width, mean_val))
        p_right = ax.transData.transform((x + box_half_width, mean_val))
        width_px = abs(p_right[0] - p_left[0])
        n_dots = max(3, int(round(width_px / dot_spacing_px)) + 1)
        dot_xs = np.linspace(x - box_half_width, x + box_half_width, n_dots)
        ax.scatter(
            dot_xs, [mean_val] * n_dots,
            s=6, marker='s', color='white', edgecolor='black', linewidth=0.8,
            zorder=9,
        )
        ax.text(
            x, mean_val, value_fmt.format(mean_val),
            va='center', ha='center', color='white', fontweight='bold', fontsize=9,
            path_effects=[pe.withStroke(linewidth=2.2, foreground='black')],
            zorder=10,
        )
    elif mean_style == 'diamond':
        ax.scatter(
            [x], [mean_val],
            s=32, marker='D', color='white', edgecolor='black', linewidth=1.0,
            zorder=9,
        )
        # Text placed a small fixed pixel offset above the marker (close
        # by, not touching it), regardless of the y-axis scale/units of
        # the plot — white fill with a black outline, same style as the
        # 'dots' variant, for legibility over any box color
        text = ax.annotate(
            value_fmt.format(mean_val),
            xy=(x, mean_val), xycoords='data',
            xytext=(0, 5), textcoords='offset points',
            ha='center', va='bottom', color='white', fontweight='bold', fontsize=9,
            zorder=10,
        )
        text.set_path_effects([pe.withStroke(linewidth=2.2, foreground='black')])
    else:
        ax.text(
            x, mean_val, value_fmt.format(mean_val),
            va='bottom', ha='center', color='black', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
        )


# ============================================================
# Classification (2d, 3d, chd, ablation, ...) — columns:
# Model, Dataset, Run, <metric columns e.g. MCC, BAcc, AUROC, AP>
# ============================================================
def compute_pvalue_avg(full_df, best_model, other_model, metric, use_wilcoxon, single_dataset):
    if single_dataset:
        best_data = full_df[full_df['Model'] == best_model].sort_values('Run')[metric].values
        other_data = full_df[full_df['Model'] == other_model].sort_values('Run')[metric].values
        try:
            _, p = ttest_rel(best_data, other_data, alternative='greater')
            return p
        except Exception:
            return None
    if use_wilcoxon:
        best_data = full_df[full_df['Model'] == best_model].groupby('Dataset')[metric].mean().sort_index().values
        other_data = full_df[full_df['Model'] == other_model].groupby('Dataset')[metric].mean().sort_index().values
        try:
            _, p = wilcoxon(best_data, other_data)
            return p
        except Exception:
            return None
    _, p = mixed_effects_test(full_df, best_model, other_model, metric)
    return p


def compute_pvalue_per_dataset(dataset_df, best_model, other_model, metric):
    best_data = pd.to_numeric(dataset_df[dataset_df['Model'] == best_model].sort_values('Run')[metric], errors='coerce').values
    other_data = pd.to_numeric(dataset_df[dataset_df['Model'] == other_model].sort_values('Run')[metric], errors='coerce').values
    if len(best_data) != len(other_data) or len(best_data) < 2:
        return None
    try:
        _, p = ttest_rel(best_data, other_data, alternative='greater')
        return p
    except Exception:
        return None


def cls_print_latex_table(full_df, prescribed_order, experiment, metrics, use_wilcoxon, single_dataset):
    rows = []
    for model_name in prescribed_order:
        model_df = full_df[full_df['Model'] == model_name]
        row = {'Model': model_name}
        for m in metrics:
            if m not in model_df.columns:
                row[f'{m}_mean'], row[f'{m}_std'] = float('nan'), float('nan')
                continue
            if single_dataset:
                values = pd.to_numeric(model_df[m], errors='coerce')
                row[f'{m}_mean'], row[f'{m}_std'] = values.mean(), values.std()
            else:
                per_dataset = model_df.groupby('Dataset')[m].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
                row[f'{m}_mean'], row[f'{m}_std'] = per_dataset.mean(), per_dataset.std()
        rows.append(row)
    results_df = pd.DataFrame(rows).set_index('Model')
    best_per_metric = {m: results_df[f'{m}_mean'].idxmax() for m in metrics if f'{m}_mean' in results_df.columns}
    second_best_per_metric = {m: results_df[f'{m}_mean'].nlargest(2).idxmin() for m in metrics if f'{m}_mean' in results_df.columns}
    pvalues = {}
    for m in metrics:
        if m not in best_per_metric:
            continue
        pvalues[m] = compute_pvalue_avg(full_df, best_per_metric[m], second_best_per_metric[m], m, use_wilcoxon, single_dataset)
    color_defs, model_to_colorname = get_model_colors(prescribed_order)
    n_cols = len(metrics)
    col_header = ' & '.join([f'\\textbf{{{m}}}' for m in metrics])
    header_row = f'\\textbf{{Model}} & {col_header} \\\\'
    caption = (
        '\\caption{Performance comparison across models. Best results in \\textbf{bold}, '
        'second best \\underline{underlined}. Significance of best vs. second best: '
        '$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}'
    )
    lines = list(color_defs) + ['\\footnotesize', f'\\begin{{longtable}}{{l{"l" * n_cols}}}']
    longtable_header(lines, 1 + n_cols, header_row, caption, f'tab:{experiment}')
    for model_name in prescribed_order:
        color_name = model_to_colorname.get(model_name)
        cells = [f'\\textcolor{{{color_name}}}{{{model_name}}}' if color_name else model_name]
        for m in metrics:
            if f'{m}_mean' not in results_df.columns or pd.isna(results_df.loc[model_name, f'{m}_mean']):
                cells.append('--')
                continue
            mean_val, std_val = results_df.loc[model_name, f'{m}_mean'], results_df.loc[model_name, f'{m}_std']
            is_best = model_name == best_per_metric.get(m)
            stars = get_sig_stars(pvalues.get(m)) if is_best else ''
            cell_str = f'{mean_val:.3f}$\\pm${std_val:.3f}{stars}'
            if is_best:
                cell_str = f'\\textbf{{{cell_str}}}'
            elif model_name == second_best_per_metric.get(m):
                cell_str = f'\\underline{{{cell_str}}}'
            cells.append(cell_str)
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\end{longtable}', '\\normalsize']
    save_latex(lines, BASE_PATH / experiment / f'{experiment}_latex_table.tex')


def cls_print_latex_table_per_dataset(full_df, prescribed_order, experiment, metrics):
    datasets = sorted(full_df['Dataset'].unique())
    color_defs, model_to_colorname = get_model_colors(prescribed_order)
    n_cols = len(metrics)
    col_header = ' & '.join([f'\\textbf{{{m}}}' for m in metrics])
    header_row = f'\\textbf{{Dataset}} & \\textbf{{Model}} & {col_header} \\\\'
    caption = (
        '\\caption{Per-dataset performance comparison across models. '
        'Best results in \\textbf{bold}, second best \\underline{underlined}. '
        "Significance (one-tailed Student's t-test, paired by seed): "
        '$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}'
    )
    lines = list(color_defs) + ['\\footnotesize', f'\\begin{{longtable}}{{ll{"l" * n_cols}}}']
    longtable_header(lines, 2 + n_cols, header_row, caption, f'tab:{experiment}_per_dataset')
    for dataset in datasets:
        dataset_df = full_df[full_df['Dataset'] == dataset]
        rows = []
        for model_name in prescribed_order:
            model_df = dataset_df[dataset_df['Model'] == model_name]
            row = {'Model': model_name}
            for m in metrics:
                values = pd.to_numeric(model_df[m], errors='coerce') if m in model_df.columns else pd.Series(dtype=float)
                row[f'{m}_mean'], row[f'{m}_std'] = values.mean(), values.std()
            rows.append(row)
        results_df = pd.DataFrame(rows).set_index('Model')
        best_per_metric, second_best_per_metric, pvalues = {}, {}, {}
        for m in metrics:
            valid = results_df[f'{m}_mean'].dropna()
            if valid.empty:
                continue
            best_per_metric[m] = valid.idxmax()
            if len(valid) >= 2:
                second_best_per_metric[m] = valid.nlargest(2).idxmin()
                pvalues[m] = compute_pvalue_per_dataset(dataset_df, best_per_metric[m], second_best_per_metric[m], m)
        for i, model_name in enumerate(prescribed_order):
            if model_name not in results_df.index:
                continue
            color_name = model_to_colorname.get(model_name)
            cells = [f'\\multirow{{{len(prescribed_order)}}}{{*}}{{{dataset.replace("_", " ")}}}' if i == 0 else '']
            cells.append(f'\\textcolor{{{color_name}}}{{{model_name}}}' if color_name else model_name)
            for m in metrics:
                mean_val = results_df.loc[model_name, f'{m}_mean']
                if pd.isna(mean_val):
                    cells.append('--')
                    continue
                std_val = results_df.loc[model_name, f'{m}_std']
                is_best = best_per_metric.get(m) == model_name
                stars = get_sig_stars(pvalues.get(m)) if is_best else ''
                cell_str = f'{mean_val:.3f}$\\pm${std_val:.3f}{stars}'
                if is_best:
                    cell_str = f'\\textbf{{{cell_str}}}'
                elif second_best_per_metric.get(m) == model_name:
                    cell_str = f'\\underline{{{cell_str}}}'
                cells.append(cell_str)
            lines.append(' & '.join(cells) + ' \\\\')
        lines.append('\\midrule')
    if lines[-1] == '\\midrule':
        lines.pop()
    lines += ['\\end{longtable}', '\\normalsize']
    save_latex(lines, BASE_PATH / experiment / f'{experiment}_latex_table_per_dataset.tex')


def cls_plot_boxplot(full_df, prescribed_order, experiment, metric, single_dataset, use_wilcoxon,
                      fig_width=None, second_vs_third=False, mean_style='marker'):
    means_by_model = full_df.groupby('Model', observed=True)[metric].mean()
    ranked = means_by_model.sort_values(ascending=False)
    best_model = ranked.index[0]
    second_best_model = ranked.index[1]
    p_value_1 = compute_pvalue_avg(full_df, best_model, second_best_model, metric, use_wilcoxon, single_dataset)
    best_idx = prescribed_order.index(best_model)
    second_best_idx = prescribed_order.index(second_best_model)

    p_value_2 = None
    third_idx = None
    if second_vs_third and len(ranked) >= 3:
        third_model = ranked.index[2]
        p_value_2 = compute_pvalue_avg(full_df, second_best_model, third_model, metric, use_wilcoxon, single_dataset)
        third_idx = prescribed_order.index(third_model)

    width = fig_width if fig_width is not None else len(prescribed_order)
    plt.figure(figsize=(width, 5))
    sns.set_style("whitegrid")
    ax = sns.boxplot(
        data=full_df, hue='Model', x='Model', y=metric,
        palette=get_palette(prescribed_order), order=prescribed_order,
        showmeans=(mean_style == 'marker'),
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"} if mean_style == 'marker' else None,
        legend=False,
    )
    sns.stripplot(data=full_df, x='Model', y=metric, order=prescribed_order, color="black", alpha=0.3, jitter=True, size=4)
    ax.set_xticklabels([wrap_for_plot(m) for m in prescribed_order])

    means = full_df.groupby('Model', observed=True)[metric].mean()
    for i, model_name in enumerate(prescribed_order):
        if model_name in means:
            draw_mean_annotation(ax, i, means[model_name], mean_style=mean_style)

    y_min, y_max = ax.get_ylim()
    span = y_max - y_min

    def draw_bracket(idx_a, idx_b, p_value, height):
        ax.plot([idx_a, idx_b], [height, height], 'k-', linewidth=2)
        ax.plot([idx_a, idx_a], [height - span * 0.02, height], 'k-', linewidth=2)
        ax.plot([idx_b, idx_b], [height - span * 0.02, height], 'k-', linewidth=2)
        ax.text((idx_a + idx_b) / 2, height + span * 0.02, sig_str_from_pvalue(p_value),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    top_extra = 0.1
    if p_value_1 is not None:
        draw_bracket(best_idx, second_best_idx, p_value_1, y_max)
        top_extra += 0.07
    if p_value_2 is not None:
        draw_bracket(second_best_idx, third_idx, p_value_2, y_max + span * 0.12)
        top_extra += 0.07

    ax.set_ylim(y_min, y_max + span * top_extra)

    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.tight_layout()
    save_fn = BASE_PATH / experiment / f'{experiment}_boxplot_{metric}.{FIG_FORMAT}'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().patch.set_alpha(0)
    plt.gca().set_facecolor('white')
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Boxplot saved to {save_fn}")


def cls_plot_barplot(full_df, prescribed_order, experiment, metric):
    datasets = sorted(full_df['Dataset'].unique())
    plt.figure(figsize=(max(8, 5 * len(datasets)), 8))
    ax = sns.barplot(
        data=full_df, x='Dataset', y=metric, hue='Model',
        palette=get_palette(prescribed_order), hue_order=prescribed_order,
        errorbar='sd', capsize=.1,
    )
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.text(p.get_x() + p.get_width() / 2., height + 0.005, f'{height * 100:.1f}',
                     ha='center', va='bottom', color='black', fontweight='bold')
    plt.ylabel(metric)
    plt.xlabel('Dataset')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, [wrap_for_plot(l) for l in labels], title='Model',
               bbox_to_anchor=(0.5, -0.15), loc='center', ncol=len(prescribed_order))
    plt.tight_layout()
    save_fn = BASE_PATH / experiment / f'{experiment}_barplot_{metric}.{FIG_FORMAT}'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().patch.set_alpha(0)
    plt.gca().set_facecolor('white')
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Barplot saved to {save_fn}")


def run_classification(cfg, table_only, mean_style='marker'):
    full_df = pd.read_csv(CSV_DIR / cfg['csv'])
    if cfg.get('rename'):
        full_df['Model'] = full_df['Model'].replace(cfg['rename'])
    prescribed_order = cfg['order'] or sorted(full_df['Model'].unique())
    metrics = [m for m in cfg['metrics'] if m in full_df.columns]
    fig_width = cfg.get('fig_width')
    second_vs_third = cfg.get('second_vs_third', False)
    cls_print_latex_table(full_df, prescribed_order, cfg['experiment'], metrics, cfg['wilcoxon'], cfg['single_dataset'])
    cls_print_latex_table_per_dataset(full_df, prescribed_order, cfg['experiment'], metrics)
    if not table_only:
        for metric in metrics:
            cls_plot_boxplot(full_df, prescribed_order, cfg['experiment'], metric, cfg['single_dataset'], cfg['wilcoxon'],
                              fig_width, second_vs_third, mean_style)
            cls_plot_barplot(full_df, prescribed_order, cfg['experiment'], metric)


# ============================================================
# Segmentation — columns: ID, Class, Dice, IoU, HD95, HD952D,
# MAD, Dataset (already _layers/_lesions split), Model
#
# `level` controls the significance-test method AND the granularity
# used for the mean/std reported in the averaged table/boxplot:
#   'dataset'         Mean/std computed ACROSS DATASETS (one value per
#                      dataset first, via patient-then-dataset averaging,
#                      then mean/std over those per-dataset values).
#                      Significance: Wilcoxon on those per-dataset means.
#                      This is the default and matches the original
#                      script's non-'volume' behavior.
#   'volume'          Mean/std computed ACROSS PATIENTS (patient-level
#                      granularity, pooled over all datasets). Significance:
#                      mixed-effects model (dataset as random intercept),
#                      falling back to Wilcoxon on per-dataset means if
#                      the model fails to converge.
#   'volume_wilcoxon' Same patient-level mean/std as 'volume'. Significance:
#                      Wilcoxon pooled over all patients, paired by
#                      (Dataset, ID).
# ============================================================
def seg_volume_level(full_df, metric):
    """Average across classes within each (Dataset, Model, ID) — i.e.
    one row per patient/volume."""
    return full_df.groupby(['Dataset', 'Model', 'ID'], observed=True)[metric].mean().reset_index()


def seg_level_df(full_df, metric, level):
    """
    Returns the granularity that mean/std and the significance test
    should be computed on, matching the original per-level semantics:
      - 'volume' / 'volume_wilcoxon': patient-level (one row per ID),
        pooled across all datasets.
      - 'dataset': one row per (Dataset, Model), averaged across patients
        within each dataset first (so mean/std afterward is computed
        ACROSS DATASETS, not across patients).
    """
    vol_df = seg_volume_level(full_df, metric)
    if level in ('volume', 'volume_wilcoxon'):
        return vol_df
    return vol_df.groupby(['Dataset', 'Model'], observed=True)[metric].mean().reset_index()


def seg_compute_pvalue_avg(level_df, best_model, other_model, metric, higher_is_better, level):
    """
    level_df must already be at the granularity produced by seg_level_df
    for the given `level` (patient-level for 'volume'/'volume_wilcoxon',
    one-row-per-dataset for 'dataset').
    """
    alt = 'greater' if higher_is_better else 'less'
    if level == 'volume':
        _, p = mixed_effects_test(level_df, best_model, other_model, metric)
        if p is not None:
            return p
        print("LMM failed, falling back to Wilcoxon on dataset means instead.")
        best_data = level_df[level_df['Model'] == best_model].groupby('Dataset')[metric].mean().sort_index().values
        other_data = level_df[level_df['Model'] == other_model].groupby('Dataset')[metric].mean().sort_index().values
        try:
            _, p = wilcoxon(best_data, other_data, alternative=alt)
            return p
        except Exception:
            return None
    elif level == 'volume_wilcoxon':
        best_data = level_df[level_df['Model'] == best_model].sort_values(['Dataset', 'ID'])
        other_data = level_df[level_df['Model'] == other_model].sort_values(['Dataset', 'ID'])
        valid_mask = best_data[metric].notna().values & other_data[metric].notna().values
        best_vals = best_data[metric].values[valid_mask]
        other_vals = other_data[metric].values[valid_mask]
        try:
            _, p = wilcoxon(best_vals, other_vals, alternative=alt)
            return p
        except Exception:
            return None
    else:  # 'dataset' — level_df already has one row per (Dataset, Model)
        best_data = level_df[level_df['Model'] == best_model].sort_values('Dataset')[metric].values
        other_data = level_df[level_df['Model'] == other_model].sort_values('Dataset')[metric].values
        try:
            _, p = wilcoxon(best_data, other_data, alternative=alt)
            return p
        except Exception:
            return None


def seg_compute_pvalue_per_dataset(dataset_df, best_model, other_model, metric, higher_is_better):
    """Always volume-level within a single dataset — the only granularity
    with repeated (patient) measures to pair on inside one dataset."""
    alt = 'greater' if higher_is_better else 'less'
    best_data = dataset_df[dataset_df['Model'] == best_model].groupby('ID')[metric].mean().sort_index()
    other_data = dataset_df[dataset_df['Model'] == other_model].groupby('ID')[metric].mean().sort_index()
    common_ids = best_data.index.intersection(other_data.index)
    if len(common_ids) < 2:
        return None
    try:
        _, p = wilcoxon(best_data.loc[common_ids].values, other_data.loc[common_ids].values, alternative=alt)
        return p
    except Exception:
        return None


def seg_print_latex_table(full_df, prescribed_order, experiment, metrics, level):
    results_df, pvalues, best_per_metric, second_best_per_metric = None, {}, {}, {}
    for m in metrics:
        if m not in full_df.columns:
            continue
        level_df = seg_level_df(full_df, m, level)
        higher_is_better = (m == 'Dice')
        stats = level_df.groupby('Model', observed=True)[m].agg(['mean', 'std'])
        stats.columns = [f'{m}_mean', f'{m}_std']
        results_df = stats if results_df is None else results_df.join(stats, how='outer')
        means = level_df.groupby('Model', observed=True)[m].mean()
        best = means.idxmax() if higher_is_better else means.idxmin()
        second = means.nlargest(2).idxmin() if higher_is_better else means.nsmallest(2).idxmax()
        best_per_metric[m], second_best_per_metric[m] = best, second
        pvalues[m] = seg_compute_pvalue_avg(level_df, best, second, m, higher_is_better, level)
    color_defs, model_to_colorname = get_model_colors(prescribed_order)
    n_cols = len(metrics)
    col_header = ' & '.join([f'\\textbf{{{m}}}' for m in metrics])
    header_row = f'\\textbf{{Model}} & {col_header} \\\\'
    level_desc = {
        'dataset': "Wilcoxon signed-rank test on per-dataset means",
        'volume': "linear mixed-effects model (dataset as random intercept) on patient-level data",
        'volume_wilcoxon': "Wilcoxon signed-rank test pooled over all patients",
    }[level]
    caption = (
        '\\caption{Performance comparison across models. Best results in \\textbf{bold}, '
        'second best \\underline{underlined}. Significance of best vs. second best '
        f'({level_desc}): $^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$.}}'
    )
    lines = list(color_defs) + ['\\footnotesize', f'\\begin{{longtable}}{{l{"l" * n_cols}}}']
    longtable_header(lines, 1 + n_cols, header_row, caption, f'tab:seg_{experiment}')
    for model_name in prescribed_order:
        if model_name not in results_df.index:
            continue
        color_name = model_to_colorname.get(model_name)
        cells = [f'\\textcolor{{{color_name}}}{{{model_name}}}' if color_name else model_name]
        for m in metrics:
            mean_val = results_df.loc[model_name, f'{m}_mean']
            if pd.isna(mean_val):
                cells.append('--')
                continue
            std_val = results_df.loc[model_name, f'{m}_std']
            is_best = best_per_metric.get(m) == model_name
            stars = get_sig_stars(pvalues.get(m)) if is_best else ''
            cell_str = f'{mean_val:.3f}$\\pm${std_val:.3f}{stars}'
            if is_best:
                cell_str = f'\\textbf{{{cell_str}}}'
            elif second_best_per_metric.get(m) == model_name:
                cell_str = f'\\underline{{{cell_str}}}'
            cells.append(cell_str)
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\end{longtable}', '\\normalsize']
    save_latex(lines, BASE_PATH / f'seg_{experiment}' / 'seg_latex_table.tex')


def seg_print_latex_table_per_dataset(full_df, prescribed_order, experiment, metrics):
    """Per-dataset breakdown is always volume-level (patient granularity),
    regardless of `level` — a per-dataset test needs repeated patient
    measures to pair on, which only exists at the volume level."""
    datasets = sorted(full_df['Dataset'].unique())
    color_defs, model_to_colorname = get_model_colors(prescribed_order)
    n_cols = len(metrics)
    col_header = ' & '.join([f'\\textbf{{{m}}}' for m in metrics])
    header_row = f'\\textbf{{Dataset}} & \\textbf{{Model}} & {col_header} \\\\'
    caption = (
        '\\caption{Per-dataset segmentation performance comparison across models. '
        'Best results in \\textbf{bold}, second best \\underline{underlined}. '
        'Statistical significance (Wilcoxon signed-rank test, paired by patient ID): '
        '$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}'
    )
    lines = list(color_defs) + ['\\footnotesize', f'\\begin{{longtable}}{{ll{"l" * n_cols}}}']
    longtable_header(lines, 2 + n_cols, header_row, caption, f'tab:seg_{experiment}_per_dataset')
    for dataset in datasets:
        dataset_df = full_df[full_df['Dataset'] == dataset]
        results_df, best_per_metric, second_best_per_metric, pvalues = None, {}, {}, {}
        for m in metrics:
            if m not in dataset_df.columns:
                continue
            vol_df = seg_volume_level(dataset_df, m)
            higher_is_better = (m == 'Dice')
            stats = vol_df.groupby('Model', observed=True)[m].agg(['mean', 'std'])
            stats.columns = [f'{m}_mean', f'{m}_std']
            results_df = stats if results_df is None else results_df.join(stats, how='outer')
            means = vol_df.groupby('Model', observed=True)[m].mean()
            if means.empty:
                continue
            best = means.idxmax() if higher_is_better else means.idxmin()
            best_per_metric[m] = best
            if len(means) >= 2:
                second = means.nlargest(2).idxmin() if higher_is_better else means.nsmallest(2).idxmax()
                second_best_per_metric[m] = second
                pvalues[m] = seg_compute_pvalue_per_dataset(vol_df, best, second, m, higher_is_better)
        if results_df is None:
            continue
        for i, model_name in enumerate(prescribed_order):
            if model_name not in results_df.index:
                continue
            color_name = model_to_colorname.get(model_name)
            cells = [f'\\multirow{{{len(prescribed_order)}}}{{*}}{{{dataset.replace("_", " ")}}}' if i == 0 else '']
            cells.append(f'\\textcolor{{{color_name}}}{{{model_name}}}' if color_name else model_name)
            for m in metrics:
                if f'{m}_mean' not in results_df.columns:
                    cells.append('--')
                    continue
                mean_val = results_df.loc[model_name, f'{m}_mean']
                if pd.isna(mean_val):
                    cells.append('--')
                    continue
                std_val = results_df.loc[model_name, f'{m}_std']
                is_best = best_per_metric.get(m) == model_name
                stars = get_sig_stars(pvalues.get(m)) if is_best else ''
                cell_str = f'{mean_val:.3f}$\\pm${std_val:.3f}{stars}'
                if is_best:
                    cell_str = f'\\textbf{{{cell_str}}}'
                elif second_best_per_metric.get(m) == model_name:
                    cell_str = f'\\underline{{{cell_str}}}'
                cells.append(cell_str)
            lines.append(' & '.join(cells) + ' \\\\')
        lines.append('\\midrule')
    if lines[-1] == '\\midrule':
        lines.pop()
    lines += ['\\end{longtable}', '\\normalsize']
    save_latex(lines, BASE_PATH / f'seg_{experiment}' / 'seg_latex_table_per_dataset.tex')


def seg_plot_boxplot(full_df, prescribed_order, experiment, metric, level, mean_style='marker'):
    # FIX: use seg_level_df (matching the requested `level`) instead of
    # always plotting raw patient-level data — this is what was broken:
    # for level='dataset' the plot must show one point-of-truth per
    # dataset (patient-averaged first), not raw per-patient spread.
    level_df = seg_level_df(full_df, metric, level)
    higher_is_better = (metric == 'Dice')
    means = level_df.groupby('Model', observed=True)[metric].mean()
    best_model = means.idxmax() if higher_is_better else means.idxmin()
    second_best_model = means.nlargest(2).idxmin() if higher_is_better else means.nsmallest(2).idxmax()
    p_value = seg_compute_pvalue_avg(level_df, best_model, second_best_model, metric, higher_is_better, level)
    best_idx, second_best_idx = prescribed_order.index(best_model), prescribed_order.index(second_best_model)

    plt.figure(figsize=(len(prescribed_order), 5))
    sns.set_style("whitegrid")
    ax = sns.boxplot(
        data=level_df, hue='Model', x='Model', y=metric,
        palette=get_palette(prescribed_order), order=prescribed_order,
        showmeans=(mean_style == 'marker'),
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"} if mean_style == 'marker' else None,
        legend=False,
    )
    means_plot = level_df.groupby('Model', observed=True)[metric].mean()
    for i, model_name in enumerate(prescribed_order):
        if model_name in means_plot:
            draw_mean_annotation(ax, i, means_plot[model_name], mean_style=mean_style)
    if p_value is not None:
        y_min, y_max = ax.get_ylim()
        line_y = y_max
        ax.plot([best_idx, second_best_idx], [line_y, line_y], 'k-', linewidth=2)
        ax.plot([best_idx, best_idx], [line_y - (y_max - y_min) * 0.02, line_y], 'k-', linewidth=2)
        ax.plot([second_best_idx, second_best_idx], [line_y - (y_max - y_min) * 0.02, line_y], 'k-', linewidth=2)
        ax.text((best_idx + second_best_idx) / 2, line_y + (y_max - y_min) * 0.02, sig_str_from_pvalue(p_value),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.tight_layout()
    save_fn = BASE_PATH / f'seg_{experiment}' / f'boxplot_{metric}.{FIG_FORMAT}'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().patch.set_alpha(0)
    plt.gca().set_facecolor('white')
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Boxplot saved to {save_fn}")


def run_segmentation(cfg, level, table_only, mean_style='marker'):
    full_df = pd.read_csv(CSV_DIR / cfg['csv'])
    prescribed_order = cfg['order'] or sorted(full_df['Model'].unique())
    metrics = [m for m in cfg['metrics'] if m in full_df.columns]
    seg_print_latex_table(full_df, prescribed_order, cfg['experiment'], metrics, level)
    seg_print_latex_table_per_dataset(full_df, prescribed_order, cfg['experiment'], metrics)
    if not table_only:
        for metric in metrics:
            seg_plot_boxplot(full_df, prescribed_order, cfg['experiment'], metric, level, mean_style)


# ============================================================
# Retrieval — columns: model_name, dataset_name, top_1_acc,
# top_5_acc, top_10_acc, mAP
# ============================================================
def ret_plot_boxplot(df, prescribed_order, metric, metric_label, mean_style='marker'):
    means = df.groupby('model_name', observed=True)[metric].mean()
    ranked = means.sort_values(ascending=False)
    best_model, second_best_model = ranked.index[0], ranked.index[1]

    best_data = df[df['model_name'] == best_model].sort_values('dataset_name')[metric].values
    second_data = df[df['model_name'] == second_best_model].sort_values('dataset_name')[metric].values
    p_value = None
    if len(best_data) == len(second_data) and len(best_data) >= 2:
        try:
            _, p_value = wilcoxon(best_data, second_data, alternative='greater')
        except Exception:
            p_value = None

    best_idx = prescribed_order.index(best_model)
    second_best_idx = prescribed_order.index(second_best_model)

    plt.figure(figsize=(max(2.4, len(prescribed_order) * 1.2), 5))
    sns.set_style("whitegrid")
    ax = sns.boxplot(
        data=df, hue='model_name', x='model_name', y=metric,
        palette=get_palette(prescribed_order), order=prescribed_order,
        showmeans=(mean_style == 'marker'),
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"} if mean_style == 'marker' else None,
        legend=False,
    )
    sns.stripplot(data=df, x='model_name', y=metric, order=prescribed_order, color="black", alpha=0.3, jitter=True, size=4)
    ax.set_xticklabels([wrap_for_plot(m) for m in prescribed_order])

    for i, model_name in enumerate(prescribed_order):
        if model_name in means:
            draw_mean_annotation(ax, i, means[model_name], value_fmt='{:.3f}', mean_style=mean_style)

    if p_value is not None:
        y_min, y_max = ax.get_ylim()
        span = y_max - y_min
        ax.plot([best_idx, second_best_idx], [y_max, y_max], 'k-', linewidth=2)
        ax.plot([best_idx, best_idx], [y_max - span * 0.02, y_max], 'k-', linewidth=2)
        ax.plot([second_best_idx, second_best_idx], [y_max - span * 0.02, y_max], 'k-', linewidth=2)
        ax.text((best_idx + second_best_idx) / 2, y_max + span * 0.02, sig_str_from_pvalue(p_value),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylim(y_min, y_max + span * 0.1)

    plt.ylabel(metric_label)
    plt.xlabel('Model')
    plt.tight_layout()
    save_fn = BASE_PATH / 'retrieval' / f'boxplot_{metric}.{FIG_FORMAT}'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().patch.set_alpha(0)
    plt.gca().set_facecolor('white')
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Boxplot saved to {save_fn}")


def ret_plot_barplot(df, prescribed_order, metric, metric_label):
    datasets = sorted(df['dataset_name'].unique())
    plt.figure(figsize=(max(8, 1.2 * len(datasets)), 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(
        data=df, x='dataset_name', y=metric, hue='model_name',
        palette=get_palette(prescribed_order), hue_order=prescribed_order,
    )
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.text(p.get_x() + p.get_width() / 2., height + 0.005, f'{height:.3f}',
                     ha='center', va='bottom', color='black', fontsize=7, fontweight='bold', rotation=90)
    plt.ylabel(metric_label)
    plt.xlabel('Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(0.5, -0.25), loc='center', ncol=len(prescribed_order))
    plt.tight_layout()
    save_fn = BASE_PATH / 'retrieval' / f'barplot_{metric}.{FIG_FORMAT}'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().patch.set_alpha(0)
    plt.gca().set_facecolor('white')
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Barplot saved to {save_fn}")


def run_retrieval(cfg, mean_style='marker'):
    df = pd.read_csv(CSV_DIR / cfg['csv'])
    metrics = ['top_1_acc', 'top_5_acc', 'top_10_acc', 'mAP']
    df[metrics] = df[metrics] / 100  # stored as 0-100; convert to 0-1 decimals for display
    metric_labels = {'top_1_acc': 'Top-1', 'top_5_acc': 'Top-5', 'top_10_acc': 'Top-10', 'mAP': 'mAP'}
    prescribed_order = cfg['order'] or list(df['model_name'].unique())
    color_defs, model_to_colorname = get_model_colors(prescribed_order)
    datasets = sorted(df['dataset_name'].unique())
    avg_stats = df.groupby('model_name', observed=True)[metrics].agg(['mean', 'std'])
    col_header = ' & '.join([f'\\textbf{{{metric_labels[m]}}}' for m in metrics])
    n_cols = len(metrics)
    lines = list(color_defs) + ['\\footnotesize', f'\\begin{{longtable}}{{ll{"c" * n_cols}}}']
    header_row = f'\\textbf{{Dataset}} & \\textbf{{Model}} & {col_header} \\\\'
    caption = (
        '\\caption{Per-dataset retrieval performance. Best result per dataset '
        'and metric in \\textbf{bold}. Average row reports mean$\\pm$std across datasets.}'
    )
    longtable_header(lines, 2 + n_cols, header_row, caption, 'tab:retrieval_per_dataset')
    for dataset in datasets:
        dataset_df = df[df['dataset_name'] == dataset]
        best_model_per_metric = {m: dataset_df.loc[dataset_df[m].idxmax(), 'model_name'] for m in metrics}
        for i, model in enumerate(prescribed_order):
            row = dataset_df[dataset_df['model_name'] == model]
            if row.empty:
                continue
            row = row.iloc[0]
            color_name = model_to_colorname.get(model)
            colored_label = f'\\textcolor{{{color_name}}}{{{model}}}' if color_name else model
            cells = [dataset.replace('_', ' ') if i == 0 else '', colored_label]
            for m in metrics:
                cell_str = f'{row[m]:.3f}'
                if best_model_per_metric[m] == model:
                    cell_str = f'\\textbf{{{cell_str}}}'
                cells.append(cell_str)
            lines.append(' & '.join(cells) + ' \\\\')
        lines.append('\\midrule')
    best_model_avg = {m: avg_stats[(m, 'mean')].idxmax() for m in metrics}
    for i, model in enumerate(prescribed_order):
        if model not in avg_stats.index:
            continue
        color_name = model_to_colorname.get(model)
        colored_label = f'\\textcolor{{{color_name}}}{{{model}}}' if color_name else model
        cells = ['\\textit{Average}' if i == 0 else '', colored_label]
        for m in metrics:
            mean_val, std_val = avg_stats.loc[model, (m, 'mean')], avg_stats.loc[model, (m, 'std')]
            cell_str = f'{mean_val:.3f}$\\pm${std_val:.3f}'
            if best_model_avg[m] == model:
                cell_str = f'\\textbf{{{cell_str}}}'
            cells.append(cell_str)
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\end{longtable}', '\\normalsize']
    save_latex(lines, BASE_PATH / 'retrieval' / 'retrieval_simple_table.tex')

    for m in metrics:
        ret_plot_boxplot(df, prescribed_order, m, metric_labels[m], mean_style)
        ret_plot_barplot(df, prescribed_order, m, metric_labels[m])


# ============================================================
# Data efficiency — columns: Dataset, Model, Seed, n_per_class,
# <metric columns>
# ============================================================
def de_compute_seed_level_stats(all_df, metric):
    per_seed = all_df.groupby(['Model', 'n_per_class', 'Seed'], observed=True)[metric].mean().reset_index()
    stats = per_seed.groupby(['Model', 'n_per_class'], observed=True)[metric].agg(['mean', 'std']).reset_index()
    return per_seed, stats


def de_plot(all_df, metric, prescribed_order):
    per_seed, _ = de_compute_seed_level_stats(all_df, metric)
    all_sentinel = per_seed['n_per_class'].max()
    plot_position_for_all = 70
    per_seed_plot = per_seed.copy()
    per_seed_plot['n_per_class_plot'] = per_seed_plot['n_per_class'].replace(all_sentinel, plot_position_for_all)
    plt.figure(figsize=(5, 5))
    sns.set_style("whitegrid")
    ax = sns.lineplot(
        data=per_seed_plot, x='n_per_class_plot', y=metric, hue='Model',
        hue_order=prescribed_order, palette=get_palette(prescribed_order),
        marker='o', errorbar='sd',
    )
    ax.set_xscale('log')
    shot_counts_plot = sorted(per_seed_plot['n_per_class_plot'].unique())
    ax.set_xticks(shot_counts_plot)
    ax.set_xticklabels(['All' if n == plot_position_for_all else str(int(n)) for n in shot_counts_plot])
    ax.minorticks_off()
    ax.set_xlabel('Number of training examples per class')
    ax.set_ylabel(metric)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [wrap_for_plot(l) for l in labels])
    plt.tight_layout()
    save_fn = BASE_PATH / 'data_efficiency' / f'data_efficiency_{metric}.{FIG_FORMAT}'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().patch.set_alpha(0)
    plt.gca().set_facecolor('white')
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_fn}")


def de_build_combined_table(all_df, metrics, prescribed_order):
    all_stats = {m: de_compute_seed_level_stats(all_df, m)[1] for m in metrics}
    n_shots = sorted(all_df['n_per_class'].unique())
    n_models, n_metrics = len(prescribed_order), len(metrics)
    col_spec = 'l' + ('l' * n_metrics) * n_models
    header_top = ' & ' + ' & '.join(f'\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{{m}}}}}' for m in prescribed_order) + ' \\\\'
    cmidrules = ' '.join(f'\\cmidrule(lr){{{2 + n_metrics*i}-{1 + n_metrics*(i+1)}}}' for i in range(n_models))
    header_bottom = '\\textbf{N per class}' + ' & ' + ' & '.join(' & '.join(f'\\textbf{{{m}}}' for m in metrics) for _ in prescribed_order) + ' \\\\'
    lines = ['\\begin{table}[h]', '\\centering',
             f'\\caption{{Data efficiency comparison ({", ".join(metrics)}; mean $\\pm$ std across seeds, averaged over datasets).}}',
             '\\label{tab:data_efficiency}', f'\\begin{{tabular}}{{{col_spec}}}', '\\toprule',
             header_top, cmidrules, header_bottom, '\\midrule']
    for n in n_shots:
        cells = [str(n)]
        for model in prescribed_order:
            for metric in metrics:
                stats = all_stats[metric]
                match = stats[(stats['n_per_class'] == n) & (stats['Model'] == model)]
                if len(match) == 0 or pd.isna(match['mean'].values[0]):
                    cells.append('--')
                else:
                    cells.append(f"{match['mean'].values[0]:.3f}$\\pm${match['std'].values[0]:.3f}")
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    save_latex(lines, BASE_PATH / 'data_efficiency' / 'data_efficiency_table.tex')


def run_data_efficiency(cfg):
    df = pd.read_csv(CSV_DIR / cfg['csv'])
    prescribed_order = cfg['order'] or sorted(df['Model'].unique())
    metrics = [m for m in cfg['metrics'] if m in df.columns]
    for metric in metrics:
        de_plot(df, metric, prescribed_order)
    de_build_combined_table(df, metrics, prescribed_order)


# ============================================================
# Computational efficiency — reads the raw long-format CSV
# (comp_efficiency_comparison.csv), one row per (n_bscans, model):
# n_bscans, input_size, mean_time_s, std_time_s, peak_mem_gb, model
# ============================================================
def ce_fit_scaling_exponent(model_df):
    sub = model_df.dropna(subset=['mean_time_s']).sort_values('n_bscans')
    log_n = np.log(sub['n_bscans'])
    log_t = np.log(sub['mean_time_s'])
    b, _ = np.polyfit(log_n, log_t, 1)
    return b


def ce_build_wide_table(dfs_by_model, reference_name):
    all_bscans = sorted(set().union(*[set(df['n_bscans']) for df in dfs_by_model.values()]))
    other_names = [k for k in dfs_by_model if k != reference_name]

    rows = []
    for n in all_bscans:
        row = {'n_bscans': n}
        values = {}
        for model_name, df in dfs_by_model.items():
            match = df[df['n_bscans'] == n]
            if len(match) == 0 or pd.isna(match['mean_time_s'].values[0]):
                t, m = None, None
            else:
                t = match['mean_time_s'].values[0]
                m = match['peak_mem_gb'].values[0]
            row[f'{model_name}_time'] = t
            row[f'{model_name}_mem'] = m
            values[model_name] = (t, m)

        ref_t, ref_m = values.get(reference_name, (None, None))
        for other_name in other_names:
            other_t, other_m = values.get(other_name, (None, None))
            if ref_t is not None and other_t is not None and ref_t > 0:
                row[f'{other_name}_speedup'] = other_t / ref_t
            else:
                row[f'{other_name}_speedup'] = None
            if ref_m is not None and other_m is not None and ref_m > 0:
                row[f'{other_name}_mem_ratio'] = other_m / ref_m
            else:
                row[f'{other_name}_mem_ratio'] = None
        rows.append(row)
    return pd.DataFrame(rows), other_names


def ce_plot_efficiency(dfs_by_model, model_names):
    palette = get_palette(model_names)

    all_bscans = set()
    for model_name in model_names:
        all_bscans.update(dfs_by_model[model_name]['n_bscans'].tolist())
    bscan_ticks = sorted(all_bscans)

    def fmt(x, pos):
        if x >= 1:
            return f'{x:.0f}' if x == int(x) else f'{x:.1f}'
        return f'{x:.2f}'

    def make_plot(value_col, ylabel, save_name):
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for model_name in model_names:
            df = dfs_by_model[model_name].dropna(subset=['mean_time_s']).sort_values('n_bscans')
            color = palette[model_name] if isinstance(palette, dict) else None
            ax.plot(df['n_bscans'], df[value_col], marker='o', label=model_name, color=color)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(bscan_ticks)
        ax.set_xticklabels([str(int(n)) for n in bscan_ticks])
        ax.minorticks_off()
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(1, 10))))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
        ax.set_xlabel('Number of B-scans')
        ax.set_ylabel(ylabel)
        ax.legend()

        plt.tight_layout()
        save_fn = BASE_PATH / 'comp_efficiency' / save_name
        save_fn.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().patch.set_alpha(0)
        plt.gca().set_facecolor('white')
        plt.savefig(save_fn, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_fn}")

    make_plot('mean_time_s', 'Inference time (s)', f'efficiency_plot_time.{FIG_FORMAT}')
    make_plot('peak_mem_gb', 'Peak GPU memory (GB)', f'efficiency_plot_mem.{FIG_FORMAT}')


def ce_print_latex_table(wide_df, model_names, reference_name, other_names):
    n_models = len(model_names)
    col_spec = 'l' + 'll' * n_models + 'll' * len(other_names)

    header_top_parts = [f'\\multicolumn{{2}}{{c}}{{\\textbf{{{name}}}}}' for name in model_names]
    for other_name in other_names:
        header_top_parts.append(f'\\multicolumn{{2}}{{c}}{{\\textbf{{{other_name}}} / \\textbf{{{reference_name}}}}}')
    header_top = ' & ' + ' & '.join(header_top_parts) + ' \\\\'

    n_groups = n_models + len(other_names)
    cmidrules = ' '.join(f'\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}' for i in range(n_groups))

    header_bottom_parts = ['\\textbf{B-scans}']
    for _ in model_names:
        header_bottom_parts.append('\\textbf{Time (s)} & \\textbf{Mem. (GB)}')
    for _ in other_names:
        header_bottom_parts.append('\\textbf{Time ratio} & \\textbf{Mem. ratio}')
    header_bottom = ' & '.join(header_bottom_parts) + ' \\\\'

    lines = ['\\begin{table}[h]', '\\centering',
             f'\\caption{{Inference time and peak GPU memory usage as a function of the '
             f'number of B-scans per volume. Ratio columns show the relative time and '
             f'memory usage of each model with respect to {reference_name}.}}',
             '\\label{tab:efficiency}', f'\\begin{{tabular}}{{{col_spec}}}', '\\toprule',
             header_top, cmidrules, header_bottom, '\\midrule']

    for _, row in wide_df.sort_values('n_bscans').iterrows():
        cells = [str(int(row['n_bscans']))]
        for name in model_names:
            t, m = row.get(f'{name}_time'), row.get(f'{name}_mem')
            cells += (['--', '--'] if (t is None or pd.isna(t)) else [f'{t:.3f}', f'{m:.2f}'])
        for other_name in other_names:
            sp, mr = row.get(f'{other_name}_speedup'), row.get(f'{other_name}_mem_ratio')
            cells += (['--', '--'] if (sp is None or pd.isna(sp)) else [f'{sp:.2f}$\\times$', f'{mr:.2f}$\\times$'])
        lines.append(' & '.join(cells) + ' \\\\')

    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    save_latex(lines, BASE_PATH / 'comp_efficiency' / 'efficiency_latex_table.tex')


def run_comp_efficiency(cfg):
    df = pd.read_csv(CSV_DIR / cfg['csv'])
    reference_name = cfg['reference_model']

    dfs_by_model = {name: sub for name, sub in df.groupby('model')}
    if reference_name not in dfs_by_model:
        raise ValueError(f"Reference model '{reference_name}' not found among "
                          f"models in CSV: {list(dfs_by_model.keys())}")
    model_names = sorted(dfs_by_model.keys())

    print("\nFitted scaling exponents (time ~ n_bscans^b):")
    for name in model_names:
        b = ce_fit_scaling_exponent(dfs_by_model[name])
        print(f"  {name}: b = {b:.3f}")

    wide_df, other_names = ce_build_wide_table(dfs_by_model, reference_name)
    ce_print_latex_table(wide_df, model_names, reference_name, other_names)

    ce_plot_efficiency(dfs_by_model, model_names)


# ============================================================
# CLI — exactly two real arguments (plus an optional style flag)
# ============================================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=list(KINDS.keys()))
    parser.add_argument('--level', default='dataset', choices=['dataset', 'volume', 'volume_wilcoxon'],
                         help='Significance test level, used only for segmentation kinds')
    parser.add_argument('--table-only', action='store_true')
    parser.add_argument('--mean-style', default='marker', choices=['marker', 'dots', 'diamond'],
                         help="How to show the mean on boxplots: 'marker' (default seaborn-style "
                              "circle + boxed number), 'dots' (row of small white squares spanning "
                              "the box width, value centered on top), or 'diamond' (white diamond "
                              "marker with a label placed just above it).")
    parser.add_argument('--format', default='svg', choices=['svg', 'pdf'],
                         help="Output format for saved figures. 'svg' (default) is easiest to "
                              "edit/compose in third-party vector tools; 'pdf' is also available.")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    FIG_FORMAT = args.format
    cfg = KINDS[args.kind]
    if cfg['type'] == 'classification':
        run_classification(cfg, args.table_only, args.mean_style)
    elif cfg['type'] == 'segmentation':
        run_segmentation(cfg, args.level, args.table_only, args.mean_style)
    elif cfg['type'] == 'retrieval':
        run_retrieval(cfg, args.mean_style)
    elif cfg['type'] == 'data_efficiency':
        run_data_efficiency(cfg)
    elif cfg['type'] == 'comp_efficiency':
        run_comp_efficiency(cfg)
