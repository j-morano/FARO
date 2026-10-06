from pathlib import Path
from collections import OrderedDict
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel
import statsmodels.formula.api as smf

from visualization.plot_utils import get_palette, get_model_colors



BASE_PATH = '/home/morano/SW/Documents/My_Papers/MIRAGEv2/results/'
BASE_PATH = '/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/complete_results/'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--metric', type=str, default='BAcc')
parser.add_argument('--experiment', type=str, default='2D')
parser.add_argument('--feat', action='store_true')
parser.add_argument('--table-only', action='store_true')
parser.add_argument('--wilcoxon', action='store_true')
args = parser.parse_args()


selected_seeds = [
    '0', '1', '2', '3', '4', '5', '6',
    # '0', '1',
    # '0',
]
IGNORE_DATASETS = [
    # 'UMNandDukeSrinivasan_to_Noor_cross'
    # 'GOALS',
    # 'OCTAVE',
    # 'OCT_MD_MS',
    '9C_complete',
    # 'OCTID',
    # 'Noor_to_UMNandDukeSrinivasan_cross',
    # 'UMNandDukeSrinivasan_to_Noor_cross',
]
# if not args.ablation:
#     IGNORE_DATASETS += [
#         '9C',
#     ]
selected_datasets = [
    #
    # '9C_complete',
    # 'Noor_to_UMNandDukeSrinivasan_cross',
    # 'UMNandDukeSrinivasan_to_Noor_cross',
    # 'Kermany_correct',
    # 'OCTID',
    # 'OCTDL',
]

if args.experiment == '2d':
    model_mapping = OrderedDict({
        "ibot-base_linear_w_cls": "iBOT-base\n(CLS)",
        "ibot-base_linear_w_cls_patch": "iBOT-base\n(CLS+Patch)",
        "octcube_linear_w_cls_patch": "OCTCube\n(CLS+Patch)",
        "octcube_linear_w_patch": "OCTCube\n(Patch)",
        "retfound_linear_w_patch": "RETFound\n(Patch)",
        "retfound_linear_w_cls_patch": "RETFound\n(CLS+Patch)",
        "visionfm-base_linear_w_cls": "VisionFM\n(CLS)",
        "visionfm-base_linear_w_cls_patch": "VisionFM\n(CLS+Patch)",
        "miragev1-large_linear_w_patch": "MIRAGEv1\n(Patch)",
        "PB-Mv2-Le_linear_w_conw_PaLe": "Mv2-Le\n(CONW-PaLe)",
        "Mv2-Large-099_linear_w_conw_PaLe": "Mv2-Le\n(CONW-PaLe\nLarge,99)",
    })
    model_mapping = OrderedDict({
        "retfound_linear_w_patch": "RETFound",
        "visionfm-base_linear_w_cls_patch": "VisionFM",
        "urfound_linear_w_cls_patch": "UrFound",
        "octcube_linear_w_cls_patch": "OCTCube",
        "miragev1-large_linear_w_patch": "MIRAGE",
        "PB-Mv2-Le_linear_w_conw_PaLe": "CLOSER",
    })

elif args.experiment == 'abl':
    model_mapping = OrderedDict({
        "nonPB-MAE-only_linear_w_patch": "nonPB-MAE-only\n(Patch)",
        "nonPB-MIRAGE_linear_w_patch": "nonPB-MIRAGE\n(Patch)",
        "nonPB-MIRAGE-tgt-layers_linear_w_patch": "nonPB-MIRAGE-tgt-layers\n(Patch)",
        "PB-MIRAGE-tgt-layers_linear_w_patch": "PB-MIRAGE-tgt-layers\n(Patch)",
        "PB-Mv2-Le_linear_w_conw_PaLe": "PB-Mv2-Le\n(PaLe)",
        "Mv2-Large-099_linear_w_conw_PaLe": "Mv2-Large-099\n(PaLe)",
    })
    model_mapping = OrderedDict({
        "miragev1-base_linear_w_patch": "MIRAGEv1",
        "nonPB-MIRAGE_linear_w_patch": "MIRAGEv1\n(new data)",
        "nonPB-MIRAGE-tgt-layers_linear_w_patch": "MIRAGEv1\n(seg. targets)",
        "PB-MIRAGE-tgt-layers_linear_w_patch": "MIRAGEv1\n(seg. targets\n+ Pat. bal.)",
        "PB-Mv2-Le_linear_w_conw_PaLe": "MIRAGEv2 =\nMIRAGEv1\n(seg. targets\n+ Pat. bal.\n+lesions token)",
        "Mv2-Le-v6_linear_w_conw_PaLe": "Mv2-Le-v6\n(PaLe)",
        # "Mv2-Le-v6-Dice_linear_w_conw_PaLe": "Mv2-Le-v6-Dice\n(PaLe)",
    })
    model_mapping = OrderedDict({
        # "miragev1-base_linear_w_patch": "MIRAGE\n(old data)",
        "nonPB-MAE-only_linear_w_patch": "MAE",
        "iBOT-base_linear_w_cls_patch": "iBOT",
        "nonPB-MIRAGE_linear_w_patch": "MIRAGE",
        # "nonPB-MIRAGE-tgt-layers_linear_w_patch": "MIRAGE\n(seg. targets)",
        # "PB-MIRAGE-tgt-layers_linear_w_patch": "MIRAGE\n(seg. targets\n+ Pat. bal.)",
        "PB-Mv2-Le_linear_w_conw_PaLe": "CLOSER",
    })

elif args.experiment == 'abl_components':
    model_mapping = OrderedDict({
        # "miragev1-base_linear_w_patch": "MIRAGE",
        # "miragev1-bscan-only_linear_w_patch": "Rec (old data)",
        "nonPB-MAE-only_linear_w_patch": "Rec",
        "nonPB-MIRAGE_linear_w_patch": "Rec+Seg",
        # "nonPB-MIRAGE-tgt-layers_linear_w_patch": "Rec.+Seg. 2",
        # "PB-MIRAGE-tgt-layers_linear_w_patch": "Rec.+Seg. 3",
        "PB-Mv2-Le_linear_w_conw_PaLe": "Rec+Seg+Les",
    })

elif args.experiment == 'abl_new_data':
    model_mapping = OrderedDict({
        "miragev1-base_linear_w_patch": "MIRAGE\n(old data)",
        "nonPB-MIRAGE_linear_w_patch": "MIRAGE\n(new data)",
    })

elif args.experiment == 'prets':
    model_mapping = OrderedDict({
        # "miragev1-base_linear_w_patch": "MIRAGE\n(old data)",
        "nonPB-MIRAGE_linear_w_patch": "MIRAGE",
        "iBOT-base_linear_w_cls_patch": "iBOT",
        "nonPB-MAE-only_linear_w_patch": "MAE",
        "PB-Mv2-Le_linear_w_conw_PaLe": "CLOSER",
    })


elif args.experiment == '3d':
    model_mapping = OrderedDict({
        "octcube3d_linear_w": "OCTCube",
        "octcube3d_60_linear_w_cls_patch": "OCTCube\n(60)",
        "miragefm3d_naive_linear_w": "Mv2\n+meanpool\n+max_avg",
        "miragefm3d_linear_w": "Mv2\n+mLSTM\n+mLSTM",
        "miragefm3d_xattn_xattn_linear_w": "Mv2\n+XAttn\n+XAttn",
        "miragefm3d_xattn_xattn_60_linear_w_2avg": "Mv2\n+XAttn\n+XAttn\n(60)",
        "miragefm3d_xattn_xlstm_linear_w": "Mv2\n+XAttn\n+mLSTM",
        "miragefm3d_xlstm_xattn_linear_w": "Mv2\n+mLSTM\n+XAttn",
    })
    model_mapping = OrderedDict({
        "octcube3d_60_linear_w_cls_patch": "OCTCube\n(60 B-scans)",
        # "miragefm3d_naive_linear_w": "MIRAGEv2\n(naive pooling\nall B-scans)",
        "miragefm3d_xattn_xattn_60_linear_w_2avg": "CLOSER\n(60 B-scans)",
        "miragefm3d_xattn_xattn_linear_w": "CLOSER\n(all B-scans)",
    })

elif args.experiment == '3dabl':
    model_mapping = OrderedDict({
        "miragefm3d_naive_linear_w": "CLOSER\n(naive pooling)",
        "miragefm3d_naive_proj_linear_w_2avg": "CLOSER\n(naive pooling\n+ proj)",
        "miragefm3d_xattn_xattn_linear_w": "CLOSER",
    })
    # selected_datasets = [
    #     'GAMMA',
    #     'M3Ret',
    #     'OCTAVE',
    # ]

elif args.experiment == 'chd':
    model_mapping = OrderedDict({
        "octcube-chd_linear_w_cls_patch": "OCTCube\n(CLS+Patch)",
        "miragev1-large-chd_linear_w_patch": "MIRAGEv1-Large\n(Patch)",
        "retfound-chd_linear_w_patch": "RETFound\n(Patch)",
        "visionfm-base-chd_linear_w_cls_patch": "VisionFM-Base\n(CLS+Patch)",
        "PB-Mv2-Le_ChD_linear_w_conw_PaLe": "PB-Mv2-Le\n(CONW-PaLe)",
    })
    model_mapping = OrderedDict({
        "octcube-chd_linear_w_cls_patch": "OCTCube",
        "miragev1-large-chd_linear_w_patch": "MIRAGE",
        "retfound-chd_linear_w_patch": "RETFound",
        "urfound-chd_linear_w_cls_patch": "UrFound",
        "visionfm-base-chd_linear_w_cls_patch": "VisionFM",
        "PB-Mv2-Le_ChD_linear_w_conw_PaLe": "CLOSER",
    })
    selected_seeds = [str(i) for i in range(14)]


pattern = "*/*/*/*/test_eval.csv"


def get_sig_stars(p_value):
    if p_value < 0.001:
        return '$^{***}$'
    elif p_value < 0.01:
        return '$^{**}$'
    elif p_value < 0.05:
        return '$^{*}$'
    return ''

def mixed_effects_test(full_df, best_model, other_model, metric):
    """
    Linear mixed-effects model treating dataset as random intercept,
    to properly account for seed correlation within datasets when few
    datasets are available (avoids pseudoreplication).
    """
    subset = full_df[full_df['Model'].isin([best_model, other_model])].copy()
    subset['is_best'] = (subset['Model'] == best_model).astype(int)
    subset[metric] = pd.to_numeric(subset[metric], errors='coerce')
    subset = subset.dropna(subset=[metric])
    try:
        model = smf.mixedlm(f"{metric} ~ is_best", data=subset, groups=subset["Dataset"])
        result = model.fit()
        coef = result.params['is_best']
        p_value = result.pvalues['is_best']
        return coef, p_value
    except Exception as e:
        print(f"Mixed effects model failed: {e}")
        return None, None

def compute_pvalue(full_df, best_model, other_model, metric, experiment):
    """Compute p-value between best model and another model for a given metric."""
    if experiment != 'chd' and not args.wilcoxon:
        _, p_value = mixed_effects_test(full_df, best_model, other_model, metric)
    elif experiment != 'chd' and args.wilcoxon:
        best_data = (
            full_df[full_df['Model'] == best_model]
            .groupby('Dataset')[metric].mean()
            .sort_index()
            .values
        )
        other_data = (
            full_df[full_df['Model'] == other_model]
            .groupby('Dataset')[metric].mean()
            .sort_index()
            .values
        )
        try:
            _, p_value = wilcoxon(best_data, other_data)
        except Exception:
            return None
    else:
        best_data = (
            full_df[full_df['Model'] == best_model]
            .sort_values('Run')[metric].values
        )
        other_data = (
            full_df[full_df['Model'] == other_model]
            .sort_values('Run')[metric].values
        )
        try:
            _, p_value = ttest_rel(best_data, other_data, alternative='greater')
        except Exception:
            return None
    return p_value


def print_latex_table(full_df, prescribed_order, experiment, metrics=None):
    if metrics is None:
        metrics = ['MCC', 'BAcc', 'AUROC', 'AP']

    # For each metric, compute mean and std per model
    rows = []
    for model_name in prescribed_order:
        model_df = full_df[full_df['Model'] == model_name]
        row = {'Model': model_name.replace('\n', ' ')}
        for m in metrics:
            if m not in model_df.columns:
                row[f'{m}_mean'] = float('nan')
                row[f'{m}_std'] = float('nan')
                continue
            if experiment == 'chd' or (args.experiment != 'chd' and not args.wilcoxon):
                # Single dataset: average over seeds, std over seeds
                values = pd.to_numeric(model_df[m], errors='coerce')
                row[f'{m}_mean'] = values.mean()
                row[f'{m}_std'] = values.std()
            else:
                # Multiple datasets: average over seeds per dataset first,
                # then mean and std over datasets
                per_dataset = (
                    model_df.groupby('Dataset')[m]
                    .apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
                )
                row[f'{m}_mean'] = per_dataset.mean()
                row[f'{m}_std'] = per_dataset.std()
        rows.append(row)

    results_df = pd.DataFrame(rows).set_index('Model')

    # Find best and second best per metric
    best_per_metric = {
        m: results_df[f'{m}_mean'].idxmax()
        for m in metrics
        if f'{m}_mean' in results_df.columns
    }
    if experiment != '3d':
        second_best_per_metric = {
            m: results_df[f'{m}_mean'].nlargest(2).idxmin()
            for m in metrics
            if f'{m}_mean' in results_df.columns
        }
    else:
        second_best_per_metric = {
            m: results_df[f'{m}_mean'].nlargest(3).idxmin()
            for m in metrics
            if f'{m}_mean' in results_df.columns
        }

    # Compute p-value: best model vs second best, per metric
    pvalues = {}
    for m in metrics:
        if f'{m}_mean' not in results_df.columns:
            continue
        best_model_m = best_per_metric[m]
        second_best_model_m = second_best_per_metric[m]
        best_model_full = next(
            (mn for mn in prescribed_order if mn.replace('\n', ' ') == best_model_m),
            None
        )
        second_best_model_full = next(
            (mn for mn in prescribed_order if mn.replace('\n', ' ') == second_best_model_m),
            None
        )
        if best_model_full is None or second_best_model_full is None:
            continue
        p = compute_pvalue(full_df, best_model_full, second_best_model_full, m, experiment)
        pvalues[m] = p

    # --- Build styled LaTeX longtable ---
    color_defs, model_to_colorname = get_model_colors(prescribed_order)
    color_defs = color_defs + ['\\definecolor{rowgray}{gray}{0.93}']

    n_cols = len(metrics)
    col_header = ' & '.join([f'\\textbf{{{m}}}' for m in metrics])
    header_row = f'\\textbf{{Model}} & {col_header} \\\\'

    lines = list(color_defs)
    lines.append('\\footnotesize')
    lines.append(f'\\begin{{longtable}}{{l{"l" * n_cols}}}')
    lines.append(
        '\\caption{Performance comparison across models. Best results in '
        '\\textbf{bold}, second best \\underline{underlined}. Significance of '
        'best vs. second best: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}'
    )
    if args.experiment == '3d':
        lines.append('\\label{tab:cls_3d} \\\\')
    elif args.experiment == 'chd':
        lines.append('\\label{tab:chd} \\\\')
    elif args.experiment == '2d':
        lines.append('\\label{tab:cls_2d} \\\\')
    else:
        lines.append('\\label{tab:results} \\\\')
    lines.append('\\toprule')
    lines.append(header_row)
    lines.append('\\midrule')
    lines.append('\\endfirsthead')

    lines.append(f'\\multicolumn{{{1 + n_cols}}}{{c}}{{\\tablename\\ \\thetable{{}} -- continued from previous page}} \\\\')
    lines.append('\\toprule')
    lines.append(header_row)
    lines.append('\\midrule')
    lines.append('\\endhead')

    lines.append('\\midrule')
    lines.append(f'\\multicolumn{{{1 + n_cols}}}{{r}}{{\\textit{{continued on next page}}}} \\\\')
    lines.append('\\endfoot')

    lines.append('\\bottomrule')
    lines.append('\\endlastfoot')

    for model_name in prescribed_order:
        model_label = model_name.replace('\n', ' ')
        is_best = {m: (model_label == best_per_metric.get(m, '')) for m in metrics}
        color_name = model_to_colorname.get(model_label)

        cells = [f'\\textcolor{{{color_name}}}{{{model_label}}}' if color_name else model_label]
        for m in metrics:
            if f'{m}_mean' not in results_df.columns:
                cells.append('--')
                continue
            mean_val = results_df.loc[model_label, f'{m}_mean']
            std_val = results_df.loc[model_label, f'{m}_std']
            if pd.isna(mean_val):
                cells.append('--')
                continue

            # Stars only on best model row
            stars = ''
            if is_best[m]:
                p = pvalues.get(m, None)
                if p is not None:
                    stars = get_sig_stars(p)

            cell_str = f'{mean_val:.3f}$\\pm${std_val:.3f}{stars}'

            if is_best[m]:
                cell_str = f'\\textbf{{{cell_str}}}'
                if color_name:
                    cell_str = f'\\textcolor{{{color_name}}}{{{cell_str}}}'
            elif model_label == second_best_per_metric.get(m, ''):
                cell_str = f'\\underline{{{cell_str}}}'
            cells.append(cell_str)

        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\end{longtable}')
    lines.append('\\normalsize')

    latex_str = '\n'.join(lines)
    print('\n' + '='*60)
    print('LATEX TABLE')
    print('='*60)
    print(latex_str)

    # Save to file
    save_fn = Path(BASE_PATH, f"{args.experiment}/{args.experiment}_latex_table.tex")
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    with open(save_fn, 'w') as f:
        f.write(latex_str)
    print(f'LaTeX table saved to {save_fn}')


def compute_pvalue_per_dataset(dataset_df, best_model, other_model, metric):
    """Paired one-tailed Student's t-test between best and other model, within a single dataset (paired by seed/Run)."""
    best_data = (
        dataset_df[dataset_df['Model'] == best_model]
        .sort_values('Run')[metric]
    )
    other_data = (
        dataset_df[dataset_df['Model'] == other_model]
        .sort_values('Run')[metric]
    )
    best_data = pd.to_numeric(best_data, errors='coerce').values
    other_data = pd.to_numeric(other_data, errors='coerce').values
    if len(best_data) != len(other_data) or len(best_data) < 2:
        return None
    try:
        _, p_value = ttest_rel(best_data, other_data, alternative='greater')
    except Exception:
        return None
    return p_value



def print_latex_table_per_dataset(full_df, prescribed_order, experiment, metrics=None):
    """
    Same idea as print_latex_table, but broken down per dataset instead of
    averaged across datasets, rendered as a longtable so it spans multiple
    pages with repeating headers. Significance (best vs. second best) is
    computed per dataset using the one-tailed Student's t-test, paired by
    seed/Run, matching the per-dataset testing convention used in the paper.
    """
    if metrics is None:
        metrics = ['MCC', 'BAcc', 'AUROC', 'AP']
    datasets = sorted(full_df['Dataset'].unique())
    n_cols = len(metrics)

    color_defs, model_to_colorname = get_model_colors(prescribed_order)

    col_header = ' & '.join([f'\\textbf{{{m}}}' for m in metrics])
    header_row = f'\\textbf{{Dataset}} & \\textbf{{Model}} & {col_header} \\\\'
    caption = (
        '\\caption{Per-dataset performance comparison across models. '
        'Best results in \\textbf{bold}, second best \\underline{underlined}. '
        'Statistical significance between the best and second best models in '
        'each dataset was assessed using the one-tailed Student\'s t-test '
        '($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$).}'
    )

    lines = list(color_defs)
    lines.append('\\footnotesize')  # shrink to help it fit; drop or change if you prefer
    lines.append(f'\\begin{{longtable}}{{ll{"l" * n_cols}}}')
    lines.append(caption + f' \\label{{tab:{experiment}_per_dataset}} \\\\')
    lines.append('\\toprule')
    lines.append(header_row)
    lines.append('\\midrule')
    lines.append('\\endfirsthead')

    lines.append(f'\\multicolumn{{{2 + n_cols}}}{{c}}{{\\tablename\\ \\thetable{{}} -- continued from previous page}} \\\\')
    lines.append('\\toprule')
    lines.append(header_row)
    lines.append('\\midrule')
    lines.append('\\endhead')

    lines.append('\\midrule')
    lines.append(f'\\multicolumn{{{2 + n_cols}}}{{r}}{{\\textit{{continued on next page}}}} \\\\')
    lines.append('\\endfoot')

    lines.append('\\bottomrule')
    lines.append('\\endlastfoot')

    for dataset in datasets:
        dataset_df = full_df[full_df['Dataset'] == dataset]
        # Per-model mean/std for this dataset only (averaged over seeds)
        rows = []
        for model_name in prescribed_order:
            model_df = dataset_df[dataset_df['Model'] == model_name]
            row = {'Model': model_name.replace('\n', ' ')}
            for m in metrics:
                if m not in model_df.columns or model_df.empty:
                    row[f'{m}_mean'] = float('nan')
                    row[f'{m}_std'] = float('nan')
                    continue
                values = pd.to_numeric(model_df[m], errors='coerce')
                row[f'{m}_mean'] = values.mean()
                row[f'{m}_std'] = values.std()
            rows.append(row)
        results_df = pd.DataFrame(rows).set_index('Model')

        best_per_metric = {}
        second_best_per_metric = {}
        for m in metrics:
            if f'{m}_mean' not in results_df.columns:
                continue
            valid = results_df[f'{m}_mean'].dropna()
            if len(valid) < 1:
                continue
            best_per_metric[m] = valid.idxmax()
            if len(valid) >= 2:
                second_best_per_metric[m] = valid.nlargest(2).idxmin()

        pvalues = {}
        for m in metrics:
            if m not in best_per_metric or m not in second_best_per_metric:
                continue
            best_model_full = next(
                (mn for mn in prescribed_order if mn.replace('\n', ' ') == best_per_metric[m]), None
            )
            second_best_model_full = next(
                (mn for mn in prescribed_order if mn.replace('\n', ' ') == second_best_per_metric[m]), None
            )
            if best_model_full is None or second_best_model_full is None:
                continue
            pvalues[m] = compute_pvalue_per_dataset(dataset_df, best_model_full, second_best_model_full, m)

        for i, model_name in enumerate(prescribed_order):
            model_label = model_name.replace('\n', ' ')
            if model_label not in results_df.index:
                continue
            is_best = {m: (best_per_metric.get(m) == model_label) for m in metrics}
            is_second = {m: (second_best_per_metric.get(m) == model_label) for m in metrics}
            color_name = model_to_colorname.get(model_label)
            cells = []
            cells.append(f'\\multirow{{{len(prescribed_order)}}}{{*}}{{{dataset.replace("_", " ")}}}' if i == 0 else '')
            cells.append(f'\\textcolor{{{color_name}}}{{{model_label}}}' if color_name else model_label)
            for m in metrics:
                if f'{m}_mean' not in results_df.columns:
                    cells.append('--')
                    continue
                mean_val = results_df.loc[model_label, f'{m}_mean']
                std_val = results_df.loc[model_label, f'{m}_std']
                if pd.isna(mean_val):
                    cells.append('--')
                    continue
                stars = ''
                if is_best[m]:
                    p = pvalues.get(m)
                    if p is not None:
                        stars = get_sig_stars(p)
                cell_str = f'{mean_val:.3f}$\\pm${std_val:.3f}{stars}'
                if is_best[m]:
                    cell_str = f'\\textbf{{{cell_str}}}'
                    if color_name:
                        cell_str = f'\\textcolor{{{color_name}}}{{{cell_str}}}'
                elif is_second[m]:
                    cell_str = f'\\underline{{{cell_str}}}'
                cells.append(cell_str)
            lines.append(' & '.join(cells) + ' \\\\')
        lines.append('\\midrule')

    if lines[-1] == '\\midrule':
        lines.pop()
    lines.append('\\end{longtable}')
    lines.append('\\normalsize')

    latex_str = '\n'.join(lines)
    print('\n' + '=' * 60)
    print('LATEX TABLE (PER DATASET)')
    print('=' * 60)
    print(latex_str)
    save_fn = Path(BASE_PATH, f"{experiment}/{experiment}_latex_table_per_dataset.tex")
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    with open(save_fn, 'w') as f:
        f.write(latex_str)
    print(f'Per-dataset LaTeX table saved to {save_fn}')


def print_markdown_tables(full_df, prescribed_order, experiment, metrics=None):
    """
    Simple, no-frills markdown tables: one per dataset, plus an overall
    average table. No bold/underline/significance markers.
    """
    if metrics is None:
        metrics = ['MCC', 'BAcc', 'AUROC', 'AP']

    def build_results_df(sub_df, per_dataset_avg=False):
        rows = []
        for model_name in prescribed_order:
            model_df = sub_df[sub_df['Model'] == model_name]
            row = {'Model': model_name.replace('\n', ' ')}
            for m in metrics:
                if m not in model_df.columns or model_df.empty:
                    row[f'{m}_mean'] = float('nan')
                    row[f'{m}_std'] = float('nan')
                    continue
                if per_dataset_avg:
                    # average over seeds per dataset first, then over datasets
                    per_dataset = (
                        model_df.groupby('Dataset')[m]
                        .apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
                    )
                    row[f'{m}_mean'] = per_dataset.mean()
                    row[f'{m}_std'] = per_dataset.std()
                else:
                    values = pd.to_numeric(model_df[m], errors='coerce')
                    row[f'{m}_mean'] = values.mean()
                    row[f'{m}_std'] = values.std()
            rows.append(row)
        return pd.DataFrame(rows).set_index('Model')

    def render_table(results_df, title):
        block = [f'\n### {title}\n']
        block.append('| Model | ' + ' | '.join(metrics) + ' |')
        block.append('|---|' + '---|' * len(metrics))
        for model_label in results_df.index:
            cells = [model_label]
            for m in metrics:
                mean_val = results_df.loc[model_label, f'{m}_mean']
                std_val = results_df.loc[model_label, f'{m}_std']
                cells.append('--' if pd.isna(mean_val) else f'{mean_val:.3f} ± {std_val:.3f}')
            block.append('| ' + ' | '.join(cells) + ' |')
        return block

    all_lines = []
    for dataset in sorted(full_df['Dataset'].unique()):
        dataset_df = full_df[full_df['Dataset'] == dataset]
        results_df = build_results_df(dataset_df, per_dataset_avg=False)
        all_lines += render_table(results_df, dataset)

    avg_results_df = build_results_df(full_df, per_dataset_avg=True)
    all_lines += render_table(avg_results_df, 'Average')

    md_str = '\n'.join(all_lines)
    print(md_str)

    save_fn = Path(BASE_PATH, f"{experiment}/{experiment}_markdown_tables.md")
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    with open(save_fn, 'w') as f:
        f.write(md_str)
    print(f'\nMarkdown tables saved to {save_fn}')


def print_csv_table(full_df, prescribed_order, experiment, metrics=None):
    """
    Same as print_markdown_tables, but writes one flat CSV with a 'Dataset'
    column (plus an 'Average' row per model) instead of markdown text.
    """
    if metrics is None:
        metrics = ['MCC', 'BAcc', 'AUROC', 'AP']

    def build_results_df(sub_df, per_dataset_avg=False):
        rows = []
        for model_name in prescribed_order:
            model_df = sub_df[sub_df['Model'] == model_name]
            row = {'Model': model_name.replace('\n', ' ')}
            for m in metrics:
                if m not in model_df.columns or model_df.empty:
                    row[f'{m}_mean'] = float('nan')
                    row[f'{m}_std'] = float('nan')
                    continue
                if per_dataset_avg:
                    per_dataset = (
                        model_df.groupby('Dataset')[m]
                        .apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
                    )
                    row[f'{m}_mean'] = per_dataset.mean()
                    row[f'{m}_std'] = per_dataset.std()
                else:
                    values = pd.to_numeric(model_df[m], errors='coerce')
                    row[f'{m}_mean'] = values.mean()
                    row[f'{m}_std'] = values.std()
            rows.append(row)
        return pd.DataFrame(rows).set_index('Model')

    all_rows = []
    for dataset in sorted(full_df['Dataset'].unique()):
        dataset_df = full_df[full_df['Dataset'] == dataset]
        results_df = build_results_df(dataset_df, per_dataset_avg=False)
        results_df = results_df.reset_index()
        results_df.insert(0, 'Dataset', dataset)
        all_rows.append(results_df)

    avg_results_df = build_results_df(full_df, per_dataset_avg=True).reset_index()
    avg_results_df.insert(0, 'Dataset', 'Average')
    all_rows.append(avg_results_df)

    csv_df = pd.concat(all_rows, ignore_index=True)

    save_fn = Path(BASE_PATH, f"{experiment}/{experiment}_results_table.csv")
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(save_fn, index=False)
    print(f'CSV table saved to {save_fn}')


def map_model_name(folder_name):
    for key, pretty_name in model_mapping.items():
        if key.lower() == folder_name.lower():
            return pretty_name
    return folder_name

def plot_model_performance(root_dir, metric='BAcc'):
    data_list = []
    root_path = Path(root_dir)
    print(f"Looking for files in {root_path} with pattern {pattern}")

    for csv_path in root_path.glob(pattern):
        run_id = csv_path.parts[-2]
        dataset = csv_path.parts[-5]
        model_folder = csv_path.parts[-4]

        if selected_seeds:
            if run_id not in selected_seeds:
                continue
        if selected_datasets:
            if dataset not in selected_datasets:
                continue
        elif dataset in IGNORE_DATASETS:
            # This elif ONLY runs if selected_datasets is empty/falsy
            continue

        model_display_name = map_model_name(model_folder)

        try:
            df = pd.read_csv(csv_path)
            best_row = df[df['Epoch'] == 'Best'].copy()
            best_row[metric] = pd.to_numeric(best_row[metric], errors='coerce')

            best_row['Run'] = run_id
            best_row['Dataset'] = dataset
            best_row['Model'] = model_display_name

            data_list.append(best_row)
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    # Assert the models have the same number of data points
    model_counts = set()
    for model_name in model_mapping.values():
        model_count = sum(1 for df in data_list if df['Model'].iloc[0] == model_name)
        print(f"Model: {model_name.replace('\n', ' ')}, Data Points: {model_count}")
        model_counts.add(model_count)
        # You can add an assertion here if you want to enforce equal counts
    if len(model_counts) > 1:
        print(
            "⚠️ Warning: Not all models have the same number of data points."
            f" Counts: {model_counts}"
        )
        raise ValueError("Not all models have the same number of data points.")

    if not data_list:
        print("No data found.")
        return

    full_df = pd.concat(data_list, ignore_index=True)

    # Remove models not in the mapping (if any)
    full_df = full_df[full_df['Model'].isin(model_mapping.values())]
    # Enforce the order defined in the OrderedDict
    prescribed_order = list(model_mapping.values())
    # full_df['Model'] = pd.Categorical(
    #     full_df['Model'],
    #     categories=prescribed_order,
    #     ordered=True
    # )

    best_model = full_df.groupby('Model', observed=True)[metric].mean().idxmax()
    second_best_model = full_df.groupby('Model', observed=True)[metric].mean().nlargest(2).idxmin()
    if args.experiment == '3d':
        # get the third instead
        second_best_model = full_df.groupby('Model', observed=True)[metric].mean().nlargest(3).idxmin()
    # Add line connecting best and second best with p-value
    best_idx = prescribed_order.index(best_model)
    second_best_idx = prescribed_order.index(second_best_model)
    if args.experiment != 'chd' and not args.wilcoxon:  # or whatever flag you use
        coef, p_value = mixed_effects_test(full_df, best_model, second_best_model, metric)
        print(f"\nLinear mixed-effects model (dataset as random intercept) "
              f"between {best_model.replace(chr(10), ' ')} and {second_best_model.replace(chr(10), ' ')}:")
        print(f"Coefficient: {coef:.4f}, p-value: {p_value:.4f}")
    elif args.experiment != 'chd' and args.wilcoxon:
        # Compute statistical test between best and second best method
        # best_data = full_df[full_df['Model'] == best_model].sort_values(['Dataset', 'Run'])[metric].values
        # second_best_data = full_df[full_df['Model'] == second_best_model].sort_values(['Dataset', 'Run'])[metric].values
        best_data = (
            full_df[full_df['Model'] == best_model]
            .groupby('Dataset')[metric].mean()
            .sort_index()
            .values
        )
        second_best_data = (
            full_df[full_df['Model'] == second_best_model]
            .groupby('Dataset')[metric].mean()
            .sort_index()
            .values
        )
        # print()
        # print(best_model.replace('\n', ' '), best_data.values)
        # print(second_best_model.replace('\n', ' '), second_best_data.values)
        stat, p_value = wilcoxon(best_data, second_best_data)
        print(f"\nWilcoxon signed-rank test between {best_model.replace('\n', ' ')} and {second_best_model.replace('\n', ' ')}:")
        print(f"Statistic: {stat}, p-value: {p_value}")
    else:
        best_data = (
            full_df[full_df['Model'] == best_model]
            .sort_values('Run')[metric].values
        )
        second_best_data = (
            full_df[full_df['Model'] == second_best_model]
            .sort_values('Run')[metric].values
        )
        stat, p_value = ttest_rel(best_data, second_best_data, alternative='greater')
        print(f"\nPaired t-test (one-tailed) between {best_model.replace('\n', ' ')} and {second_best_model.replace('\n', ' ')}:")
        print(f"Statistic: {stat}, p-value: {p_value}")

    print_latex_table(full_df, prescribed_order, args.experiment)

    # print_latex_table_per_dataset(full_df, prescribed_order, args.experiment)
    #
    # print_markdown_tables(full_df, prescribed_order, args.experiment)
    #
    # print_csv_table(full_df, prescribed_order, args.experiment)

    if args.table_only:
        return

    # Plotting
    if args.experiment == '3d':
        fig_w = len(prescribed_order) * 1.5
    else:
        fig_w = len(prescribed_order)
    plt.figure(figsize=(fig_w, 5))
    sns.set_style("whitegrid")

    # Create the Boxplot
    ax = sns.boxplot(
        data=full_df,
        hue='Model',
        x='Model',
        y=metric,
        palette=get_palette(prescribed_order),
        order=prescribed_order,
        showmeans=True,  # Adds a point for the mean
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"},
        # Deactivate legend
        legend=False
    )

    # BOLD SPECIFIC X-AXIS LABEL
    for label in ax.get_xticklabels():
        if 'MIRAGE-OCT' in label.get_text():
            label.set_fontweight('bold')
            label.set_color('black') # Optional: make it pop more
            # label.set_fontsize(12) # Optional: make it slightly larger

    # Optional: Overlay individual data points (seeds/datasets)
    sns.stripplot(
        data=full_df,
        x='Model',
        y=metric,
        order=prescribed_order,
        color="black",
        alpha=0.3,
        jitter=True,
        # Change size of points if needed
        size=4
    )

    # Add text labels for the MEANS
    # We calculate the means manually to place the text correctly
    means = full_df.groupby('Model', observed=True)[metric].mean()
    for i, model_name in enumerate(prescribed_order):
        if model_name in means:
            ax.text(
                i,
                means[model_name], # + 0.01,
                f'{means[model_name]:.3f}',
                va='bottom',
                ha='center',
                color='black',
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )

    # Get y-axis limits and position line near the top
    y_min, y_max = ax.get_ylim()
    line_y = y_max

    # Draw connecting line with vertical ends
    ax.plot(
        [best_idx, second_best_idx],
        [line_y, line_y],
        'k-',
        linewidth=2
    )
    ax.plot([best_idx, best_idx], [line_y - (y_max-y_min)*0.02, line_y], 'k-', linewidth=2)
    ax.plot([second_best_idx, second_best_idx], [line_y - (y_max-y_min)*0.02, line_y], 'k-', linewidth=2)

    # Add p-value text
    mid_x = (best_idx + second_best_idx) / 2
    sig_str = f'$p={p_value:.4f}$'
    # if p_value < 0.00001:
    #     sig_str = '$p<0.00001$'
    # elif p_value < 0.0001:
    #     sig_str = '$p<0.0001$'
    if p_value < 0.001:
        sig_str = '$p<0.001$'
    elif p_value < 0.01:
        sig_str = f'$p={p_value:.4f}$'
        sig_str = '$p<0.01$'
    elif p_value < 0.05:
        sig_str = f'$p={p_value:.4f}$'
        sig_str = '$p<0.05$'

    ax.text(
        mid_x,
        line_y + (y_max-y_min)*0.02,
        sig_str,
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold'
    )

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


    # plt.title(f'Performance Distribution: {metric} (All Seeds & Datasets)')
    plt.ylabel(metric)
    plt.xlabel('Model')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    # if args.ablation:
    #     plt.ylim(0.78, 0.9)
    plt.ylim(y_min, y_max + (y_max-y_min)*0.07)  # Add some space for the p-value text
    # save_fn = Path(f'./visualization/__plots_feat/{args.experiment}/{args.experiment}_boxplot_model_performance_{metric}.png')
    save_fn = Path(BASE_PATH, f"{args.experiment}/{args.experiment}_boxplot_model_performance_{metric}.png")
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(save_fn, dpi=300)
    # Also PDF
    pdf_save_fn = save_fn.with_suffix('.pdf')
    plt.savefig(pdf_save_fn, dpi=300)
    print(f"Plot saved to {save_fn}")
    plt.close()

    # 1. Create the plot (Note: Seaborn barplot defaults to the mean)
    datasets = sorted(full_df['Dataset'].unique())
    plt.figure(figsize=(max(8, 5 * len(datasets)), 8))
    ax = sns.barplot(
        data=full_df,
        x='Dataset',
        y=metric,
        hue='Model',
        palette=get_palette(prescribed_order),
        hue_order=prescribed_order,
        # 'errorbar' replaces 'ci' in newer Seaborn versions
        # 'sd' shows standard deviation; default is 'ci' (95% confidence interval)
        errorbar='sd',
        capsize=.1     # Adds small horizontal lines to the top of error bars
    )

    # Plot the mean values on top of the bars
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.text(
                p.get_x() + p.get_width() / 2.,
                height + 0.005,  # Slightly above the bar
                f'{height*100:.1f}', # Format as percentage
                ha='center',
                va='bottom',
                color='black',
                fontweight='bold'
            )


    # 2. Styling and Labels
    plt.title(f'Mean Performance per Dataset: {metric} (with Std Dev)')
    plt.ylabel(metric)
    # if args.ablation:
    #     plt.ylim(0.75, 1)
    plt.xlabel('Dataset')

    # 3. Legend handling
    plt.legend(
        title='Model',
        bbox_to_anchor=(0.5, -0.15),
        loc='center',
        # As many columns as models
        ncol=len(prescribed_order),
    )

    # 4. Save the figure
    plt.tight_layout()
    # save_fn = Path(f'./visualization/__plots_feat/{args.experiment}/{args.experiment}_barplot_model_performance_per_dataset_{metric}.png')
    save_fn = Path(BASE_PATH, f"{args.experiment}/{args.experiment}_barplot_model_performance_per_dataset_{metric}.png")
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    # Also save as PDF
    pdf_save_fn = save_fn.with_suffix('.pdf')
    plt.savefig(pdf_save_fn, dpi=300, bbox_inches='tight')

    print(f"Plot saved to {save_fn}")
    plt.close()

    # Also print tables with results per dataset (models as rows)
    print(f"\nMean Performance per Dataset ({metric}):")
    mean_table = full_df.groupby(['Model', 'Dataset'], observed=True)[metric].mean().unstack()
    print(mean_table.round(4))

    # Also print table with mean and standard deviation over all points
    print(f"\nOverall Mean and Std Dev ({metric}):")
    overall_stats = full_df.groupby('Model', observed=True)[metric].agg(['mean', 'std'])
    print(overall_stats.round(4))



    # # Compute statistical test between best and VisionFM
    # if not args.ablation:
    #     visionfm_data = full_df[full_df['Model'] == 'VisionFM\n(CLS)'][metric]
    #     stat, p_value = wilcoxon(best_data, visionfm_data)
    #     print(f"\nWilcoxon signed-rank test between {best_model.replace('\n', ' ')} and VisionFM (CLS) for metric {metric}:")
    #     print(f"Statistic: {stat}, p-value: {p_value}")




# '/home/morano/Bookmarks/SSHFS/msc_server/MIRAGEv2_eval/__output/cls/prueba_v7',
# '__output/cls/ablation',
if args.experiment in ['abl', 'abl_new_data', 'prets', 'abl_components']:
    print("ABLATION MODELS")
    root_dir = '/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/cls/ablation_features/0'
elif args.experiment in ['3d', '3dabl']:
    print("3D MODELS")
    root_dir = '/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/cls_3d/SOTA_comparison_features/0'
elif args.experiment == '2d':
    print("2D MODELS")
    root_dir = '/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/cls/SOTA_comparison_features/0'
elif args.experiment == 'chd':
    root_dir = '/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/cls/SOTA_change_detection/0'



# Usage
plot_model_performance(
    Path(root_dir),
    metric=args.metric
)
