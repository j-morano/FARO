from pathlib import Path
from collections import OrderedDict
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf

from visualization.plot_utils import get_palette, LESION_CLASSES, get_model_colors


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--metric', type=str, default='Dice')
parser.add_argument('--experiment', type=str, default='SOTA')
parser.add_argument('--add_strip', action='store_true')
parser.add_argument('--level', default='dataset', type=str)
args = parser.parse_args()

SAVE_DIR = Path('/home/morano/SW/Documents/My_Papers/MIRAGEv2/results/seg')
SAVE_DIR = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/visualization/__plots_feat/seg')
SAVE_DIR = Path('/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/complete_results/seg')



INCLUDE_DATASETS = [
    "AMD_DME_3D",
    "AMD_SD_vol",
    "AROI_nosclera",
    "Duke_DME_nosclera",
    "Duke_iAMD_labeled_nosclera_70",
    "GOALS",
    "OCTAVE",
    "OIMHS",
    "RETOUCH",
    "RVO_Lesion",
    "UMN",
]


def load_seg_metric_df(dir, level, metric):
    """Reloads data for a single metric, mirroring plot_results' loading logic."""
    all_data = []
    for dataset in sorted(dir.iterdir()):
        if not dataset.is_dir() or (dataset.name not in INCLUDE_DATASETS and INCLUDE_DATASETS):
            continue
        for model_key, model_label in model_mapping.items():
            results_file_dir = dataset / model_key
            result_files = sorted(results_file_dir.glob('results*.csv'))
            for results_file in result_files:
                if results_file.stem not in ['results_test_all', 'results']:
                    dataset_name = results_file.stem.replace('results_', '')
                else:
                    dataset_name = dataset.name
                if dataset_name not in INCLUDE_DATASETS and INCLUDE_DATASETS:
                    continue
                df = pd.read_csv(results_file)
                if metric not in df.columns:
                    continue
                df['SplitDataset'] = df['Class'].apply(
                    lambda c: get_split_dataset_name(dataset_name, c)
                )
                if level in ('volume', 'volume_wilcoxon'):
                    for split_name, split_df in df.groupby('SplitDataset'):
                        volume_mean = split_df.groupby('ID')[metric].mean()
                        for volume_id, mean_value in volume_mean.items():
                            all_data.append({
                                'Dataset': split_name, 'Model': model_label,
                                'ID': volume_id, metric: mean_value,
                            })
                else:
                    for split_name, split_df in df.groupby('SplitDataset'):
                        mean_value = split_df.groupby('ID')[metric].mean().mean()
                        all_data.append({
                            'Dataset': split_name, 'Model': model_label, metric: mean_value,
                        })
    metric_df = pd.DataFrame(all_data)
    if not metric_df.empty:
        metric_df = metric_df[metric_df['Model'].isin(model_mapping.values())]
    return metric_df


def mixed_effects_test(volume_df, best_model, other_model, metric, higher_is_better=True):
    """
    Linear mixed-effects model treating dataset as random intercept,
    to properly account for volume-level correlation within datasets
    (avoids pseudoreplication when comparing at volume level).
    """
    subset = volume_df[volume_df['Model'].isin([best_model, other_model])].copy()
    subset['is_best'] = (subset['Model'] == best_model).astype(int)
    subset[metric] = pd.to_numeric(subset[metric], errors='coerce')
    subset = subset.dropna(subset=[metric])
    try:
        model = smf.mixedlm(f"{metric} ~ is_best", data=subset, groups=subset["Dataset"])
        result = model.fit()
        coef = result.params['is_best']
        p_value = result.pvalues['is_best']
        # For "lower is better" metrics (e.g. HD95), a negative coef means
        # best_model has lower values than other_model, which is the
        # expected direction — report p-value as-is (two-sided by default
        # from statsmodels); if you want a one-sided interpretation, halve
        # it only when the sign of coef matches the hypothesized direction.
        return coef, p_value
    except Exception as e:
        print(f"Mixed effects model failed: {e}")
        return None, None



def run_analysis(all_df, metric, prescribed_order, suffix_label=""):
    """
    Runs the full boxplot/barplot/table/statistical-test pipeline on a
    given subset of all_df. suffix_label is used to distinguish output
    filenames (e.g., '_layers', '_lesions', or '' for combined).
    """
    if all_df.empty:
        print(f"No data for split '{suffix_label}', skipping.")
        return

    # --- Statistical test: best vs second best (computed early so the
    # p-value is available for the boxplot annotation) ---
    higher_is_better = (metric == 'Dice')
    if higher_is_better:
        best_model = all_df.groupby('Model', observed=True)[metric].mean().idxmax()
        second_best_model = all_df.groupby('Model', observed=True)[metric].mean().nlargest(2).idxmin()
    else:
        best_model = all_df.groupby('Model', observed=True)[metric].mean().idxmin()
        second_best_model = all_df.groupby('Model', observed=True)[metric].mean().nsmallest(2).idxmax()

    print(f"\n[{suffix_label or 'combined'}] Comparing {best_model.replace(chr(10), ' ')} (best) vs "
          f"{second_best_model.replace(chr(10), ' ')} (second best), metric={metric}, "
          f"higher_is_better={higher_is_better}, level={args.level}")

    alt = 'greater' if higher_is_better else 'less'
    p_value = None

    if args.level == 'volume':
        coef, p_value = mixed_effects_test(all_df, best_model, second_best_model, metric)
        if coef is not None:
            print("Linear mixed-effects model (dataset as random intercept):")
            print(f"Coefficient (best - second_best): {coef:.4f}, p-value: {p_value:.4f}")
        else:
            print("LMM failed, falling back to Wilcoxon on dataset means instead.")
            best_data = all_df[all_df['Model'] == best_model].groupby('Dataset')[metric].mean().sort_index().values
            second_best_data = all_df[all_df['Model'] == second_best_model].groupby('Dataset')[metric].mean().sort_index().values
            try:
                stat, p_value = wilcoxon(best_data, second_best_data, alternative=alt)
                print(f"Wilcoxon test (fallback, dataset-level means): statistic={stat}, p-value={p_value:.4f}")
            except Exception as e:
                print(f"Wilcoxon test failed: {e}")
    elif args.level == 'volume_wilcoxon':
        # Wilcoxon directly on all individual volumes (paired by Dataset+ID)
        best_data = all_df[all_df['Model'] == best_model].sort_values(['Dataset', 'ID'])
        second_best_data = all_df[all_df['Model'] == second_best_model].sort_values(['Dataset', 'ID'])
        assert (best_data[['Dataset', 'ID']].values == second_best_data[['Dataset', 'ID']].values).all(), \
            "Mismatched (Dataset, ID) pairs between models"
        # Drop any (Dataset, ID) pair where either model has a NaN value,
        # since wilcoxon silently propagates NaN into the result otherwise
        valid_mask = best_data[metric].notna().values & second_best_data[metric].notna().values
        n_dropped = (~valid_mask).sum()
        if n_dropped > 0:
            print(f"Dropping {n_dropped} volume(s) with NaN {metric} in either model before Wilcoxon test.")
        best_vals = best_data[metric].values[valid_mask]
        second_best_vals = second_best_data[metric].values[valid_mask]
        try:
            stat, p_value = wilcoxon(best_vals, second_best_vals, alternative=alt)
            print(f"Wilcoxon test (all volumes pooled): n={len(best_vals)} volumes, statistic={stat}, p-value={p_value:.4f}")
        except Exception as e:
            print(f"Wilcoxon test failed: {e}")
    else:
        best_data = all_df[all_df['Model'] == best_model].sort_values('Dataset')[metric].values
        second_best_data = all_df[all_df['Model'] == second_best_model].sort_values('Dataset')[metric].values
        assert len(best_data) == len(second_best_data), \
            f"Mismatched lengths: {len(best_data)} vs {len(second_best_data)}"
        try:
            stat, p_value = wilcoxon(best_data, second_best_data, alternative=alt)
            print(f"Wilcoxon test ({alt}): n={len(best_data)} datasets, statistic={stat}, p-value={p_value:.4f}")
        except Exception as e:
            print(f"Wilcoxon test failed: {e}")

    best_idx = prescribed_order.index(best_model)
    second_best_idx = prescribed_order.index(second_best_model)

    # --- Boxplot: all datasets combined ---
    fig_w = len(prescribed_order)
    plt.figure(figsize=(fig_w, 5))
    sns.set_style("whitegrid")
    ax = sns.boxplot(
        data=all_df,
        hue='Model',
        x='Model',
        y=metric,
        palette=get_palette(prescribed_order),
        order=prescribed_order,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"},
        legend=False,
    )
    for label in ax.get_xticklabels():
        if 'PB-Mv2-Le' in label.get_text():
            label.set_fontweight('bold')
            label.set_color('black')
    if args.add_strip:
        sns.stripplot(
            data=all_df, x='Model', y=metric, order=prescribed_order,
            color="black", alpha=0.3, jitter=True,
        )
    means = all_df.groupby('Model', observed=True)[metric].mean()
    for i, model_name in enumerate(prescribed_order):
        if model_name in means:
            ax.text(
                i, means[model_name], f'{means[model_name]:.3f}',
                va='bottom', ha='center', color='black', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
            )

    # --- p-value bracket ---
    if p_value is not None:
        y_min, y_max = ax.get_ylim()
        line_y = y_max
        ax.plot([best_idx, second_best_idx], [line_y, line_y], 'k-', linewidth=2)
        ax.plot([best_idx, best_idx], [line_y - (y_max - y_min) * 0.02, line_y], 'k-', linewidth=2)
        ax.plot([second_best_idx, second_best_idx], [line_y - (y_max - y_min) * 0.02, line_y], 'k-', linewidth=2)

        mid_x = (best_idx + second_best_idx) / 2
        # if p_value < 0.00001:
        #     sig_str = '$p<0.00001$'
        # elif p_value < 0.0001:
        #     sig_str = '$p<0.0001$'
        if p_value < 0.001:
            sig_str = '$p<0.001$'
        elif p_value < 0.01:
            sig_str = '$p<0.01$'
        elif p_value < 0.05:
            sig_str = '$p<0.05$'
        else:
            sig_str = f'$p={p_value:.3f}$'
        ax.text(
            mid_x, line_y + (y_max - y_min) * 0.02, sig_str,
            ha='center', va='bottom', fontsize=11, fontweight='bold',
        )
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)

    char = "b"
    if metric == "Dice":
        char = "a"
    ax.text(
        -0.15, 1.025, char,
        transform=ax.transAxes,
        fontsize=16,
        fontweight='bold',
        va='top',
        ha='right',
    )

    # plt.title(f'Performance Distribution: {metric}{suffix_label} (All Datasets)')
    plt.ylabel(metric)
    plt.xlabel('Model')
    # plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    base_dir = 'ablation' if args.experiment == 'ablation' else 'SOTA_cx'
    save_fn = SAVE_DIR / base_dir / f'boxplot_{metric}{suffix_label}.pdf'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_fn}")

    # --- Barplot: per dataset ---
    datasets = sorted(all_df['Dataset'].unique())
    plt.figure(figsize=(max(4, 2 * len(datasets)), 4))
    ax = sns.barplot(
        data=all_df, x='Dataset', y=metric, hue='Model',
        palette=get_palette(prescribed_order), hue_order=prescribed_order,
        errorbar='sd', capsize=0.1,
    )
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height) and height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2., height + 0.005, f'{height:.3f}',
                ha='center', va='bottom', color='black', fontsize=6, fontweight='bold',
            )
    plt.title(f'Mean {metric}{suffix_label} per Dataset (with Std Dev)')
    plt.ylabel(metric)
    plt.xlabel('Dataset')
    plt.legend(title='Model', bbox_to_anchor=(0.5, -0.15), loc='center', ncol=len(prescribed_order))
    plt.tight_layout()
    save_fn = SAVE_DIR / base_dir / f'barplot_{metric}{suffix_label}.pdf'
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_fn}")

    # --- Tables ---
    print(f"\nMean {metric}{suffix_label} per Dataset:")
    mean_table = all_df.groupby(['Model', 'Dataset'], observed=True)[metric].mean().unstack()
    print(mean_table.round(4))
    print(f"\nOverall Mean and Std Dev ({metric}{suffix_label}):")
    overall_stats = all_df.groupby('Model', observed=True)[metric].agg(['mean', 'std'])
    print(overall_stats.round(4))


if args.experiment == 'sota':
    model_mapping = OrderedDict({
        "visionfm-base_linear_w": "VisionFM-Base\n(Linear)",
        "retfound_linear_w": "RetFound\n(Linear)",
        "octcube_linear_w": "OctCube\n(Linear)",
        "miragev1-large_linear_w": "MIRAGEv1-Large\n(Linear)",
        "PB-Mv2-Le_linear_w": "PB-Mv2-Le\n(Linear)",

        # "ibot-base_linear_w": "iBOT-Base\n(Linear)",
        # "ibot-base_convnext_w": "iBOT-Base\n(ConvNeXt)",
        "visionfm-base_convnext_w": "VisionFM-Base\n(ConvNeXt)",
        "retfound_convnext_w": "RetFound\n(ConvNeXt)",
        "octcube_convnext_w": "OctCube\n(ConvNeXt)",
        "miragev1-large_convnext_w": "MIRAGEv1-Large\n(ConvNeXt)",
        "PB-Mv2-Le_convnext_w": "PB-Mv2-Le\n(ConvNeXt)",
    })
    model_mapping = OrderedDict({
        "visionfm-base_convnext_w": "VisionFM",
        "octcube_convnext_w": "OCTCube",
        "retfound_convnext_w": "RETFound",
        "urfound_convnext_w": "UrFound",
        "miragev1-large_convnext_w": "MIRAGE",
        "PB-Mv2-Le_convnext_w": "CLOSER",
        # "PB-Mv2-Le_film_convnext_w": "CLOSER FILM",
        # "PB-Mv2-Le_convnext_w_pw": "CLOSER PRETRAINED",
        # "PB-Mv2-Le_convnext_xattn_w": "CLOSER XATTN",
        # "PB-Mv2-Le_gated_convnext_w": "CLOSER GATED",
        # "PB-Mv2-Le_xattn_full_w": "CLOSER XATTN-FULL",
    })
    # model_mapping = OrderedDict({
    #     "visionfm-base_cnn_w": "VisionFM",
    #     "octcube_cnn_w": "OCTCube",
    #     "retfound_cnn_w": "RETFound",
    #     "miragev1-large_cnn_w": "MIRAGE",
    #     "PB-Mv2-Le_cnn_w": "CLOSER",
    # })

elif args.experiment == 'ablation':
    model_mapping = OrderedDict({
        "ibot-base_linear_w": "iBOT-Base\n(Linear)",
        "nonPB-MAE-only_linear_w": "nonPB-MAE-only\n(Linear)",
        "nonPB-MIRAGE_linear_w": "nonPB-MIRAGE\n(Linear)",
        "nonPB-MIRAGE-tgt-layers_linear_w": "nonPB-MIRAGE-tgt-layers\n(Linear)",
        "PB-MIRAGE-tgt-layers_linear_w": "PB-MIRAGE-tgt-layers\n(Linear)",
        "PB-Mv2-Le_linear_w": "PB-Mv2-Le\n(Linear)",

        "ibot-base_convnext_w": "iBOT-Base\n(ConvNeXt)",
        "nonPB-MAE-only_convnext_w": "nonPB-MAE-only\n(ConvNeXt)",
        "nonPB-MIRAGE_convnext_w": "nonPB-MIRAGE\n(ConvNeXt)",
        "nonPB-MIRAGE-tgt-layers_convnext_w": "nonPB-MIRAGE-tgt-layers\n(ConvNeXt)",
        "PB-MIRAGE-tgt-layers_convnext_w": "PB-MIRAGE-tgt-layers\n(ConvNeXt)",
        "PB-Mv2-Le_convnext_w": "PB-Mv2-Le\n(ConvNeXt)",
    })
    model_mapping = OrderedDict({
        "ibot-base_linear_w": "iBOT",
        "nonPB-MAE-only_linear_w": "MAE",
        "nonPB-MIRAGE_linear_w": "MIRAGEv1\n(-SLO)",
        # "nonPB-MIRAGE-tgt-layers_linear_w": "MIRAGEv1\n(seg. targets)",
        # "PB-MIRAGE-tgt-layers_linear_w": "MIRAGEv1\n(seg. targets\n+ Pat. bal.)",
        "PB-Mv2-Le_linear_w": "MIRAGEv2 =\nMIRAGEv1\n(seg. targets\n+ Pat. bal.\n+lesions token)",
        "Mv2-Le-v6_linear_w": "Mv2-Le-v6\n(PaLe)",
        "Mv2-Le-v6-Dice_linear_w": "Mv2-Le-v6-Dice\n(PaLe)",
    })
    model_mapping = OrderedDict({
        "ibot-base_convnext_w": "iBOT",
        "nonPB-MAE-only_convnext_w": "MAE",
        "nonPB-MIRAGE_convnext_w": "MIRAGE",
        # "miragev1-base_linear_w": "MIRAGE",
        # "nonPB-MIRAGE-tgt-layers_convnext_w": "MIRAGE\n(seg. targets)",
        # "PB-MIRAGE-tgt-layers_convnext_w": "MIRAGE\n(seg. targets\n+ Pat. bal.)",
        # "PB-Mv2-Le_convnext_w": "CLOSER =\nMIRAGE\n(seg. targets\n+ Pat. bal.\n+lesions token)",
        "PB-Mv2-Le_convnext_w": "CLOSER",
    })
    model_mapping = OrderedDict({
        "ibot-base_linear_w": "iBOT",
        "nonPB-MAE-only_linear_w": "MAE",
        # "nonPB-MIRAGE_linear_w": "MIRAGE",
        # "miragev1-base_linear_w": "MIRAGE",
        # "nonPB-MIRAGE-tgt-layers_linear_w": "MIRAGE\n(seg. targets)",
        # "PB-MIRAGE-tgt-layers_linear_w": "MIRAGE\n(seg. targets\n+ Pat. bal.)",
        "PB-Mv2-Le_linear_w": "CLOSER",
    })
    model_mapping = OrderedDict({
        "ibot-base_linear_w": "iBOT",
        "nonPB-MAE-only_linear_w": "MAE",
        "nonPB-MIRAGE_linear_w": "MIRAGE",
        # "miragev1-base_linear_w": "MIRAGE",
        # "nonPB-MIRAGE-tgt-layers_linear_w": "MIRAGE\n(seg. targets)",
        # "PB-MIRAGE-tgt-layers_linear_w": "MIRAGE\n(seg. targets\n+ Pat. bal.)",
        "PB-Mv2-Le_linear_w": "CLOSER",
    })
    # model_mapping = OrderedDict({
    #     # "ibot-base_linear_w": "iBOT",
    #     # "miragev1-base_linear_w": "Rec+Seg\n(central-bscan)",
    #     "nonPB-MAE-only_linear_w": "Rec",
    #     "nonPB-MIRAGE_linear_w": "Rec+Seg",
    #     # "nonPB-MIRAGE-tgt-layers_linear_w": "Rec+Seg 2",
    #     # "PB-MIRAGE-tgt-layers_linear_w": "Rec+Seg 3",
    #     "PB-Mv2-Le_linear_w": "Rec+Seg+Les",
    # })
    # model_mapping = OrderedDict({
    #     "ibot-base_cnn_w": "iBOT",
    #     "nonPB-MAE-only_cnn_w": "MAE",
    #     "nonPB-MIRAGE_cnn_w": "MIRAGE",
    #     "miragev1-base_cnn_w": "MIRAGE\n(old data)",
    #     "PB-Mv2-Le_cnn_w": "CLOSER",
    # })
    # model_mapping = OrderedDict({
    #     # "ibot-base_cnn_w": "iBOT",
    #     "nonPB-MAE-only_cnn_w": "Rec.",
    #     "nonPB-MIRAGE_cnn_w": "Rec.+Seg.",
    #     # "miragev1-base_cnn_w": "MIRAGE\n(old data)",
    #     "PB-Mv2-Le_cnn_w": "Rec.+Seg.+Les.",
    # })


def get_split_dataset_name(dataset_name, class_name):
    """Split dataset into '_layers' / '_lesions' based on class name."""
    if class_name in LESION_CLASSES:
        return f"{dataset_name}_lesions"
    return f"{dataset_name}_layers"

def plot_results(dir, metric='Dice'):
    all_data = []
    for dataset in sorted(dir.iterdir()):
        if not dataset.is_dir() or (dataset.name not in INCLUDE_DATASETS and INCLUDE_DATASETS):
            continue
        for model_key, model_label in model_mapping.items():
            results_file_dir = dir / dataset.name / model_key
            result_files = sorted(results_file_dir.glob('results*.csv'))
            # if (
            #     'results_test_all.csv' in [f.name for f in result_files]
            #     and 'results.csv' in [f.name for f in result_files]
            # ):
            #     # Remove 'results.csv' if 'results_test_all.csv' exists
            #     result_files = [f for f in result_files if f.name != 'results.csv']
            for results_file in result_files:
                # results_file = results_file_dir / 'results_test_all.csv'
                # if not results_file.exists():
                #     results_file = results_file_dir / 'results.csv'
                # else:
                #     print(f"Found results_test_all.csv for {dataset.name} - {model_label.replace(chr(10), ' ')}")
                # if not results_file.exists():
                #     continue
                # if results_file.stem not in ['results_test_all', 'results']:
                if results_file.stem != 'results':
                    dataset_name = results_file.stem.replace('results_', '')
                else:
                    dataset_name = dataset.name
                if dataset_name not in INCLUDE_DATASETS and INCLUDE_DATASETS:
                    continue
                df = pd.read_csv(results_file)
                if metric not in df.columns:
                    continue
                # if args.level == 'volume':
                #     # Columns: ID,Class,Dice,IoU,HD95
                #     volume_mean = df.groupby('ID')[metric].mean()
                #     for volume_id, mean_value in volume_mean.items():
                #         all_data.append({
                #             'Dataset': dataset_name,
                #             'Model': model_label,
                #             'ID': volume_id,
                #             metric: mean_value,
                #         })
                # elif args.level == 'volume-lesion':
                #     for _, row in df.iterrows():
                #         all_data.append({
                #             'Dataset': dataset_name,
                #             'Model': model_label,
                #             'Class': row['Class'],
                #             metric: row[metric],
                #         })
                # elif args.level == 'dataset':
                #     mean_value = df.groupby('ID')[metric].mean().mean()  # patient mean, then dataset mean
                #     all_data.append({
                #         'Dataset': dataset_name,
                #         'Model': model_label,
                #         metric: mean_value,
                #     })
                if args.level in ('volume', 'volume_wilcoxon'):
                    df['SplitDataset'] = df['Class'].apply(
                        lambda c: get_split_dataset_name(dataset_name, c)
                    )
                    for split_name, split_df in df.groupby('SplitDataset'):
                        volume_mean = split_df.groupby('ID')[metric].mean()
                        for volume_id, mean_value in volume_mean.items():
                            all_data.append({
                                'Dataset': split_name,
                                'Model': model_label,
                                'ID': volume_id,
                                metric: mean_value,
                            })
                elif args.level == 'dataset':
                    df['SplitDataset'] = df['Class'].apply(
                        lambda c: get_split_dataset_name(dataset_name, c)
                    )
                    for split_name, split_df in df.groupby('SplitDataset'):
                        mean_value = split_df.groupby('ID')[metric].mean().mean()
                        all_data.append({
                            'Dataset': split_name,
                            'Model': model_label,
                            metric: mean_value,
                        })

    if not all_data:
        print("No results found.")
        return

    all_df = pd.DataFrame(all_data)
    all_df = all_df[all_df['Model'].isin(model_mapping.values())]
    prescribed_order = list(model_mapping.values())

    # Model counts check (on combined data, for sanity)
    for model_name in model_mapping.values():
        count = len(all_df[all_df['Model'] == model_name])
        print(f"Model: {model_name.replace(chr(10), ' ')}, Data Points: {count}")
    model_counts = all_df['Model'].value_counts()
    if len(model_counts.unique()) > 1:
        raise ValueError(f"Not all models have the same number of data points: {model_counts.to_dict()}")

    layers_df = all_df[all_df['Dataset'].str.endswith('_layers')]
    lesions_df = all_df[all_df['Dataset'].str.endswith('_lesions')]

    print("\n" + "="*70)
    print("COMBINED ANALYSIS (layers + lesions)")
    print("="*70)
    run_analysis(all_df, metric, prescribed_order, suffix_label="")

    print("\n" + "="*70)
    print("LAYERS ANALYSIS")
    print("="*70)
    run_analysis(layers_df, metric, prescribed_order, suffix_label="_layers")

    print("\n" + "="*70)
    print("LESIONS ANALYSIS")
    print("="*70)
    run_analysis(lesions_df, metric, prescribed_order, suffix_label="_lesions")


def get_sig_stars(p_value):
    if p_value is None:
        return ''
    if p_value < 0.001:
        return '$^{***}$'
    elif p_value < 0.01:
        return '$^{**}$'
    elif p_value < 0.05:
        return '$^{*}$'
    return ''


def compute_seg_pvalue(all_df, best_model, second_best_model, metric, higher_is_better, level):
    """Mirrors the logic in run_analysis, but returns just the p-value."""
    alt = 'greater' if higher_is_better else 'less'
    if level == 'volume':
        coef, p_value = mixed_effects_test(all_df, best_model, second_best_model, metric)
        if p_value is not None:
            return p_value
        # fallback
        best_data = all_df[all_df['Model'] == best_model].groupby('Dataset')[metric].mean().sort_index().values
        second_best_data = all_df[all_df['Model'] == second_best_model].groupby('Dataset')[metric].mean().sort_index().values
        try:
            _, p_value = wilcoxon(best_data, second_best_data, alternative=alt)
            return p_value
        except Exception:
            return None
    elif level == 'volume_wilcoxon':
        best_data = all_df[all_df['Model'] == best_model].sort_values(['Dataset', 'ID'])
        second_best_data = all_df[all_df['Model'] == second_best_model].sort_values(['Dataset', 'ID'])
        valid_mask = best_data[metric].notna().values & second_best_data[metric].notna().values
        best_vals = best_data[metric].values[valid_mask]
        second_best_vals = second_best_data[metric].values[valid_mask]
        try:
            _, p_value = wilcoxon(best_vals, second_best_vals, alternative=alt)
            return p_value
        except Exception:
            return None
    else:
        best_data = all_df[all_df['Model'] == best_model].sort_values('Dataset')[metric].values
        second_best_data = all_df[all_df['Model'] == second_best_model].sort_values('Dataset')[metric].values
        try:
            _, p_value = wilcoxon(best_data, second_best_data, alternative=alt)
            return p_value
        except Exception:
            return None


def print_seg_latex_table(dir, prescribed_order, level, metrics=('Dice', 'HD95')):
    """
    Builds a LaTeX table for segmentation results across the given
    metrics (default Dice, HD95), using the COMBINED (layers+lesions)
    level of analysis, styled with model-colored names and longtable
    pagination.
    """
    results_df = None
    pvalues = {}
    best_per_metric = {}
    second_best_per_metric = {}
    for metric in metrics:
        higher_is_better = (metric == 'Dice')
        metric_df = load_seg_metric_df(dir, level, metric)

        stats = metric_df.groupby('Model', observed=True)[metric].agg(['mean', 'std'])
        stats.columns = [f'{metric}_mean', f'{metric}_std']
        results_df = stats if results_df is None else results_df.join(stats, how='outer')

        if higher_is_better:
            best_model = metric_df.groupby('Model', observed=True)[metric].mean().idxmax()
            second_best_model = metric_df.groupby('Model', observed=True)[metric].mean().nlargest(2).idxmin()
        else:
            best_model = metric_df.groupby('Model', observed=True)[metric].mean().idxmin()
            second_best_model = metric_df.groupby('Model', observed=True)[metric].mean().nsmallest(2).idxmax()
        best_per_metric[metric] = best_model
        second_best_per_metric[metric] = second_best_model
        pvalues[metric] = compute_seg_pvalue(metric_df, best_model, second_best_model, metric, higher_is_better, level)

    color_defs, model_to_colorname = get_model_colors(prescribed_order)

    n_cols = len(metrics)
    col_header = ' & '.join([f'\\textbf{{{m}}}' for m in metrics])
    header_row = f'\\textbf{{Model}} & {col_header} \\\\'
    caption = (
        '\\caption{Performance comparison across models. Best results in \\textbf{bold}, '
        'second best \\underline{underlined}. Significance of best vs. second best: '
        '$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}'
    )

    lines = list(color_defs)
    lines.append('\\footnotesize')
    lines.append(f'\\begin{{longtable}}{{l{"l" * n_cols}}}')
    lines.append(caption + ' \\label{tab:seg_results} \\\\')
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
        model_label = model_name.replace(chr(10), ' ')
        color_name = model_to_colorname.get(model_label)
        cells = [f'\\textcolor{{{color_name}}}{{{model_label}}}' if color_name else model_label]
        for m in metrics:
            if model_name not in results_df.index:
                cells.append('--')
                continue
            mean_val = results_df.loc[model_name, f'{m}_mean']
            std_val = results_df.loc[model_name, f'{m}_std']
            if pd.isna(mean_val):
                cells.append('--')
                continue
            is_best = (model_name == best_per_metric[m])
            is_second = (model_name == second_best_per_metric[m])
            cell_str = f'{mean_val:.3f}$\\pm${std_val:.3f}'
            if is_best:
                stars = get_sig_stars(pvalues[m])
                cell_str = f'\\textbf{{{cell_str}}}{stars}'
                if color_name:
                    cell_str = f'\\textcolor{{{color_name}}}{{{cell_str}}}'
            elif is_second:
                cell_str = f'\\underline{{{cell_str}}}'
            cells.append(cell_str)
        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\end{longtable}')
    lines.append('\\normalsize')

    latex_str = '\n'.join(lines)
    print('\n' + '=' * 60)
    print('LATEX TABLE (Segmentation)')
    print('=' * 60)
    print(latex_str)

    base_dir = 'ablation' if args.experiment == 'ablation' else 'SOTA_cx'
    save_fn = SAVE_DIR / base_dir / 'seg_latex_table.tex'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    with open(save_fn, 'w') as f:
        f.write(latex_str)
    print(f'\nSaved to {save_fn}')


def compute_seg_pvalue_per_dataset(dataset_df, best_model, second_best_model, metric, higher_is_better):
    """
    Wilcoxon signed-rank test between best and second best model, paired by
    patient ID, within a single dataset (volume-level data).
    """
    alt = 'greater' if higher_is_better else 'less'
    best_data = dataset_df[dataset_df['Model'] == best_model].sort_values('ID')
    second_data = dataset_df[dataset_df['Model'] == second_best_model].sort_values('ID')
    if len(best_data) != len(second_data) or len(best_data) < 2:
        return None
    valid_mask = best_data[metric].notna().values & second_data[metric].notna().values
    best_vals = best_data[metric].values[valid_mask]
    second_vals = second_data[metric].values[valid_mask]
    if len(best_vals) < 2:
        return None
    try:
        _, p_value = wilcoxon(best_vals, second_vals, alternative=alt)
        return p_value
    except Exception:
        return None

def print_seg_latex_table_per_dataset(dir, prescribed_order, metrics=('Dice', 'HD95')):
    """
    Per-dataset breakdown of segmentation results, styled like the classification
    per-dataset table: colored model names, bold best (with significance stars,
    Wilcoxon signed-rank test paired by patient ID within each dataset),
    underlined second best, longtable spanning multiple pages.
    """
    metric_dfs = {m: load_seg_metric_df(dir, 'volume', m) for m in metrics}

    all_datasets = sorted(set().union(*[
        set(df['Dataset'].unique()) for df in metric_dfs.values() if not df.empty
    ]))

    color_defs, model_to_colorname = get_model_colors(prescribed_order)

    n_cols = len(metrics)
    col_header = ' & '.join([f'\\textbf{{{m}}}' for m in metrics])
    header_row = f'\\textbf{{Dataset}} & \\textbf{{Model}} & {col_header} \\\\'
    caption = (
        '\\caption{Per-dataset segmentation performance comparison across models. '
        'Best results in \\textbf{bold}, second best \\underline{underlined}. '
        'Statistical significance between the best and second best models in '
        'each dataset was assessed using the Wilcoxon signed-rank test, paired '
        'by patient ID ($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$).}'
    )

    lines = list(color_defs)
    lines.append('\\footnotesize')
    lines.append(f'\\begin{{longtable}}{{ll{"l" * n_cols}}}')
    lines.append(caption + ' \\label{tab:seg_per_dataset} \\\\')
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

    for dataset in all_datasets:
        results_df = None
        best_per_metric = {}
        second_best_per_metric = {}
        pvalues = {}

        for m in metrics:
            higher_is_better = (m == 'Dice')
            metric_df = metric_dfs[m]
            if metric_df.empty:
                continue
            dataset_df = metric_df[metric_df['Dataset'] == dataset]
            if dataset_df.empty:
                continue

            stats = dataset_df.groupby('Model', observed=True)[m].agg(['mean', 'std'])
            stats.columns = [f'{m}_mean', f'{m}_std']
            results_df = stats if results_df is None else results_df.join(stats, how='outer')

            means = dataset_df.groupby('Model', observed=True)[m].mean()
            if means.empty:
                continue
            if higher_is_better:
                best_model = means.idxmax()
                second_best_model = means.nlargest(2).idxmin() if len(means) >= 2 else None
            else:
                best_model = means.idxmin()
                second_best_model = means.nsmallest(2).idxmax() if len(means) >= 2 else None
            best_per_metric[m] = best_model
            if second_best_model is not None:
                second_best_per_metric[m] = second_best_model
                pvalues[m] = compute_seg_pvalue_per_dataset(
                    dataset_df, best_model, second_best_model, m, higher_is_better
                )

        if results_df is None:
            continue

        for i, model_name in enumerate(prescribed_order):
            if model_name not in results_df.index:
                continue
            color_name = model_to_colorname.get(model_name)
            cells = []
            cells.append(f'\\multirow{{{len(prescribed_order)}}}{{*}}{{{dataset.replace("_", " ")}}}' if i == 0 else '')
            cells.append(f'\\textcolor{{{color_name}}}{{{model_name}}}' if color_name else model_name)
            for m in metrics:
                if f'{m}_mean' not in results_df.columns:
                    cells.append('--')
                    continue
                mean_val = results_df.loc[model_name, f'{m}_mean']
                std_val = results_df.loc[model_name, f'{m}_std']
                if pd.isna(mean_val):
                    cells.append('--')
                    continue
                is_best = (best_per_metric.get(m) == model_name)
                is_second = (second_best_per_metric.get(m) == model_name)
                stars = get_sig_stars(pvalues.get(m)) if is_best else ''
                cell_str = f'{mean_val:.3f}$\\pm${std_val:.3f}{stars}'
                if is_best:
                    cell_str = f'\\textbf{{{cell_str}}}'
                    if color_name:
                        cell_str = f'\\textcolor{{{color_name}}}{{{cell_str}}}'
                elif is_second:
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
    print('LATEX TABLE (Segmentation, Per Dataset)')
    print('=' * 60)
    print(latex_str)

    base_dir = 'ablation' if args.experiment == 'ablation' else 'SOTA_cx'
    save_fn = SAVE_DIR / base_dir / 'seg_latex_table_per_dataset.tex'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    with open(save_fn, 'w') as f:
        f.write(latex_str)
    print(f'\nSaved to {save_fn}')


if args.experiment == 'ablation':
    print("ABLATION MODELS")
    dir = '/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/seg/Ablation_new_paper_lin/0/'
else:
    print("SOTA MODELS")
    dir = '/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/seg/SOTA_comparison_new/0/'


dir = Path(dir)
plot_results(dir, args.metric)

prescribed_order = list(model_mapping.values())
print_seg_latex_table(dir, prescribed_order, args.level, metrics=('Dice', 'HD952D'))
print_seg_latex_table_per_dataset(dir, prescribed_order, metrics=('Dice', 'HD952D'))
