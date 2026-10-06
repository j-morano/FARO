from pathlib import Path
from collections import OrderedDict
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
from visualization.plot_utils import get_palette

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--metric', type=str, default='Dice')
parser.add_argument('--experiment', type=str, default='SOTA')
parser.add_argument('--add_strip', action='store_true')
parser.add_argument('--level', default='dataset', type=str)
parser.add_argument('--seeds', type=str, default='2',
                     help='Comma-separated list of seed folder names to load')
args = parser.parse_args()

SEEDS = args.seeds.split(',')

SAVE_DIR = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/visualization/__plots_feat/seg')
INCLUDE_DATASETS = [
    "AMD_DME_3D",
    "AMD_SD_vol",
    "AROI_nosclera",
    "Duke_DME_nosclera",
    "Duke_iAMD_labeled_nosclera_70",
    "GOALS",
    "OCTAVE",
    "RETOUCH",
    "RVO_Lesion",
    "UMN",
]


def mixed_effects_test(volume_df, best_model, other_model, metric, group_col="Dataset"):
    """
    Linear mixed-effects model treating `group_col` as random intercept,
    to properly account for correlated observations within each group
    (avoids pseudoreplication).
    """
    subset = volume_df[volume_df['Model'].isin([best_model, other_model])].copy()
    subset['is_best'] = (subset['Model'] == best_model).astype(int)
    subset[metric] = pd.to_numeric(subset[metric], errors='coerce')
    subset = subset.dropna(subset=[metric])
    try:
        model = smf.mixedlm(f"{metric} ~ is_best", data=subset, groups=subset[group_col])
        result = model.fit()
        coef = result.params['is_best']
        p_value = result.pvalues['is_best']
        return coef, p_value
    except Exception as e:
        print(f"Mixed effects model failed: {e}")
        return None, None


def compute_pvalue(all_df, best_model, second_best_model, metric, higher_is_better, n_seeds):
    """
    Decide the correct statistical test given the data structure:
    - Multiple seeds -> LMM (random intercept for Dataset), using all
      seed-level (or seed x volume-level) observations, regardless of
      args.level, since seeds always introduce a repeated-measures
      structure per dataset.
    - Single seed -> plain Wilcoxon on dataset (or dataset x volume)
      values, since there is no seed-level correlation to model.
    """
    alt = 'greater' if higher_is_better else 'less'

    if n_seeds > 1:
        # Multiple seeds: use LMM with Dataset as random intercept.
        # This works whether args.level is 'dataset' (one row per
        # dataset x seed) or 'volume' (one row per dataset x seed x volume) —
        # in both cases Dataset is the grouping variable that needs to
        # absorb between-dataset variance while seeds/volumes provide the
        # repeated observations within each dataset.
        coef, p_value = mixed_effects_test(all_df, best_model, second_best_model, metric, group_col="Dataset")
        if coef is not None:
            print("Linear mixed-effects model (dataset as random intercept):")
            print(f"Coefficient (best - second_best): {coef:.4f}, p-value: {p_value:.4f}")
            return p_value
        else:
            print("LMM failed, falling back to Wilcoxon on dataset means instead.")

    # Single seed (or LMM fallback): plain Wilcoxon on dataset-level means
    best_data = all_df[all_df['Model'] == best_model].groupby('Dataset')[metric].mean().sort_index().values
    second_best_data = all_df[all_df['Model'] == second_best_model].groupby('Dataset')[metric].mean().sort_index().values
    try:
        stat, p_value = wilcoxon(best_data, second_best_data, alternative=alt)
        print(f"Wilcoxon test ({alt}, n={len(best_data)} datasets): statistic={stat}, p-value={p_value:.4f}")
        return p_value
    except Exception as e:
        print(f"Wilcoxon test failed: {e}")
        return None


def run_analysis(all_df, metric, prescribed_order, suffix_label="", n_seeds=1):
    """
    Runs the full boxplot/barplot/table/statistical-test pipeline on a
    given subset of all_df. suffix_label is used to distinguish output
    filenames (e.g., '_layers', '_lesions', or '' for combined).
    """
    if all_df.empty:
        print(f"No data for split '{suffix_label}', skipping.")
        return

    higher_is_better = (metric != 'HD95')
    if higher_is_better:
        best_model = all_df.groupby('Model', observed=True)[metric].mean().idxmax()
        second_best_model = all_df.groupby('Model', observed=True)[metric].mean().nlargest(2).idxmin()
    else:
        best_model = all_df.groupby('Model', observed=True)[metric].mean().idxmin()
        second_best_model = all_df.groupby('Model', observed=True)[metric].mean().nsmallest(2).idxmax()

    print(f"\n[{suffix_label or 'combined'}] Comparing {best_model.replace(chr(10), ' ')} (best) vs "
          f"{second_best_model.replace(chr(10), ' ')} (second best), metric={metric}, "
          f"higher_is_better={higher_is_better}, level={args.level}, n_seeds={n_seeds}")

    p_value = compute_pvalue(all_df, best_model, second_best_model, metric, higher_is_better, n_seeds)

    best_idx = prescribed_order.index(best_model)
    second_best_idx = prescribed_order.index(second_best_model)

    # --- Boxplot: all datasets combined ---
    fig_w = len(prescribed_order) * 1.5
    plt.figure(figsize=(fig_w, 7))
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

    if p_value is not None:
        y_min, y_max = ax.get_ylim()
        line_y = y_max
        ax.plot([best_idx, second_best_idx], [line_y, line_y], 'k-', linewidth=2)
        ax.plot([best_idx, best_idx], [line_y - (y_max - y_min) * 0.02, line_y], 'k-', linewidth=2)
        ax.plot([second_best_idx, second_best_idx], [line_y - (y_max - y_min) * 0.02, line_y], 'k-', linewidth=2)
        mid_x = (best_idx + second_best_idx) / 2
        if p_value < 0.00001:
            sig_str = '$p<0.00001$'
        elif p_value < 0.0001:
            sig_str = '$p<0.0001$'
        elif p_value < 0.001:
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
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.07)

    plt.title(f'Performance Distribution: {metric}{suffix_label} (All Datasets)')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    base_dir = 'ablation' if args.experiment == 'ablation' else 'SOTA'
    save_fn = SAVE_DIR / base_dir / f'boxplot_{metric}{suffix_label}.pdf'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_fn}")

    # --- Barplot: per dataset ---
    datasets = sorted(all_df['Dataset'].unique())
    plt.figure(figsize=(max(8, 5 * len(datasets)), 8))
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
        "visionfm-base_cnn_w": "VisionFM",
        "octcube_cnn_w": "OCTCube",
        "retfound_cnn_w": "RETFound",
        "urfound_cnn_w": "UrFound",
        "miragev1-large_cnn_w": "MIRAGE",
        "PB-Mv2-Le_cnn_w": "CLOSER",
    })
elif args.experiment == 'ablation':
    model_mapping = OrderedDict({
        "ibot-base_cnn_w": "iBOT",
        "nonPB-MAE-only_cnn_w": "MAE",
        "nonPB-MIRAGE_cnn_w": "MIRAGE",
        "miragev1-base_cnn_w": "MIRAGE\n(old data)",
        "PB-Mv2-Le_cnn_w": "CLOSER",
    })

LESION_CLASSES = {
    'Cyst', 'PED', 'SRF', 'Fluid', 'IRF', 'PED', 'SHRM', 'Fluid/cyst',
    'SRM', 'ERM', 'SES', 'HTD', 'FLU', 'HRM',
}


def get_split_dataset_name(dataset_name, class_name):
    """Split dataset into '_layers' / '_lesions' based on class name."""
    if class_name in LESION_CLASSES:
        return f"{dataset_name}_lesions"
    return f"{dataset_name}_layers"


def plot_results(base_dir, metric='Dice'):
    all_data = []
    for seed in SEEDS:
        seed_dir = base_dir / seed
        if not seed_dir.exists():
            print(f"WARNING: seed dir {seed_dir} does not exist, skipping.")
            continue
        for dataset in sorted(seed_dir.iterdir()):
            if not dataset.is_dir() or (dataset.name not in INCLUDE_DATASETS and INCLUDE_DATASETS):
                continue
            for model_key, model_label in model_mapping.items():
                results_file_dir = dataset / model_key
                result_files = sorted(results_file_dir.glob('results*.csv'))
                if (
                    'results_test_all.csv' in [f.name for f in result_files]
                    and 'results.csv' in [f.name for f in result_files]
                ):
                    result_files = [f for f in result_files if f.name != 'results.csv']
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
                    if args.level == 'volume':
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
                                    'Seed': seed,
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
                                'Seed': seed,
                                metric: mean_value,
                            })
    if not all_data:
        print("No results found.")
        return

    all_df = pd.DataFrame(all_data)
    all_df = all_df[all_df['Model'].isin(model_mapping.values())]
    prescribed_order = list(model_mapping.values())

    n_seeds = all_df['Seed'].nunique()
    print(f"Number of seeds found: {n_seeds} ({sorted(all_df['Seed'].unique())})")

    for model_name in model_mapping.values():
        count = len(all_df[all_df['Model'] == model_name])
        print(f"Model: {model_name.replace(chr(10), ' ')}, Data Points: {count}")
    model_counts = all_df['Model'].value_counts()
    if len(model_counts.unique()) > 1:
        raise ValueError(f"Not all models have the same number of data points: {model_counts.to_dict()}")

    layers_df = all_df[all_df['Dataset'].str.endswith('_layers')]
    lesions_df = all_df[all_df['Dataset'].str.endswith('_lesions')]

    print("\n" + "=" * 70)
    print("COMBINED ANALYSIS (layers + lesions)")
    print("=" * 70)
    run_analysis(all_df, metric, prescribed_order, suffix_label="", n_seeds=n_seeds)

    print("\n" + "=" * 70)
    print("LAYERS ANALYSIS")
    print("=" * 70)
    run_analysis(layers_df, metric, prescribed_order, suffix_label="_layers", n_seeds=n_seeds)

    print("\n" + "=" * 70)
    print("LESIONS ANALYSIS")
    print("=" * 70)
    run_analysis(lesions_df, metric, prescribed_order, suffix_label="_lesions", n_seeds=n_seeds)


if args.experiment == 'ablation':
    print("ABLATION MODELS")
    root_dir = Path('/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/seg/Ablation_new_paper/')
else:
    print("SOTA MODELS")
    root_dir = Path('/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/seg/SOTA_comparison_new_paper/')

plot_results(root_dir, args.metric)
