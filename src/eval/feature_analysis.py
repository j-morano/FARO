from pathlib import Path
import argparse
import json
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from mutils import misc


def get_results(remaining_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True)
    parser.add_argument("--datasets_root", type=str, default="./_datasets/Classification_3D/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        '--accum_iter', default=1, type=int,
        help='Accumulate gradient iterations (for increasing the effective'
            ' batch size under memory constraints). (default: %(default)s)'
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results.')
    args = parser.parse_args(remaining_args)
    misc.fix_seeds(args.seed)
    dataset_name = Path(args.root).parent.stem
    model_name = Path(args.root).stem
    features_fn = (
        Path(args.root) / "cached_features.npz"
    )
    assert features_fn.exists(), features_fn
    args.datasets_root = Path(args.datasets_root)
    args.output_dir = Path("./__outputs_feature_analysis")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = np.load(features_fn)
    features_train = torch.tensor(features["features_train"])[:, 0, 0]
    targets_train = torch.tensor(features["targets_train"])[:, 0]
    features_val = torch.tensor(features["features_val"])[:, 0, 0]
    targets_val = torch.tensor(features["targets_val"])[:, 0]
    features_test = torch.tensor(features["features_test"])[:, 0, 0]
    targets_test = torch.tensor(features["targets_test"])[:, 0]
    num_classes = features["num_classes"].item()
    print(features_train.shape, targets_train.shape)
    print(features_val.shape, targets_val.shape)
    print(features_test.shape, targets_test.shape)
    print("num_classes", num_classes)
    dataset_dir = args.datasets_root / dataset_name
    classes_list = sorted((dataset_dir / "train").glob("*/"))
    def get_samples_ordered(subset):
        samples = []
        root = dataset_dir / subset
        for cls in sorted(root.iterdir()):
            if not cls.is_dir():
                continue
            for vol in sorted(cls.iterdir()):
                if vol.suffix == '.npz':
                    samples.append(vol.stem)
        return samples
    sample_ids_train = get_samples_ordered("train")
    sample_ids_val   = get_samples_ordered("val")
    sample_ids_test  = get_samples_ordered("test")
    ####### t-SNE
    class_names = [c.stem for c in classes_list]
    all_features = torch.cat([features_train, features_val, features_test], dim=0)
    all_targets  = torch.cat([targets_train,  targets_val,  targets_test],  dim=0)
    print(f"Running t-SNE on {len(all_features)} samples...")
    embeddings = TSNE(
        n_components=2,
        perplexity=30,
        random_state=args.seed,
        n_jobs=-1,
    ).fit_transform(all_features.numpy())
    fig, ax = plt.subplots(figsize=(12, 10))
    for class_idx in range(num_classes):
        mask = all_targets.numpy() == class_idx
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            label=class_names[class_idx],
            s=10,
            alpha=0.6,
        )
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title(f"{model_name} – t-SNE (train + val + test)")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    ax.set_facecolor('white')
    fig.savefig(args.output_dir / f"{dataset_name}_{model_name}_tsne.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    metadata_fn = dataset_dir / 'metadata.json'
    if metadata_fn.exists():
        with open(metadata_fn, 'r') as f:
            metadata = json.load(f)
        all_ids = sample_ids_train + sample_ids_val + sample_ids_test
        # Extract vendor for each sample, in the same order as embeddings
        vendors = [
            metadata[sid]["Vendor"] if sid in metadata else "Unknown"
            for sid in all_ids
        ]
        vendors = np.array(vendors)
        unique_vendors = sorted(set(vendors))
        fig, ax = plt.subplots(figsize=(12, 10))
        for vendor in unique_vendors:
            mask = vendors == vendor
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                label=vendor,
                s=10,
                alpha=0.6,
            )
        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.set_title(f"{model_name} – t-SNE by vendor (train + val + test)")
        fig.tight_layout()
        fig.patch.set_alpha(0)
        ax.set_facecolor('white')
        fig.savefig(args.output_dir / f"{dataset_name}_{model_name}_tsne_vendor.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Vendors found: {unique_vendors}")
    ####### Retrieval
    K = [1, 5, 10]
    # Normalize features for cosine similarity
    def l2_normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    train_feats = l2_normalize(features_train).numpy()  # (N_train, D)
    train_labels = targets_train.numpy()
    query_feats  = l2_normalize(torch.cat([features_val, features_test], dim=0)).numpy()
    query_labels = torch.cat([targets_val, targets_test], dim=0).numpy()
    print(f"Computing cosine similarity ({len(query_feats)} queries x {len(train_feats)} gallery)...")
    sim_matrix = cosine_similarity(query_feats, train_feats)  # (N_query, N_train)
    results = {}
    for k in K:
        top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]  # (N_query, k)
        top_k_labels  = train_labels[top_k_indices]              # (N_query, k)
        correct = (top_k_labels == query_labels[:, None]).any(axis=1)
        acc = correct.mean() * 100
        results[k] = acc
        print(f"  Top-{k} accuracy: {acc:.2f}%")
    # Also compute mAP (mean Average Precision)
    def compute_map(sim_matrix, query_labels, train_labels):
        num_queries = sim_matrix.shape[0]
        average_precisions = []
        for i in range(num_queries):
            query_label = query_labels[i]
            similarities = sim_matrix[i]
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_labels = train_labels[sorted_indices]
            # Compute precision at each rank
            relevant = (sorted_labels == query_label).astype(int)
            cumulative_relevant = np.cumsum(relevant)
            precision_at_k = cumulative_relevant / (np.arange(len(relevant)) + 1)
            # Average Precision for this query
            if relevant.sum() > 0:
                average_precision = (precision_at_k * relevant).sum() / relevant.sum()
                average_precisions.append(average_precision)
        return np.mean(average_precisions) * 100 if average_precisions else 0.0
    map = compute_map(sim_matrix, query_labels, train_labels)
    # Save results
    results_fn = args.output_dir / "retrieval.csv"
    # Columns: "model_name", "dataset_name", "top_1_acc", "top_5_acc", "top_10_acc", "mAP"
    df = pd.DataFrame([{
        "model_name": model_name,
        "dataset_name": dataset_name,
        "top_1_acc": results[1],
        "top_5_acc": results[5],
        "top_10_acc": results[10],
        "mAP": map,
    }])
    if results_fn.exists() and not args.overwrite:
        df_existing = pd.read_csv(results_fn)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(results_fn, index=False)
    print(f"Saved to {results_fn}")
    ####### Qualitative Retrieval Examples
    N_QUERIES = 5
    N_NEIGHBORS = 5
    rng = np.random.default_rng(args.seed)
    # Build index: sample_id -> (class_idx, subset)
    def build_id_map(subset):
        id_map = {}
        for class_dir in classes_list:
            target = dataset_dir / subset / class_dir.stem
            if not target.is_dir():
                continue
            for p in target.rglob("*"):
                if p.is_file() and p.suffix.lower() == ".npz":
                    id_map[p.stem] = (class_dir.stem, p)
        return id_map
    id_map_train = build_id_map("train")
    id_map_query = build_id_map("val")
    id_map_query.update(build_id_map("test"))
    def load_central_bscan(path: Path) -> np.ndarray:
        vol = np.load(path)["vol"]  # (N_slices, H, W)
        return vol[vol.shape[0] // 2]
    # Pick random query indices
    query_indices = rng.choice(len(query_feats), size=N_QUERIES, replace=False)
    fig, axes = plt.subplots(
        N_QUERIES, 1 + N_NEIGHBORS,
        figsize=(3 * (1 + N_NEIGHBORS), 3 * N_QUERIES),
    )
    all_query_ids = sample_ids_val + sample_ids_test
    for row, q_idx in enumerate(query_indices):
        q_id    = all_query_ids[q_idx]
        q_class, q_path = id_map_query[q_id]
        # Top-k neighbors from train
        top_k_idx = np.argsort(sim_matrix[q_idx])[::-1][:N_NEIGHBORS]
        # Query image
        ax = axes[row, 0]
        ax.imshow(load_central_bscan(q_path), cmap="gray")
        ax.set_title(f"Query\n{q_class}", fontsize=7, color="steelblue")
        ax.axis("off")
        # Blue border for query
        for spine in ax.spines.values():
            spine.set_edgecolor("steelblue")
            spine.set_linewidth(3)
            spine.set_visible(True)
        # Neighbor images
        for col, n_idx in enumerate(top_k_idx):
            n_id    = sample_ids_train[n_idx]
            n_class, n_path = id_map_train[n_id]
            score   = sim_matrix[q_idx, n_idx]
            correct = n_class == q_class
            ax = axes[row, col + 1]
            ax.imshow(load_central_bscan(n_path), cmap="gray")
            ax.set_title(f"#{col+1} ({score:.2f})\n{n_class}", fontsize=7,
                         color="green" if correct else "red")
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor("green" if correct else "red")
                spine.set_linewidth(3)
                spine.set_visible(True)
    fig.suptitle(f"{model_name} – Retrieval examples", fontsize=12)
    fig.tight_layout()
    fig.patch.set_alpha(0)
    for a in axes.flat:
        a.set_facecolor('white')
    fig.savefig(args.output_dir / f"{dataset_name}_{model_name}_retrieval_qualitative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved qualitative retrieval plot.")


def get_table(remaining_args):
    fn = Path("./__outputs_feature_analysis/retrieval.csv")
    if not fn.exists():
        print(f"File not found: {fn}")
        return
    df = pd.read_csv(fn)
    metrics = ['top_1_acc', 'top_5_acc', 'top_10_acc', 'mAP']
    models = df['model_name'].unique()
    if len(models) != 2:
        print(f"Expected exactly 2 models for pairwise comparison, found {len(models)}: {models}")
        return
    # Model display name mapping — adjust as needed
    model_display = {
        'miragefm3d_xattn_xattn_linear_w_2avg': 'CLOSER',
        'octcube3d_60_linear_w_cls_patch': 'OCTCube',
    }
    def get_label(m):
        return model_display.get(m, m)
    # Ensure paired datasets align
    pivot_check = df.groupby(['dataset_name', 'model_name']).size()
    if (pivot_check != 1).any():
        print("WARNING: some dataset/model combinations have != 1 row.")
    model_a, model_b = models[0], models[1]
    data_a = df[df['model_name'] == model_a].sort_values('dataset_name')
    data_b = df[df['model_name'] == model_b].sort_values('dataset_name')
    assert (data_a['dataset_name'].values == data_b['dataset_name'].values).all(), \
        "Dataset order mismatch between models after sorting!"
    n_datasets = len(data_a)
    # Compute mean/std per model, and Wilcoxon p-value per metric
    rows = {}
    pvalues = {}
    for model in [model_a, model_b]:
        model_df = df[df['model_name'] == model]
        rows[model] = {}
        for m in metrics:
            rows[model][f'{m}_mean'] = model_df[m].mean()
            rows[model][f'{m}_std'] = model_df[m].std()
    # Determine which model is "best" per metric (higher is always better here)
    best_model = {}
    for m in metrics:
        best_model[m] = model_a if rows[model_a][f'{m}_mean'] > rows[model_b][f'{m}_mean'] else model_b
    for m in metrics:
        a_vals = data_a[m].values
        b_vals = data_b[m].values
        try:
            stat, p = wilcoxon(a_vals, b_vals, alternative='greater' if best_model[m] == model_a else 'less')
            pvalues[m] = p
        except Exception as e:
            print(f"Wilcoxon failed for {m}: {e}")
            pvalues[m] = None
    def get_sig_stars(p):
        if p is None:
            return ''
        if p < 0.001:
            return '$^{***}$'
        elif p < 0.01:
            return '$^{**}$'
        elif p < 0.05:
            return '$^{*}$'
        return ''
    metric_labels = {
        'top_1_acc': 'Top-1',
        'top_5_acc': 'Top-5',
        'top_10_acc': 'Top-10',
        'mAP': 'mAP',
    }
    # Build LaTeX table
    col_header = ' & '.join([f'\\textbf{{{metric_labels[m]}}}' for m in metrics])
    n_cols = len(metrics)
    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('\\centering')
    lines.append(f'\\caption{{Retrieval performance across {n_datasets} datasets. Best results in \\textbf{{bold}}. '
                  f'Significance vs. second-best (Wilcoxon signed-rank test, one-sided): '
                  f'$^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$.}}')
    lines.append('\\label{tab:retrieval}')
    lines.append(f'\\begin{{tabular}}{{l{"c" * n_cols}}}')
    lines.append('\\toprule')
    lines.append(f'\\textbf{{Model}} & {col_header} \\\\')
    lines.append('\\midrule')
    for model in [model_a, model_b]:
        label = get_label(model)
        cells = [label]
        for m in metrics:
            mean_val = rows[model][f'{m}_mean']
            std_val = rows[model][f'{m}_std']
            cell_str = f'{mean_val:.2f}$\\pm${std_val:.2f}'
            if best_model[m] == model:
                stars = get_sig_stars(pvalues[m])
                cell_str = f'\\textbf{{{cell_str}}}{stars}'
            cells.append(cell_str)
        lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    latex_str = '\n'.join(lines)
    print('\n' + '=' * 60)
    print('LATEX TABLE')
    print('=' * 60)
    print(latex_str)
    # Also print raw p-values for reference
    print('\nRaw p-values:')
    for m in metrics:
        print(f'  {metric_labels[m]}: p={pvalues[m]:.4f}' if pvalues[m] is not None else f'  {metric_labels[m]}: N/A')
    save_fn = Path("./__outputs_feature_analysis/retrieval_latex_table.tex")
    with open(save_fn, 'w') as f:
        f.write(latex_str)
    print(f'\nSaved to {save_fn}')


def print_retrieval_simple_table(remaining_args):
    """
    Simple per-dataset + average retrieval table: one row per dataset per
    model, plus an averaged row per model at the bottom (mean±std across
    datasets). Bold + colored = best model for that row/metric. Per-dataset
    rows show plain point estimates (no std possible there — one number per
    dataset/model). Models are shown in a fixed order (OCTCube, then CLOSER).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./__outputs_feature_analysis/retrieval.csv")
    args = parser.parse_args(remaining_args)
    df = pd.read_csv(args.csv_path)
    metrics = ['top_1_acc', 'top_5_acc', 'top_10_acc', 'mAP']
    metric_labels = {'top_1_acc': 'Top-1', 'top_5_acc': 'Top-5',
                      'top_10_acc': 'Top-10', 'mAP': 'mAP'}
    model_display = {
        'octcube3d_60_linear_w_cls_patch': 'OCTCube',
        'miragefm3d_xattn_xattn_linear_w_2avg': 'CLOSER',
    }
    def get_label(m):
        return model_display.get(m, m)
    prescribed_order = [
        'octcube3d_60_linear_w_cls_patch',
        'miragefm3d_xattn_xattn_linear_w_2avg',
    ]
    prescribed_order = [m for m in prescribed_order if m in df['model_name'].unique()]
    prescribed_order += [m for m in df['model_name'].unique() if m not in prescribed_order]
    # Hardcoded colors matching the rest of the paper's palette
    model_colors = {
        'octcube3d_60_linear_w_cls_patch': 'colOCTCube',
        'miragefm3d_xattn_xattn_linear_w_2avg': 'colCLOSER',
    }
    color_defs = [
        '\\definecolor{colOCTCube}{rgb}{0.173,0.482,0.263}',   # dark green
        '\\definecolor{colCLOSER}{rgb}{0.470,0.330,0.600}',    # bright purple
    ]
    datasets = sorted(df['dataset_name'].unique())
    avg_stats = df.groupby('model_name', observed=True)[metrics].agg(['mean', 'std'])
    col_header = ' & '.join([f'\\textbf{{{metric_labels[m]}}}' for m in metrics])
    n_cols = len(metrics)
    lines = list(color_defs)
    lines.append('\\begin{table}[h]')
    lines.append('\\centering')
    lines.append(
        '\\caption{Per-dataset retrieval performance. Best result per dataset '
        'and metric in \\textbf{bold}. Average row reports mean±std across datasets.}'
    )
    lines.append('\\label{tab:retrieval_per_dataset}')
    lines.append(f'\\begin{{tabular}}{{ll{"c" * n_cols}}}')
    lines.append('\\toprule')
    lines.append(f'\\textbf{{Dataset}} & \\textbf{{Model}} & {col_header} \\\\')
    lines.append('\\midrule')
    for dataset in datasets:
        dataset_df = df[df['dataset_name'] == dataset]
        best_model_per_metric = {
            m: dataset_df.loc[dataset_df[m].idxmax(), 'model_name'] for m in metrics
        }
        for i, model in enumerate(prescribed_order):
            row = dataset_df[dataset_df['model_name'] == model]
            if row.empty:
                continue
            row = row.iloc[0]
            color_name = model_colors.get(model)
            label = get_label(model)
            colored_label = f'\\textcolor{{{color_name}}}{{{label}}}' if color_name else label
            cells = [dataset.replace('_', ' ') if i == 0 else '', colored_label]
            for m in metrics:
                val = row[m]
                cell_str = f'{val:.2f}'
                if best_model_per_metric[m] == model:
                    cell_str = f'\\textbf{{{cell_str}}}'
                    if color_name:
                        cell_str = f'\\textcolor{{{color_name}}}{{{cell_str}}}'
                cells.append(cell_str)
            lines.append(' & '.join(cells) + ' \\\\')
        lines.append('\\midrule')
    best_model_avg = {m: avg_stats[(m, 'mean')].idxmax() for m in metrics}
    for i, model in enumerate(prescribed_order):
        if model not in avg_stats.index:
            continue
        color_name = model_colors.get(model)
        label = get_label(model)
        colored_label = f'\\textcolor{{{color_name}}}{{{label}}}' if color_name else label
        cells = ['\\textit{Average}' if i == 0 else '', colored_label]
        for m in metrics:
            mean_val = avg_stats.loc[model, (m, 'mean')]
            std_val = avg_stats.loc[model, (m, 'std')]
            cell_str = f'{mean_val:.2f}$\\pm${std_val:.2f}'
            if best_model_avg[m] == model:
                cell_str = f'\\textbf{{{cell_str}}}'
                if color_name:
                    cell_str = f'\\textcolor{{{color_name}}}{{{cell_str}}}'
            cells.append(cell_str)
        lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    latex_str = '\n'.join(lines)
    print(latex_str)
    save_fn = Path("/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/complete_results/retrieval/retrieval_simple_table.tex")
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    with open(save_fn, 'w') as f:
        f.write(latex_str)
    print(f'\nSaved to {save_fn}')


def load_features_and_targets(root, seed):
    """Shared loading logic used by run_tsne, run_retrieval, and
    analyze_class_similarity."""
    misc.fix_seeds(seed)
    dataset_name = Path(root).parent.stem
    model_name = Path(root).stem
    features_fn = Path(root) / "cached_features.npz"
    assert features_fn.exists(), features_fn
    features = np.load(features_fn)
    features_train = torch.tensor(features["features_train"])[:, 0, 0]
    targets_train = torch.tensor(features["targets_train"])[:, 0]
    features_val = torch.tensor(features["features_val"])[:, 0, 0]
    targets_val = torch.tensor(features["targets_val"])[:, 0]
    features_test = torch.tensor(features["features_test"])[:, 0, 0]
    targets_test = torch.tensor(features["targets_test"])[:, 0]
    num_classes = features["num_classes"].item()
    return {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'features_train': features_train,
        'targets_train': targets_train,
        'features_val': features_val,
        'targets_val': targets_val,
        'features_test': features_test,
        'targets_test': targets_test,
        'num_classes': num_classes,
    }


def get_class_names(datasets_root, dataset_name):
    dataset_dir = Path(datasets_root) / dataset_name
    classes_list = sorted((dataset_dir / "train").glob("*/"))
    return [c.stem for c in classes_list], dataset_dir


def get_samples_ordered(dataset_dir, subset):
    samples = []
    root = dataset_dir / subset
    for cls in sorted(root.iterdir()):
        if not cls.is_dir():
            continue
        for vol in sorted(cls.iterdir()):
            if vol.suffix == '.npz':
                samples.append(vol.stem)
    return samples
model_names = {
    "octcube3d_60_linear_w_cls_patch": "OCTCube",
    "miragefm3d_xattn_xattn_linear_w_2avg": "CLOSER",
}


def run_tsne(remaining_args):
    """
    Loads cached features and produces t-SNE visualizations (by class,
    and by vendor if metadata is available). Split out from the
    combined get_results function for clarity.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True)
    parser.add_argument("--datasets_root", type=str, default="./_datasets/Classification_3D/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30)
    args = parser.parse_args(remaining_args)
    data = load_features_and_targets(args.root, args.seed)
    dataset_name, model_name = data['dataset_name'], data['model_name']
    class_names, dataset_dir = get_class_names(args.datasets_root, dataset_name)
    output_dir = Path("/home/morano/SW/Documents/My_Papers/MIRAGEv2/results/retrieval")
    output_dir = Path("/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/complete_results/retrieval/")
    # output_dir.mkdir(parents=True, exist_ok=True)
    all_features = torch.cat([data['features_train'], data['features_val'], data['features_test']], dim=0)
    all_targets = torch.cat([data['targets_train'], data['targets_val'], data['targets_test']], dim=0)
    print(f"Running t-SNE on {len(all_features)} samples...")
    embeddings = TSNE(
        n_components=2, perplexity=args.perplexity, random_state=args.seed, n_jobs=-1,
    ).fit_transform(all_features.numpy())
    # --- t-SNE colored by class ---
    model_name_disp = model_names.get(model_name, model_name)
    if model_name_disp == "OCTCube":
        figsize = (4, 4)
    else:
        figsize = (5, 4)
    fig, ax = plt.subplots(figsize=figsize)
    for class_idx in range(data['num_classes']):
        mask = all_targets.numpy() == class_idx
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], label=class_names[class_idx], s=10, alpha=0.6)
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    # Disable axis numbers
    ax.set_xticks([])
    ax.set_yticks([])
    char = "b"
    if model_name_disp == "OCTCube":
        char = "a"
    ax.text(
        -0.05, 1.05, char,
        transform=ax.transAxes,
        fontsize=16,
        fontweight='bold',
        va='top',
        ha='right',
    )
    # If OCTCube, do not show legend
    if model_name_disp == "OCTCube":
        ax.legend().set_visible(False)
    ax.set_title(f"t-SNE – {model_name_disp}")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    ax.set_facecolor('white')
    fig.savefig(output_dir / f"{dataset_name}_{model_name_disp}_tsne.svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- t-SNE colored by vendor, if metadata available ---
    metadata_fn = dataset_dir / 'metadata.json'
    if metadata_fn.exists():
        sample_ids_train = get_samples_ordered(dataset_dir, "train")
        sample_ids_val = get_samples_ordered(dataset_dir, "val")
        sample_ids_test = get_samples_ordered(dataset_dir, "test")
        all_ids = sample_ids_train + sample_ids_val + sample_ids_test
        with open(metadata_fn, 'r') as f:
            metadata = json.load(f)
        vendors = np.array([metadata[sid]["Vendor"] if sid in metadata else "Unknown" for sid in all_ids])
        unique_vendors = sorted(set(vendors))
        fig, ax = plt.subplots(figsize=(12, 10))
        for vendor in unique_vendors:
            mask = vendors == vendor
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], label=vendor, s=10, alpha=0.6)
        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.set_title(f"{model_name} – t-SNE by vendor (train + val + test)")
        fig.tight_layout()
        fig.patch.set_alpha(0)
        ax.set_facecolor('white')
        fig.savefig(output_dir / f"{dataset_name}_{model_name}_tsne_vendor.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Vendors found: {unique_vendors}")
    print(f"Saved t-SNE plots to {output_dir}")


def run_tsne_intelli(remaining_args):
    """
    Loads cached features and produces t-SNE visualizations (by class,
    and by vendor if metadata is available). Split out from the
    combined get_results function for clarity.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True)
    parser.add_argument("--datasets_root", type=str, default="./_datasets/Classification_3D/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30)
    args = parser.parse_args(remaining_args)
    intelli_classes = {
        "Healthy": "Control",
        "Normal": "Control",
        "Multiple_Sclerosis": "MS",
        "NORMAL": "Control",
        "control": "Control",
        "early": "Early glaucoma",
        "mid_advanced": "Int./Adv. glaucoma",
        "non": "Control",
        "DryAMD": "Dry AMD",
        "WetAMD": "Wet AMD",
        "NonAMD": "Control",
        "MacularHole_123": "MH (1-3)",
        "MacularHole_4": "MH (4)",
        "ME_DR": "ME and DR",
        "glaucoma": "Glaucoma",
        "cnv1": "CNV-1",
        "cnv2": "CNV-2",
        "cnv3": "CNV-3",
        "dme": "DME",
        "ga": "GA",
        "healthy": "Control",
        "iamd": "iAMD",
        "rvo": "RVO",
        "stargardt": "Stargardt",
        "DRU": "Drusen",
    }
    data = load_features_and_targets(args.root, args.seed)
    dataset_name, model_name = data['dataset_name'], data['model_name']
    class_names, dataset_dir = get_class_names(args.datasets_root, dataset_name)
    output_dir = Path("/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/complete_results/retrieval/")
    output_dir = Path("/home/morano/SW/Documents/My_Papers/MIRAGEv2/Submission_version/results/retrieval/")
    output_dir = Path("/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/__outputs_feature_analysis/tsnes")
    output_dir.mkdir(exist_ok=True)
    # output_dir.mkdir(parents=True, exist_ok=True)
    all_features = torch.cat([data['features_train'], data['features_val'], data['features_test']], dim=0)
    all_targets = torch.cat([data['targets_train'], data['targets_val'], data['targets_test']], dim=0)
    print(f"Running t-SNE on {len(all_features)} samples...")
    embeddings = TSNE(
        n_components=2, perplexity=args.perplexity, random_state=args.seed, n_jobs=-1,
    ).fit_transform(all_features.numpy())
    # --- t-SNE colored by class ---
    model_name_disp = model_names.get(model_name, model_name)
    model_name_disp = model_name_disp.replace('CLOSER', 'FARO')
    model_name_save = model_name_disp
    if "FARO" in model_name_disp:
         model_name_save = 'OURS'
    if '9C' in dataset_name:
        bullet_size = 2
    else:
        bullet_size = 10
    figsize = (4, 4)
    fig, ax = plt.subplots(figsize=figsize)
    for class_idx in range(data['num_classes']):
        mask = all_targets.numpy() == class_idx
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], label=intelli_classes.get(class_names[class_idx], class_names[class_idx]), s=bullet_size, alpha=0.6)
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    # Disable axis numbers
    ax.set_xticks([])
    ax.set_yticks([])
    # If OCTCube, do not show legend
    if model_name_disp == "OCTCube":
        ax.legend().set_visible(False)
    # ax.set_title(f"t-SNE – {model_name_disp}")
    ax.set_title(f"{model_name_disp}")
    # fig.tight_layout()
    fig.patch.set_alpha(0)
    ax.set_facecolor('white')
    fig.savefig(output_dir / f"{dataset_name}_{model_name_save}_tsne.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- t-SNE colored by vendor, if metadata available ---
    metadata_fn = dataset_dir / 'metadata.json'
    if metadata_fn.exists():
        sample_ids_train = get_samples_ordered(dataset_dir, "train")
        sample_ids_val = get_samples_ordered(dataset_dir, "val")
        sample_ids_test = get_samples_ordered(dataset_dir, "test")
        all_ids = sample_ids_train + sample_ids_val + sample_ids_test
        with open(metadata_fn, 'r') as f:
            metadata = json.load(f)
        vendors = np.array([metadata[sid]["Vendor"] if sid in metadata else "Unknown" for sid in all_ids])
        unique_vendors = sorted(set(vendors))
        fig, ax = plt.subplots(figsize=(12, 10))
        for vendor in unique_vendors:
            mask = vendors == vendor
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], label=vendor, s=10, alpha=0.6)
        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.set_title(f"{model_name} – t-SNE by vendor (train + val + test)")
        fig.tight_layout()
        fig.patch.set_alpha(0)
        ax.set_facecolor('white')
        fig.savefig(output_dir / f"{dataset_name}_{model_name_save}_tsne_vendor.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Vendors found: {unique_vendors}")
    print(f"Saved t-SNE plots to {output_dir}")


def run_retrieval(remaining_args):
    """
    Loads cached features and computes retrieval metrics (top-k
    accuracy, mAP) plus qualitative retrieval examples. Split out from
    the combined get_results function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True)
    parser.add_argument("--datasets_root", type=str, default="./_datasets/Classification_3D/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results.')
    parser.add_argument("--n_queries", type=int, default=5)
    parser.add_argument("--n_neighbors", type=int, default=5)
    args = parser.parse_args(remaining_args)
    data = load_features_and_targets(args.root, args.seed)
    dataset_name, model_name = data['dataset_name'], data['model_name']
    class_names, dataset_dir = get_class_names(args.datasets_root, dataset_name)
    output_dir = Path("./__outputs_feature_analysis_new")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids_train = get_samples_ordered(dataset_dir, "train")
    sample_ids_val = get_samples_ordered(dataset_dir, "val")
    sample_ids_test = get_samples_ordered(dataset_dir, "test")
    def l2_normalize(x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    train_feats = l2_normalize(data['features_train']).numpy()
    train_labels = data['targets_train'].numpy()
    query_feats = l2_normalize(torch.cat([data['features_val'], data['features_test']], dim=0)).numpy()
    query_labels = torch.cat([data['targets_val'], data['targets_test']], dim=0).numpy()
    print(f"Computing cosine similarity ({len(query_feats)} queries x {len(train_feats)} gallery)...")
    sim_matrix = cosine_similarity(query_feats, train_feats)
    K = [1, 5, 10]
    results = {}
    for k in K:
        top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
        top_k_labels = train_labels[top_k_indices]
        correct = (top_k_labels == query_labels[:, None]).any(axis=1)
        acc = correct.mean() * 100
        results[k] = acc
        print(f"  Top-{k} accuracy: {acc:.2f}%")
    def compute_map(sim_matrix, query_labels, train_labels):
        average_precisions = []
        for i in range(sim_matrix.shape[0]):
            query_label = query_labels[i]
            sorted_indices = np.argsort(sim_matrix[i])[::-1]
            sorted_labels = train_labels[sorted_indices]
            relevant = (sorted_labels == query_label).astype(int)
            cumulative_relevant = np.cumsum(relevant)
            precision_at_k = cumulative_relevant / (np.arange(len(relevant)) + 1)
            if relevant.sum() > 0:
                average_precisions.append((precision_at_k * relevant).sum() / relevant.sum())
        return np.mean(average_precisions) * 100 if average_precisions else 0.0
    map_score = compute_map(sim_matrix, query_labels, train_labels)
    results_fn = output_dir / "retrieval.csv"
    df = pd.DataFrame([{
        "model_name": model_name,
        "dataset_name": dataset_name,
        "top_1_acc": results[1],
        "top_5_acc": results[5],
        "top_10_acc": results[10],
        "mAP": map_score,
    }])
    if results_fn.exists() and not args.overwrite:
        df_existing = pd.read_csv(results_fn)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(results_fn, index=False)
    print(f"Saved to {results_fn}")
    # --- Qualitative retrieval examples ---
    rng = np.random.default_rng(args.seed)
    def build_id_map(subset):
        id_map = {}
        for class_dir in sorted(dataset_dir.glob('train/*/')):
            target = dataset_dir / subset / class_dir.stem
            if not target.is_dir():
                continue
            for p in target.rglob("*"):
                if p.is_file() and p.suffix.lower() == ".npz":
                    id_map[p.stem] = (class_dir.stem, p)
        return id_map
    id_map_train = build_id_map("train")
    id_map_query = build_id_map("val")
    id_map_query.update(build_id_map("test"))
    def load_central_bscan(path):
        vol = np.load(path)["vol"]
        return vol[vol.shape[0] // 2]
    query_indices = rng.choice(len(query_feats), size=args.n_queries, replace=False)
    fig, axes = plt.subplots(
        args.n_queries, 1 + args.n_neighbors,
        figsize=(3 * (1 + args.n_neighbors), 3 * args.n_queries),
    )
    all_query_ids = sample_ids_val + sample_ids_test
    for row, q_idx in enumerate(query_indices):
        q_id = all_query_ids[q_idx]
        q_class, q_path = id_map_query[q_id]
        top_k_idx = np.argsort(sim_matrix[q_idx])[::-1][:args.n_neighbors]
        ax = axes[row, 0]
        ax.imshow(load_central_bscan(q_path), cmap="gray")
        ax.set_title(f"Query\n{q_class}", fontsize=7, color="steelblue")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("steelblue")
            spine.set_linewidth(3)
            spine.set_visible(True)
        for col, n_idx in enumerate(top_k_idx):
            n_id = sample_ids_train[n_idx]
            n_class, n_path = id_map_train[n_id]
            score = sim_matrix[q_idx, n_idx]
            correct = n_class == q_class
            ax = axes[row, col + 1]
            ax.imshow(load_central_bscan(n_path), cmap="gray")
            ax.set_title(f"#{col+1} ({score:.2f})\n{n_class}", fontsize=7, color="green" if correct else "red")
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor("green" if correct else "red")
                spine.set_linewidth(3)
                spine.set_visible(True)
    fig.suptitle(f"{model_name} – Retrieval examples", fontsize=12)
    fig.tight_layout()
    fig.patch.set_alpha(0)
    for a in axes.flat:
        a.set_facecolor('white')
    fig.savefig(output_dir / f"{dataset_name}_{model_name}_retrieval_qualitative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved qualitative retrieval plot.")


# Module-level friendly class-name mapping, reused across t-SNE and
# dendrogram plots so labels are consistent everywhere.
INTELLI_CLASS_NAMES = {
    "Healthy": "Control", "Normal": "Control", "Multiple_Sclerosis": "MS",
    "NORMAL": "Control", "control": "Control", "early": "Early glaucoma",
    "mid_advanced": "Int./Adv. glaucoma", "non": "Control",
    "DryAMD": "Dry AMD", "WetAMD": "Wet AMD", "NonAMD": "Control",
    "MacularHole_123": "MH (1-3)", "MacularHole_4": "MH (4)",
    "ME_DR": "ME and DR", "glaucoma": "Glaucoma",
    "cnv1": "CNV-1",
    "cnv2": "CNV-2",
    "cnv3": "CNV-3",
    "dme": "DME",
    "ga": "GA",
    "healthy": "Control",
    "iamd": "iAMD",
    "rvo": "RVO",
    "stargardt": "Stargardt",
    "DRU": "Drusen",
}


def analyze_class_similarity(remaining_args):
    """
    Computes pairwise cosine similarity between class centroids in the
    raw (non-reduced) feature space, plus a hierarchical clustering
    dendrogram — a t-SNE-independent, quantitative way to check whether
    classes are organized in a clinically meaningful order.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True)
    parser.add_argument("--datasets_root", type=str, default="./_datasets/Classification_3D/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold_frac", type=float, default=0.7,
                         help="Fraction of the max linkage height used as "
                              "the color_threshold (scipy default is 0.7). "
                              "Lower = more/finer colored clusters.")
    args = parser.parse_args(remaining_args)

    data = load_features_and_targets(args.root, args.seed)
    dataset_name, model_name = data['dataset_name'], data['model_name']
    model_name_disp = model_names.get(model_name, model_name)
    model_name_disp = model_name_disp.replace('CLOSER', 'FARO')
    model_name_save = model_name_disp
    if "FARO" in model_name_disp:
         model_name_save = 'OURS'
    class_names, _ = get_class_names(args.datasets_root, dataset_name)
    display_names = [INTELLI_CLASS_NAMES.get(c, c) for c in class_names]

    output_dir = Path("./__outputs_feature_analysis", "dendrograms")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = torch.cat([data['features_train'], data['features_val'], data['features_test']], dim=0).numpy()
    all_targets = torch.cat([data['targets_train'], data['targets_val'], data['targets_test']], dim=0).numpy()

    centroids = []
    for c in range(data['num_classes']):
        mask = all_targets == c
        centroids.append(all_features[mask].mean(axis=0))
    centroids = np.stack(centroids)

    sim_matrix = cosine_similarity(centroids)
    sim_df = pd.DataFrame(sim_matrix, index=class_names, columns=class_names)
    print(f"\n{model_name_disp} — Class centroid cosine similarity ({dataset_name}):")
    print(sim_df.round(3))

    csv_fn = output_dir / f"{dataset_name}_{model_name_save}_class_similarity.csv"
    sim_df.to_csv(csv_fn)
    print(f"Saved similarity matrix to {csv_fn}")

    # --- Dendrogram ---
    dist_matrix = 1 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)  # avoid tiny negative values from float error
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='average')

    FONT_SIZE = 14
    plt.rcParams.update({"font.size": FONT_SIZE})

    fig, ax = plt.subplots(figsize=(7, 5))
    color_threshold = args.threshold_frac * Z[:, 2].max()
    ddata = dendrogram(
        Z,
        labels=display_names,
        ax=ax,
        color_threshold=color_threshold,
        leaf_rotation=45,
        leaf_font_size=FONT_SIZE,
    )
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")

    # Color each leaf label to match its own link color, as computed by
    # scipy's default (distance-based) coloring -- no fixed k, so the
    # number of distinct colors reflects how well-separated the classes
    # actually are in feature space.
    leaves_color_list = ddata['leaves_color_list']  # one color per leaf, in plotted order
    for tick_label, color in zip(ax.get_xticklabels(), leaves_color_list):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    ax.set_title(model_name_disp, fontsize=FONT_SIZE + 3)
    ax.set_ylabel("Cosine distance", fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    fig.tight_layout()
    fig.patch.set_alpha(0)
    ax.set_facecolor('white')

    fig_fn = output_dir / f"{dataset_name}_{model_name_save}_class_dendrogram.pdf"
    fig_fn.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_fn, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dendrogram to {fig_fn}")


def run_retrieval_paper_examples(remaining_args):
    """
    Loads cached features, computes retrieval, and saves a clean,
    publication-ready qualitative figure with only 3 query examples
    (each with its top-5 neighbors) per dataset/model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True)
    parser.add_argument("--datasets_root", type=str, default="./_datasets/Classification_3D/")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--n_queries", type=int, default=3)
    parser.add_argument("--n_neighbors", type=int, default=5)
    args = parser.parse_args(remaining_args)
    data = load_features_and_targets(args.root, args.seed)
    dataset_name, model_name = data['dataset_name'], data['model_name']
    model_name_disp = model_names.get(model_name, model_name)
    class_names, dataset_dir = get_class_names(args.datasets_root, dataset_name)
    output_dir = Path("/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/complete_results/retrieval/")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids_train = get_samples_ordered(dataset_dir, "train")
    sample_ids_val = get_samples_ordered(dataset_dir, "val")
    sample_ids_test = get_samples_ordered(dataset_dir, "test")
    def l2_normalize(x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    train_feats = l2_normalize(data['features_train']).numpy()
    query_feats = l2_normalize(torch.cat([data['features_val'], data['features_test']], dim=0)).numpy()
    sim_matrix = cosine_similarity(query_feats, train_feats)
    def build_id_map(subset):
        id_map = {}
        for class_dir in sorted(dataset_dir.glob('train/*/')):
            target = dataset_dir / subset / class_dir.stem
            if not target.is_dir():
                continue
            for p in target.rglob("*"):
                if p.is_file() and p.suffix.lower() == ".npz":
                    id_map[p.stem] = (class_dir.stem, p)
        return id_map
    id_map_train = build_id_map("train")
    id_map_query = build_id_map("val")
    id_map_query.update(build_id_map("test"))
    def load_central_bscan(path):
        vol = np.load(path)["vol"]
        return vol[vol.shape[0] // 2]
    rng = np.random.default_rng(args.seed)
    all_query_ids = sample_ids_val + sample_ids_test
    query_indices = rng.choice(len(query_feats), size=args.n_queries, replace=False)
    fig, axes = plt.subplots(
        args.n_queries, 1 + args.n_neighbors,
        figsize=(2.2 * (1 + args.n_neighbors), 2.2 * args.n_queries),
    )
    if args.n_queries == 1:
        axes = axes[None, :]
    for row, q_idx in enumerate(query_indices):
        q_id = all_query_ids[q_idx]
        q_class, q_path = id_map_query[q_id]
        q_label = intelli_classes_default(q_class)
        top_k_idx = np.argsort(sim_matrix[q_idx])[::-1][:args.n_neighbors]
        ax = axes[row, 0]
        ax.imshow(load_central_bscan(q_path), cmap="gray")
        ax.set_title(f"Query\n{q_label}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("steelblue")
            spine.set_linewidth(3)
        for col, n_idx in enumerate(top_k_idx):
            n_id = sample_ids_train[n_idx]
            n_class, n_path = id_map_train[n_id]
            n_label = intelli_classes_default(n_class)
            correct = n_class == q_class
            ax = axes[row, col + 1]
            ax.imshow(load_central_bscan(n_path), cmap="gray")
            ax.set_title(f"#{col + 1}\n{n_label}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("green" if correct else "red")
                spine.set_linewidth(3)
    fig.tight_layout()
    fig.patch.set_alpha(0)
    for a in axes.flat:
        a.set_facecolor('white')
    save_fn = output_dir / f"{dataset_name}_{model_name_disp}_retrieval_paper.svg"
    fig.savefig(save_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paper-ready retrieval figure to {save_fn}")


def intelli_classes_default(class_name):
    """Same friendly-name mapping used in run_tsne_intelli, reused here
    for consistent class labels in the paper figure."""
    intelli_classes = {
        "Healthy": "Control", "Normal": "Control", "Multiple_Sclerosis": "MS",
        "NORMAL": "Control", "control": "Control", "early": "Early glaucoma",
        "mid_advanced": "Int./Adv. glaucoma", "non": "Control",
        "DryAMD": "Dry AMD", "WetAMD": "Wet AMD", "NonAMD": "Control",
        "MacularHole_123": "MH (1-3)", "MacularHole_4": "MH (4)",
        "ME_DR": "ME and DR", "glaucoma": "Glaucoma",
    }
    return intelli_classes.get(class_name, class_name)


def analyze_disease_trajectory(remaining_args):
    """
    Projects all samples onto a 1D "disease axis" defined by two endpoint
    classes (e.g., Normal -> GA), and checks whether intermediate classes
    (e.g., Drusen) fall in between in a way that is consistent with a
    clinically expected severity ordering.

    This directly tests trajectory/ordering structure in the feature
    space, which class-centroid similarity cannot: similarity only tells
    you how close two class means are, not whether a third class sits
    "in between" them along a meaningful axis.

    Usage example:
        python feature_analysis.py analyze_disease_trajectory \
            -r <root> \
            --order NORMAL,DRU,GA \
            --context_classes CNV,DME,ERM
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True)
    parser.add_argument("--datasets_root", type=str, default="./_datasets/Classification_3D/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--order", type=str, required=True,
        help="Comma-separated class names in expected biological/severity "
             "order, e.g. 'NORMAL,DRU,GA'. The first and last entries "
             "define the projection axis; all listed classes are checked "
             "for monotonic ordering along it."
    )
    parser.add_argument(
        "--context_classes", type=str, default="",
        help="Comma-separated class names to also project and plot (in "
             "grey, for visual context) but that are NOT used to define "
             "the axis and are NOT included in the monotonicity check."
    )
    args = parser.parse_args(remaining_args)

    data = load_features_and_targets(args.root, args.seed)
    dataset_name, model_name = data['dataset_name'], data['model_name']
    model_name_disp = model_names.get(model_name, model_name)
    class_names, _ = get_class_names(args.datasets_root, dataset_name)

    output_dir = Path("./__outputs_feature_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    order = [c.strip() for c in args.order.split(",") if c.strip()]
    context_classes = [c.strip() for c in args.context_classes.split(",") if c.strip()]
    assert len(order) >= 2, "--order needs at least a start and an end class."
    for c in order + context_classes:
        assert c in class_names, f"Class '{c}' not found in dataset classes: {class_names}"

    all_features = torch.cat(
        [data['features_train'], data['features_val'], data['features_test']], dim=0
    ).numpy()
    all_targets = torch.cat(
        [data['targets_train'], data['targets_val'], data['targets_test']], dim=0
    ).numpy()
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    # --- Define the axis from the two endpoint classes ---
    start_name, end_name = order[0], order[-1]
    start_mask = all_targets == name_to_idx[start_name]
    end_mask = all_targets == name_to_idx[end_name]
    start_centroid = all_features[start_mask].mean(axis=0)
    end_centroid = all_features[end_mask].mean(axis=0)

    axis_vec = end_centroid - start_centroid
    axis_len = np.linalg.norm(axis_vec)
    axis_unit = axis_vec / axis_len

    # --- Project every sample onto the axis, normalized so start=0, end=1 ---
    def project(feats):
        return (feats - start_centroid) @ axis_unit / axis_len

    plot_classes = order + [c for c in context_classes if c not in order]
    rows = []
    for c in plot_classes:
        mask = all_targets == name_to_idx[c]
        proj = project(all_features[mask])
        for v in proj:
            rows.append({"class": c, "projection": v, "dataset_name": dataset_name,
                         "model_name": model_name})
    proj_df = pd.DataFrame(rows)

    csv_fn = output_dir / f"{dataset_name}_{model_name}_disease_trajectory.csv"
    proj_df.to_csv(csv_fn, index=False)
    print(f"Saved per-sample projections to {csv_fn}")

    # --- Monotonicity check (Spearman correlation, axis classes only) ---
    axis_only_df = proj_df[proj_df["class"].isin(order)].copy()
    rank_map = {c: i for i, c in enumerate(order)}
    axis_only_df["expected_rank"] = axis_only_df["class"].map(rank_map)
    rho, pval = spearmanr(axis_only_df["expected_rank"], axis_only_df["projection"])

    print(f"\n{model_name_disp} — Disease trajectory ({dataset_name}): "
          f"{' -> '.join(order)}")
    means = proj_df.groupby("class")["projection"].mean().reindex(plot_classes)
    medians = proj_df.groupby("class")["projection"].median().reindex(plot_classes)
    summary_df = pd.DataFrame({"mean": means, "median": medians})
    print(summary_df.round(3))
    print(f"Spearman correlation (expected order vs. projection): "
          f"rho={rho:.3f}, p={pval:.2e}")
    is_monotonic = list(means[order].values) == sorted(means[order].values)
    print(f"Mean projection strictly monotonic along expected order: {is_monotonic}")

    summary_fn = output_dir / f"{dataset_name}_{model_name}_disease_trajectory_summary.csv"
    summary_df.to_csv(summary_fn)
    with open(output_dir / f"{dataset_name}_{model_name}_disease_trajectory_stats.json", "w") as f:
        json.dump({"spearman_rho": rho, "spearman_p": pval,
                   "monotonic": bool(is_monotonic), "order": order}, f, indent=2)

    # --- Plot: one row per class, jittered strip + mean marker ---
    fig, ax = plt.subplots(figsize=(6, 0.9 * len(plot_classes) + 1))
    rng = np.random.default_rng(args.seed)
    y_positions = {c: i for i, c in enumerate(plot_classes)}
    for c in plot_classes:
        vals = proj_df.loc[proj_df["class"] == c, "projection"].values
        y = y_positions[c] + rng.uniform(-0.15, 0.15, size=len(vals))
        color = "0.6" if c in context_classes and c not in order else None
        ax.scatter(vals, y, s=12, alpha=0.5, color=color,
                   label=c if color is None else None)
        ax.scatter(vals.mean(), y_positions[c], marker="D", s=60,
                   color="black", zorder=5)

    ax.axvline(0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(1, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlabel(f"Projection onto {start_name} \u2192 {end_name} axis")
    ax.set_title(f"{model_name_disp} \u2014 disease trajectory ({dataset_name})")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    ax.set_facecolor('white')
    fig_fn = output_dir / f"{dataset_name}_{model_name}_disease_trajectory.png"
    fig.savefig(fig_fn, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory plot to {fig_fn}")


def plot_disease_trajectory_comparison(remaining_args):
    """
    Combines per-sample projection CSVs (produced by
    analyze_disease_trajectory) from two models into a single
    publication-ready comparison figure, colored consistently with the
    rest of the paper (colCLOSER / colOCTCube).

    Usage example:
        python feature_analysis.py plot_disease_trajectory_comparison \
            --csv_a __outputs_feature_analysis/OCTAVE_octcube3d_60_linear_w_cls_patch_disease_trajectory.csv \
            --csv_b __outputs_feature_analysis/OCTAVE_miragefm3d_xattn_xattn_linear_w_2avg_disease_trajectory.csv \
            --order NORMAL,DRU,GA
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_a", type=str, required=True, help="e.g. OCTCube projections CSV")
    parser.add_argument("--csv_b", type=str, required=True, help="e.g. CLOSER projections CSV")
    parser.add_argument("--order", type=str, required=True)
    parser.add_argument("--context_classes", type=str, default="")
    args = parser.parse_args(remaining_args)

    order = [c.strip() for c in args.order.split(",") if c.strip()]
    context_classes = [c.strip() for c in args.context_classes.split(",") if c.strip()]
    plot_classes = order + [c for c in context_classes if c not in order]

    model_colors = {
        'octcube3d_60_linear_w_cls_patch': '#1B5E3A',   # dark green, colOCTCube
        'miragefm3d_xattn_xattn_linear_w_2avg': '#5A189A',  # dark purple, colCLOSER
    }

    output_dir = Path("./__outputs_feature_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_a = pd.read_csv(args.csv_a)
    df_b = pd.read_csv(args.csv_b)
    dataset_name = df_a["dataset_name"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 0.9 * len(plot_classes) + 1), sharex=True)
    rng = np.random.default_rng(0)

    for ax, df in zip(axes, [df_a, df_b]):
        model_name = df["model_name"].iloc[0]
        model_disp = model_names.get(model_name, model_name)
        color = model_colors.get(model_name, "#333333")
        y_positions = {c: i for i, c in enumerate(plot_classes)}
        for c in plot_classes:
            vals = df.loc[df["class"] == c, "projection"].values
            y = y_positions[c] + rng.uniform(-0.15, 0.15, size=len(vals))
            is_context = c in context_classes and c not in order
            pt_color = "0.7" if is_context else color
            ax.scatter(vals, y, s=12, alpha=0.5, color=pt_color)
            ax.scatter(vals.mean(), y_positions[c], marker="D", s=60,
                       color="black", zorder=5)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(1, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(y_positions.keys()))
        ax.set_xlabel(f"Projection onto {order[0]} \u2192 {order[-1]} axis")
        ax.set_title(model_disp, color=color, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.patch.set_alpha(0)
    for a in axes.flat:
        a.set_facecolor('white')
    fig_fn = output_dir / f"{dataset_name}_disease_trajectory_comparison.pdf"
    fig.savefig(fig_fn, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {fig_fn}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str)
    args, extras = parser.parse_known_args()
    globals()[args.function](extras)
