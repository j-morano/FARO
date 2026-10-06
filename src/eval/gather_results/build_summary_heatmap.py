"""
Summary version of the grouped heatmap: instead of one row per dataset,
one row per disease/task category (dataset_group), averaging the metric
across every dataset in that category. Reuses the same color palette,
panel layout, and cell-drawing logic as build_real_heatmap.py -- the only
new piece is the per-group averaging step (aggregate_by_group).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import colorsys
from pathlib import Path
from visualization.plot_utils import get_palette

# ============================================================
# Set this to the directory containing your raw result CSVs.
# ============================================================
BASE_PATH = Path('/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/raw_results')

all_models = ['RETFound', 'VisionFM', 'UrFound', 'OCTCube', 'MIRAGE', 'CLOSER']
vol_models = ['OCTCube (60 B-scans)', 'CLOSER (60 B-scans)', 'CLOSER (all B-scans)']
retr_models = ['OCTCube', 'CLOSER']

SIMPLIFY_VOLUME_DIAGNOSIS = True
if SIMPLIFY_VOLUME_DIAGNOSIS:
    vol_models = ['OCTCube (60 B-scans)', 'CLOSER (all B-scans)']


def soften_for_fill(hex_color, model_label, darken_factor=0.82,
                     yellow_darken_factor=0.9, closer_name='CLOSER'):
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    is_yellow = 0.11 <= h <= 0.19
    is_closer = closer_name.lower() in model_label.lower()
    if is_closer:
        pass
    elif is_yellow:
        v *= yellow_darken_factor
        s = min(1.0, s * 1.1)
    else:
        v *= darken_factor
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (r2, g2, b2)


_all_model_strings = list(dict.fromkeys(all_models + vol_models + retr_models))
_raw_model_colors = get_palette(_all_model_strings)
model_colors = {
    name: soften_for_fill(hexcol, name) for name, hexcol in _raw_model_colors.items()
}

# ============================================================
# Same category mapping as build_real_heatmap.py -- keep these two files
# in sync if you edit one (e.g. adding a new dataset or changing a
# category), since this is what determines which rows get averaged
# together here.
# ============================================================
dataset_group = {
    'AMD_DME_3D': 'AMD/DME', 'AMD_DME_3D_lesions': 'AMD/DME',
    'BanglaOCT2025': 'AMD', 'MARIO_T1': 'AMD',
    'MMC_AMD': 'AMD', 'Duke_iAMD': 'AMD', 'Duke_iAMD_labeled_nosclera_70_layers': 'AMD',
    'AMD_SD': 'AMD', 'AMD_SD_vol_layers': 'AMD', 'AMD_SD_vol_lesions': 'AMD',
    'AROI': 'AMD', 'AROI_nosclera_layers': 'AMD', 'AROI_nosclera_lesions': 'AMD',
    'UMNandDukeSrinivasan_to_Noor_cross': 'AMD/DME',
    'Noor_to_UMNandDukeSrinivasan_cross': 'AMD/DME',
    'UMN': 'AMD/DME', 'UMN_lesions': 'AMD/DME',
    'Noor_Eye_Hospital': 'AMD/DME', 'Kermany_correct': 'AMD/DME',
    'OAEFD_DME': 'DME', 'Duke_DME': 'DME',
    'Duke_DME_nosclera_layers': 'DME', 'Duke_DME_nosclera_lesions': 'DME',
    'OAEFD_DR': 'DR',
    'M3Ret': 'DR/DME', 'OLIVES': 'DR/DME', 'OLIVES_baseline': 'DR/DME',
    'GAMMA': 'Glaucoma', 'Harvard_Glaucoma': 'Glaucoma',
    'POAGD': 'Glaucoma', 'GOALS': 'Glaucoma', 'GOALS_layers': 'Glaucoma',
    'OCT_MD_MS': 'MS',
    'OIMHS': 'MH', 'OIMHS_layers': 'MH', 'OIMHS_lesions': 'MH',
    'RVO_ME': 'RVO', 'RVO_Lesion_lesions': 'RVO',
    'RETOUCH': 'AMD/DME/RVO', 'RETOUCH_lesions': 'AMD/DME/RVO',
    'OCTA500': 'Multi-disease', 'OCTID': 'Multi-disease', 'OCTDL': 'Multi-disease',
    'OCT_C8': 'Multi-disease', 'OCTAVE': 'Multi-disease',
    'OCTAVE_layers': 'Multi-disease', 'OCTAVE_lesions': 'Multi-disease',
}

group_display = {
    'AMD/DME': 'AMD, DME',
    'AMD': 'AMD',
    'DME': 'DME',
    'DR': 'DR',
    'DR/DME': 'DR, DME',
    'Glaucoma': 'Glaucoma',
    'MS': 'MS',
    'RVO': 'RVO',
    'AMD/DME/RVO': 'AMD, DME, RVO',
    'MH': 'MH',
    'Multi-disease': 'Multi-disease',
}

group_order = [
    'AMD/DME', 'AMD', 'DME', 'DR', 'DR/DME',
    'Glaucoma', 'MS', 'RVO', 'AMD/DME/RVO', 'MH', 'Multi-disease',
]


def _mathtext_bold(text):
    escaped = text.replace(' ', r'\ ').replace('-', r'\text{-}')
    return r'$\bf{' + escaped + '}$'


def aggregate_by_group(pivot_df):
    """
    Average a (Dataset x Model) pivot table's rows by disease/task
    category, producing a (Category x Model) pivot with one row per
    category -- the mean across every dataset in that category, per
    model. Also returns {category: n_datasets_averaged}, used to
    annotate each row with how many datasets contributed to it.
    Datasets missing from dataset_group are grouped under 'Other' rather
    than silently dropped.
    """
    df = pivot_df.copy()
    cats = [dataset_group.get(k, 'Other') for k in df.index]
    df = df.groupby(cats).mean(numeric_only=True)
    counts = pd.Series(cats).value_counts().to_dict()

    ordered = [c for c in group_order if c in df.index] + \
              [c for c in df.index if c not in group_order]
    return df.loc[ordered], counts


def build_summary_ylabels(categories, counts):
    """Bold category name, '(N datasets)' below -- singular/plural handled."""
    labels = []
    for cat in categories:
        n = counts.get(cat, 0)
        disp = group_display.get(cat, cat)
        n_label = f'{n} dataset' if n == 1 else f'{n} datasets'
        labels.append(f'{_mathtext_bold(disp)}\n({n_label})')
    return labels


# ============================================================
# Loaders -- identical to build_real_heatmap.py
# ============================================================

def load_seed_level(path, metric='MCC', dataset_col='Dataset', model_col='Model'):
    df = pd.read_csv(path)
    return df.groupby([dataset_col, model_col])[metric].mean().unstack(model_col)


def load_retrieval(path, metric='mAP'):
    df = pd.read_csv(path)
    return df.pivot(index='dataset_name', columns='model_name', values=metric)


def load_segmentation(path, metric='Dice'):
    df = pd.read_csv(path)
    return df.groupby(['Dataset', 'Model'])[metric].mean().unstack('Model')


def set_two_part_title(ax, name, metric, fontsize=9.5, metric_fontsize=7.5):
    t1 = ax.set_title(name, fontsize=fontsize, fontweight='bold')
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb1 = t1.get_window_extent(renderer=renderer)
    inv = ax.transAxes.inverted()
    end_x_axes, y_axes = inv.transform((bb1.x1, bb1.y0))
    ax.text(
        end_x_axes, y_axes, f' ({metric})', transform=ax.transAxes,
        fontsize=metric_fontsize, fontweight='normal', ha='left', va='bottom',
    )


def draw_panel(ax, pivot_df, title, models_list, metric, show_ylabels=True,
                xtick_labels=None, ylabels=None):
    pivot_df = pivot_df.reindex(columns=models_list).dropna(how='any')
    tasks = list(pivot_df.index)
    data = pivot_df.to_numpy()

    closer_col_names = [m for m in models_list if 'CLOSER' in m]
    best_idx = np.zeros(len(tasks), dtype=int)
    for i in range(len(tasks)):
        row = data[i]
        row_max = row.max()
        is_tied_for_best = np.isclose(row, row_max, atol=1e-9)
        tied_models = [j for j in range(len(models_list)) if is_tied_for_best[j]]
        closer_tied = [j for j in tied_models if models_list[j] in closer_col_names]
        best_idx[i] = closer_tied[0] if closer_tied else tied_models[0]

    img = np.ones((len(tasks), len(models_list), 4))
    for i in range(len(tasks)):
        row = data[i]
        winner = models_list[best_idx[i]]
        base_rgb = mcolors.to_rgb(model_colors[winner])
        row_min, row_max = row.min(), row.max()
        for j in range(len(models_list)):
            rel = (row[j] - row_min) / (row_max - row_min + 1e-9)
            alpha = 0.15 + 0.85 * rel
            img[i, j] = (*base_rgb, alpha)

    ax.imshow(img, aspect='auto')

    for i in range(len(tasks)):
        for j in range(len(models_list)):
            val = data[i, j]
            is_winner = (j == best_idx[i])
            if is_winner:
                ax.text(
                    j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7.2, color='white', fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=1.8, foreground='black')],
                )
            else:
                ax.text(
                    j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=5.6, color='black',
                )

    ax.set_xticks(range(len(models_list)))
    ax.set_xticklabels(xtick_labels or models_list, rotation=40, ha='right', fontsize=6.5)
    if show_ylabels:
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(ylabels if ylabels is not None else tasks, fontsize=7.0)
    else:
        ax.set_yticks([])
    set_two_part_title(ax, title, metric)
    ax.set_xticks(np.arange(-0.5, len(models_list), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0)
    ax.tick_params(which='minor', length=0)
    return len(tasks)


def _load_aggregate_label(loader_fn, *loader_args, **loader_kwargs):
    """Load a raw (Dataset x Model) pivot, then collapse it to one
    averaged row per disease/task category, returning the aggregated
    pivot and its matching y-labels together."""
    pivot = loader_fn(*loader_args, **loader_kwargs)
    grouped, counts = aggregate_by_group(pivot)
    ylabels = build_summary_ylabels(list(grouped.index), counts)
    return grouped, ylabels


def main():
    vol_pivot, vol_ylabels = _load_aggregate_label(
        load_seed_level, BASE_PATH / '3d_raw_results.csv', metric='MCC')
    retr_pivot, retr_ylabels = _load_aggregate_label(
        load_retrieval, BASE_PATH / 'retrieval_raw_results.csv', metric='mAP')
    # Divide retrieval results by 100 to match the scale of the other metrics (0-1)
    retr_pivot /= 100.0
    cd_pivot, cd_ylabels = _load_aggregate_label(
        load_seed_level, BASE_PATH / 'chd_raw_results.csv', metric='MCC')
    bscan_pivot, bscan_ylabels = _load_aggregate_label(
        load_seed_level, BASE_PATH / '2d_raw_results.csv', metric='MCC')
    seg_pivot, seg_ylabels = _load_aggregate_label(
        load_segmentation, BASE_PATH / 'sota_raw_seg_results.csv', metric='Dice')

    full_models = all_models

    # Much shorter figure than the per-dataset version, since each panel
    # now has only a handful of category rows instead of dozens of
    # dataset rows.
    fig = plt.figure(figsize=(11, 7))
    gs_outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    n_vol = len(vol_pivot.reindex(columns=vol_models).dropna(how='any'))
    n_retr = len(retr_pivot.reindex(columns=retr_models).dropna(how='any'))
    n_cd = len(cd_pivot.reindex(columns=full_models).dropna(how='any'))
    n_seg = len(seg_pivot.reindex(columns=full_models).dropna(how='any'))
    n_bscan = len(bscan_pivot.reindex(columns=full_models).dropna(how='any'))

    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_outer[0],
        height_ratios=[max(n_vol, n_retr) * 1.15, n_seg * 1.15],
        width_ratios=[1, 1],
        hspace=0.35, wspace=0.9,
    )
    ax_vol = fig.add_subplot(gs_left[0, 0])
    ax_retr = fig.add_subplot(gs_left[0, 1])
    ax_seg = fig.add_subplot(gs_left[1, :])

    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[1],
        height_ratios=[n_cd + 0.6, n_bscan * 1.15],
        hspace=0.35,
    )
    ax_cd = fig.add_subplot(gs_right[0])
    ax_bscan = fig.add_subplot(gs_right[1])

    draw_panel(
        ax_vol, vol_pivot, 'Volume diagnosis', vol_models, metric='MCC',
        xtick_labels=['OCTCube', 'CLOSER'] if SIMPLIFY_VOLUME_DIAGNOSIS else None,
        ylabels=vol_ylabels,
    )
    draw_panel(ax_retr, retr_pivot, 'Retrieval', retr_models, metric='mAP',
               show_ylabels=False, ylabels=retr_ylabels)
    draw_panel(ax_cd, cd_pivot, 'Change detection', full_models, metric='MCC',
               ylabels=cd_ylabels)
    draw_panel(ax_seg, seg_pivot, 'Segmentation', full_models, metric='Dice',
               ylabels=seg_ylabels)
    draw_panel(ax_bscan, bscan_pivot, 'B-scan diagnosis', full_models, metric='MCC',
               ylabels=bscan_ylabels)


    legend_labels = [
        r'$\bf{' + m + '}$' if m == 'CLOSER' else m
        for m in all_models
    ]

    handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m]) for m in all_models]
    fig.legend(
        handles, legend_labels, loc='lower center', ncol=len(all_models),
        bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=9,
        title='Row color = winning model on that disease group',
    )
    # handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m]) for m in all_models]
    # fig.legend(
    #     handles, all_models, loc='lower center', ncol=len(all_models),
    #     bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9,
    #     title='Row color = winning model on that disease group',
    # )
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(BASE_PATH.parent / 'gen_results' / 'heatmap_summary.svg', dpi=300, bbox_inches='tight')
    fig.savefig(BASE_PATH.parent / 'gen_results' / 'heatmap_summary.pdf', dpi=300, bbox_inches='tight')
    print('Saved heatmap_summary.svg/pdf')


if __name__ == '__main__':
    main()
