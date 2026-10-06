"""
Build the grouped heatmap directly from your raw result CSVs, using a
single, fixed colormap (e.g. cividis or greyscale) to encode performance
value directly, instead of coloring each row by its winning model.
Since color now maps to value via one shared colorbar, no per-model
legend is needed -- the winning cell per row is instead marked by
larger, bold text (with a contrast-aware outline), independent of color.

Expected input files (adjust paths in main() as needed):
    3d_raw_results.csv        -- volume diagnosis, per-seed (column 'Run')
    retrieval_raw_results.csv -- retrieval, already one row per (model, dataset)
    chd_raw_results.csv       -- change detection, per-seed
    2d_raw_results.csv        -- B-scan diagnosis, per-seed
    sota_raw_seg_results.csv  -- segmentation, per-sample (one row per ID, Class)

Each loader aggregates its raw file down to one value per (Dataset, Model),
which is what the plotting code needs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from pathlib import Path

# ============================================================
# Set these to your directories.
# ============================================================
BASE_PATH = Path('/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/raw_results')
SAVE_PATH = Path('./__output/heatmaps')
(SAVE_PATH / 'gen_results').mkdir(parents=True, exist_ok=True)

all_models = ['RETFound', 'VisionFM', 'UrFound', 'OCTCube', 'MIRAGE', 'CLOSER']
vol_models = ['OCTCube (60 B-scans)', 'CLOSER (60 B-scans)', 'CLOSER (all B-scans)']
retr_models = ['OCTCube', 'CLOSER']

SIMPLIFY_VOLUME_DIAGNOSIS = True
if SIMPLIFY_VOLUME_DIAGNOSIS:
    vol_models = ['OCTCube (60 B-scans)', 'CLOSER (all B-scans)']

# ============================================================
# Colormap used for every panel's cell fill. Pick any name from either
# list below (both are standard matplotlib colormap categories -- see
# https://matplotlib.org/stable/users/explain/colors/colormaps.html).
#
# Sequential: single hue, lighter->darker (Greys is a plain greyscale).
SEQUENTIAL_CMAPS = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
]
# Sequential (2): a second matplotlib-provided set, mostly single-hue or
# tinted greyscale variants (e.g. 'gray', 'bone', 'copper', 'hot').
SEQUENTIAL2_CMAPS = [
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    'hot', 'afmhot', 'gist_heat', 'copper',
]

CMAP_NAME = 'Purples'  # any entry from SEQUENTIAL_CMAPS or SEQUENTIAL2_CMAPS
assert CMAP_NAME in SEQUENTIAL_CMAPS + SEQUENTIAL2_CMAPS, (
    f'"{CMAP_NAME}" is not in SEQUENTIAL_CMAPS or SEQUENTIAL2_CMAPS -- '
    f'pick one of: {SEQUENTIAL_CMAPS + SEQUENTIAL2_CMAPS}'
)


# ============================================================
# Per-raw-dataset-key metadata, used to build row labels dynamically at
# plot time (see build_grouped_ylabels below), rather than baking a fixed
# label string per key. This is what makes the "show the disease group
# only on the first row of each consecutive block" behavior correct
# regardless of sort order -- a static string can't know whether it will
# end up first or not in a given panel.
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
    'AMD/DME': 'AMD, DME', 'AMD': 'AMD', 'DME': 'DME', 'DR': 'DR',
    'DR/DME': 'DR, DME', 'Glaucoma': 'Glaucoma', 'MS': 'MS', 'RVO': 'RVO',
    'AMD/DME/RVO': 'AMD, DME, RVO', 'MH': 'MH', 'Multi-disease': 'Multi-disease',
}

dataset_detail = {
    'AMD_DME_3D': '2 classes', 'BanglaOCT2025': '3 classes', 'GAMMA': '3 classes',
    'Harvard_Glaucoma': '2 classes', 'M3Ret': '4 classes',
    'UMNandDukeSrinivasan_to_Noor_cross': '3 classes',
    'Noor_to_UMNandDukeSrinivasan_cross': '3 classes',
    'OCT_MD_MS': '2 classes', 'OIMHS': '2 classes', 'OLIVES': '2 classes',
    'POAGD': '2 classes', 'MARIO_T1': '4 classes', 'GOALS': '2 classes',
    'UMN': '2 classes', 'MMC_AMD': '4 classes', 'Noor_Eye_Hospital': '3 classes',
    'OLIVES_baseline': '2 classes', 'OAEFD_DME': '2 classes', 'OAEFD_DR': '3 classes',
    'Kermany_correct': '4 classes', 'OCTA500': '2 classes',
    'Duke_iAMD': '2 classes', 'OCTID': '5 classes', 'OCTDL': '7 classes',
    'OCT_C8': '8 classes', 'OCTAVE': '6 classes',
    'AMD_DME_3D_lesions': '2 lesions', 'AMD_SD_vol_layers': '1 layer',
    'AMD_SD_vol_lesions': '4 lesions', 'AROI_nosclera_layers': '3 layers',
    'AROI_nosclera_lesions': '3 lesions', 'Duke_DME_nosclera_layers': '7 layers',
    'Duke_DME_nosclera_lesions': '1 lesion', 'GOALS_layers': '3 layers',
    'OCTAVE_layers': '4 layers', 'OCTAVE_lesions': '6 lesions',
    'OIMHS_layers': '2 layers', 'OIMHS_lesions': '2 lesions',
    'RETOUCH_lesions': '3 lesions', 'RVO_Lesion_lesions': '2 lesions',
    'UMN_lesions': '1 lesion', 'Duke_iAMD_labeled_nosclera_70_layers': '2 layers',
    'RVO_ME': None, 'AMD_SD': None, 'AROI': None, 'Duke_DME': None, 'RETOUCH': None,
}

dataset_name = {
    'AMD_DME_3D': 'AMD-DME-3D', 'AMD_DME_3D_lesions': 'AMD-DME-3D',
    'BanglaOCT2025': 'BanglaOCT\n2025', 'GAMMA': 'GAMMA',
    'Harvard_Glaucoma': 'Harvard\nGlaucoma', 'M3Ret': 'M3Ret',
    'UMNandDukeSrinivasan_to_Noor_cross': 'NEH*',
    'Noor_to_UMNandDukeSrinivasan_cross': 'UMN+Duke-\nSrinivasan*',
    'OCT_MD_MS': 'OCT-MD-MS', 'OIMHS': 'OIMHS', 'OIMHS_layers': 'OIMHS',
    'OIMHS_lesions': 'OIMHS', 'OLIVES': 'OLIVES', 'POAGD': 'POAGD',
    'MARIO_T1': 'MARIO (T1)', 'GOALS': 'GOALS', 'GOALS_layers': 'GOALS',
    'RVO_ME': 'RVO-ME', 'RVO_Lesion_lesions': 'RVO-ME', 'UMN': 'UMN',
    'UMN_lesions': 'UMN', 'MMC_AMD': 'MMC-AMD', 'Noor_Eye_Hospital': 'NEH',
    'OLIVES_baseline': 'OLIVES', 'OAEFD_DME': 'OAEFD-DME', 'OAEFD_DR': 'OAEFD-DR',
    'Kermany_correct': 'Kermany', 'OCTA500': 'OCTA-500', 'Duke_iAMD': 'Duke iAMD',
    'Duke_iAMD_labeled_nosclera_70_layers': 'Duke iAMD*',
    'AMD_SD': 'AMD-SD', 'AMD_SD_vol_layers': 'AMD-SD', 'AMD_SD_vol_lesions': 'AMD-SD',
    'AROI': 'AROI', 'AROI_nosclera_layers': 'AROI', 'AROI_nosclera_lesions': 'AROI',
    'Duke_DME': 'Duke DME', 'Duke_DME_nosclera_layers': 'Duke DME',
    'Duke_DME_nosclera_lesions': 'Duke DME', 'RETOUCH': 'RETOUCH',
    'RETOUCH_lesions': 'RETOUCH', 'OCTID': 'OCTID', 'OCTDL': 'OCTDL',
    'OCT_C8': 'OCT-C8', 'OCTAVE': 'OCTAVE', 'OCTAVE_layers': 'OCTAVE',
    'OCTAVE_lesions': 'OCTAVE',
}

group_order = [
    'AMD/DME', 'AMD', 'DME', 'DR', 'DR/DME',
    'Glaucoma', 'MS', 'RVO', 'AMD/DME/RVO', 'MH', 'Multi-disease',
]


def sort_by_group(raw_index):
    def sort_key(k):
        cat = dataset_group.get(k)
        cat_rank = group_order.index(cat) if cat in group_order else len(group_order)
        return (cat_rank, k)
    return sorted(raw_index, key=sort_key)


def _mathtext_bold(text):
    escaped = text.replace(' ', r'\ ').replace('-', r'\text{-}')
    return r'$\bf{' + escaped + '}$'


def pad_detail(detail, min_len=12):
    if detail is None:
        return None
    if '1 layer' in detail:
        min_len = min_len + 1
    return detail.rjust(min_len)


def build_grouped_ylabels(sorted_keys):
    labels = []
    prev_cat = object()
    for k in sorted_keys:
        cat = dataset_group.get(k)
        detail = dataset_detail.get(k)
        name = dataset_name.get(k, k)
        detail = pad_detail(detail, min_len=16)

        if cat != prev_cat:
            bold_part = group_display.get(cat, cat or '')
            if detail:
                bold_part = f'{bold_part}    {detail}' if bold_part else f'  {detail}'
        else:
            bold_part = f'  {detail}' if detail else ''

        label = f'{_mathtext_bold(bold_part)}\n{name}' if bold_part else name
        labels.append(label)
        prev_cat = cat
    return labels


def draw_group_lines(ax, break_rows, x_start=-0.45):
    for row in break_rows:
        ax.plot(
            [x_start, ax.get_xlim()[1]], [row - 0.5, row - 0.5],
            color='black', linewidth=1.0, clip_on=False,
            transform=ax.get_yaxis_transform(),
        )


def draw_group_box(ax, x_start, y_top, y_bottom):
    ax.plot(
        [x_start, x_start], [y_top - 0.5, y_bottom + 0.5],
        color='black', linewidth=1.0, clip_on=False,
        transform=ax.get_yaxis_transform(),
    )


def draw_group_boxes(ax, groups, x_start=-0.23, y_pad=0.1):
    for start, end in groups:
        draw_group_box(ax, x_start, y_top=start + y_pad, y_bottom=end - y_pad)


# ============================================================
# Loaders -- identical to the previous version.
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


def _text_style_for_bg(rgba):
    """
    Pick a legible text color + stroke color for a cell, based on the
    background's perceptual luminance -- needed now that cell color comes
    from a shared colormap (not a per-model hue we chose for contrast),
    so light and dark cells can appear anywhere in any panel.
    """
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance > 0.65:
        return 'black', 'white'
    return 'white', 'black'


# ============================================================
# Plotting: color now comes from a single shared colormap (value-encoded),
# not from the winning model. The winning cell per row is still marked,
# but only via bold/larger text -- color no longer indicates "who won".
# ============================================================

def draw_panel(ax, pivot_df, title, models_list, metric, cmap,
                show_ylabels=True, xtick_labels=None, ylabels=None):
    pivot_df = pivot_df.reindex(columns=models_list).dropna(how='any')
    tasks = list(pivot_df.index)
    data = pivot_df.to_numpy()

    # Still compute the winning model per row (CLOSER preferred on a tie)
    # -- used only to decide which cell's text gets the bold/larger
    # treatment, not for cell color.
    closer_col_names = [m for m in models_list if 'CLOSER' in m]
    best_idx = np.zeros(len(tasks), dtype=int)
    for i in range(len(tasks)):
        row = data[i]
        row_max = row.max()
        is_tied_for_best = np.isclose(row, row_max, atol=1e-9)
        tied_models = [j for j in range(len(models_list)) if is_tied_for_best[j]]
        closer_tied = [j for j in tied_models if models_list[j] in closer_col_names]
        best_idx[i] = closer_tied[0] if closer_tied else tied_models[0]

    # Per-row relative scaling (same idea as the original color-by-winner
    # version): each cell's color reflects its rank within its OWN row
    # (0 = row's worst, 1 = row's best), not its absolute value relative
    # to other rows/panels. This is what makes worst/best clear within
    # every single row, regardless of how absolute performance varies
    # across tasks/metrics.
    rel = np.zeros_like(data, dtype=float)
    for i in range(len(tasks)):
        row = data[i]
        row_min, row_max = row.min(), row.max()
        rel[i] = (row - row_min) / (row_max - row_min + 1e-9)

    ax.imshow(rel, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0)

    for i in range(len(tasks)):
        for j in range(len(models_list)):
            val = data[i, j]
            rgba = cmap(rel[i, j])
            text_color, stroke_color = _text_style_for_bg(rgba)
            is_winner = (j == best_idx[i])
            if is_winner:
                ax.text(
                    j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7.6, color=text_color, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=1.8, foreground=stroke_color)],
                )
            else:
                ax.text(
                    j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=5.4, color=text_color, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=1.0, foreground=stroke_color)],
                )

    ax.set_xticks(range(len(models_list)))
    ax.set_xticklabels(xtick_labels or models_list, rotation=40, ha='right', fontsize=6.5)
    if show_ylabels:
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(ylabels if ylabels is not None else tasks, fontsize=6.2)
    else:
        ax.set_yticks([])
    set_two_part_title(ax, title, metric)
    ax.set_xticks(np.arange(-0.5, len(models_list), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0)
    ax.tick_params(which='minor', length=0)
    return len(tasks)


def _load_sort_label(loader_fn, *loader_args, **loader_kwargs):
    pivot = loader_fn(*loader_args, **loader_kwargs)
    sorted_keys = sort_by_group(pivot.index)
    pivot = pivot.loc[sorted_keys]
    ylabels = build_grouped_ylabels(sorted_keys)
    return pivot, ylabels


def main():
    vol_pivot, vol_ylabels = _load_sort_label(
        load_seed_level, BASE_PATH / '3d_raw_results.csv', metric='MCC')
    retr_pivot, retr_ylabels = _load_sort_label(
        load_retrieval, BASE_PATH / 'retrieval_raw_results.csv', metric='mAP')
    retr_pivot /= 100.0  # match the 0-1 scale of the other metrics
    cd_pivot, cd_ylabels = _load_sort_label(
        load_seed_level, BASE_PATH / 'chd_raw_results.csv', metric='MCC')
    bscan_pivot, bscan_ylabels = _load_sort_label(
        load_seed_level, BASE_PATH / '2d_raw_results.csv', metric='MCC')
    seg_pivot, seg_ylabels = _load_sort_label(
        load_segmentation, BASE_PATH / 'sota_raw_seg_results.csv', metric='Dice')

    full_models = all_models

    cmap = plt.get_cmap(CMAP_NAME)

    fig = plt.figure(figsize=(11, 15))
    gs_outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.55)

    n_vol = len(vol_pivot.reindex(columns=vol_models).dropna(how='any'))
    n_retr = len(retr_pivot.reindex(columns=retr_models).dropna(how='any'))
    n_cd = len(cd_pivot.reindex(columns=full_models).dropna(how='any'))
    n_seg = len(seg_pivot.reindex(columns=full_models).dropna(how='any'))
    n_bscan = len(bscan_pivot.reindex(columns=full_models).dropna(how='any'))

    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_outer[0],
        height_ratios=[max(n_vol, n_retr) * 1.15, n_seg * 1.15],
        width_ratios=[1, 1],
        hspace=0.2, wspace=0.9,
    )
    ax_vol = fig.add_subplot(gs_left[0, 0])
    ax_retr = fig.add_subplot(gs_left[0, 1])
    ax_seg = fig.add_subplot(gs_left[1, :])

    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[1],
        height_ratios=[n_cd + 0.6, n_bscan * 1.15],
        hspace=0.2,
    )
    ax_cd = fig.add_subplot(gs_right[0])
    ax_bscan = fig.add_subplot(gs_right[1])

    draw_panel(
        ax_vol, vol_pivot, 'Volume diagnosis', vol_models, metric='MCC',
        cmap=cmap,
        xtick_labels=['OCTCube', 'CLOSER'] if SIMPLIFY_VOLUME_DIAGNOSIS else None,
        ylabels=vol_ylabels,
    )
    draw_panel(ax_retr, retr_pivot, 'Retrieval', retr_models, metric='mAP',
               cmap=cmap, show_ylabels=False, ylabels=retr_ylabels)
    draw_panel(ax_cd, cd_pivot, 'Change detection', full_models, metric='MCC',
               cmap=cmap, ylabels=cd_ylabels)
    draw_panel(ax_seg, seg_pivot, 'Segmentation', full_models, metric='Dice',
               cmap=cmap, ylabels=seg_ylabels)
    draw_panel(ax_bscan, bscan_pivot, 'B-scan diagnosis', full_models, metric='MCC',
               cmap=cmap, ylabels=bscan_ylabels)

    draw_group_boxes(
        ax_vol, [(0, 3), (4, 4), (5, 6), (7, 9), (10, 10), (11, 11), (12, 13)],
        x_start=-0.64, y_pad=0.16,
    )
    draw_group_boxes(ax_cd, [(0, 0)], x_start=-0.22, y_pad=0.16)
    draw_group_boxes(
        ax_seg, [(0, 1), (2, 6), (7, 8), (9, 9), (10, 10), (11, 11), (12, 13), (14, 15)],
        x_start=-0.22, y_pad=0.16,
    )
    draw_group_boxes(
        ax_bscan,
        [(0, 4), (5, 7), (8, 8), (9, 9), (10, 11), (12, 15), (16, 16), (17, 17), (18, 22)],
        x_start=-0.22, y_pad=0.16,
    )

    # Reserve a clear margin at the bottom of the figure for the colorbar,
    # separate from the panels -- otherwise the (often long, rotated)
    # model-name x-tick labels on the lowest panels can collide with the
    # colorbar and its label.
    fig.subplots_adjust(bottom=0.10)

    # Single shared colorbar in place of the old per-model legend. Since
    # color now encodes each cell's RELATIVE rank within its own row (not
    # an absolute value shared across panels), the colorbar is labeled
    # accordingly, with "Worst in row" / "Best in row" endpoints rather
    # than numbers. The descriptive label sits just above the bar itself
    # (rather than matplotlib's default placement below it), so it reads
    # as a caption for the bar rather than competing with the tick labels.
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_left, cbar_bottom, cbar_width, cbar_height = 0.3, 0.02, 0.4, 0.015
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Worst in row', 'Best in row'])
    fig.text(
        cbar_left + cbar_width / 2, cbar_bottom + cbar_height + 0.005,
        'Relative performance within each row',
        ha='center', va='bottom', fontsize=9,
    )

    fig.savefig(SAVE_PATH / 'gen_results' / 'heatmap_colormap.svg', dpi=300, bbox_inches='tight')
    fig.savefig(SAVE_PATH / 'gen_results' / 'heatmap_colormap.pdf', dpi=300, bbox_inches='tight')
    print('Saved heatmap_colormap.svg/pdf')


if __name__ == '__main__':
    main()
