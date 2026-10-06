"""
Build the grouped, color-by-winner heatmap directly from your raw result CSVs.

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
import matplotlib.patheffects as pe
import colorsys
from pathlib import Path
from visualization.plot_utils import get_palette

# ============================================================
# Set this to the directory containing your raw result CSVs.
# ============================================================
BASE_PATH = Path('/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/raw_results')
SAVE_PATH = Path('./__output/heatmaps')
(SAVE_PATH / 'gen_results').mkdir(parents=True, exist_ok=True)

all_models = ['RETFound', 'VisionFM', 'UrFound', 'OCTCube', 'MIRAGE', 'CLOSER']
vol_models = ['OCTCube (60 B-scans)', 'CLOSER (60 B-scans)', 'CLOSER (all B-scans)']
retr_models = ['OCTCube', 'CLOSER']

# Set to True to show Volume diagnosis as a simple OCTCube vs. CLOSER
# comparison (dropping the 'CLOSER (60 B-scans)' column) -- the full
# 3-way breakdown is already reported precisely in the main results
# table, so this overview figure doesn't need that extra granularity.
SIMPLIFY_VOLUME_DIAGNOSIS = True
if SIMPLIFY_VOLUME_DIAGNOSIS:
    vol_models = ['OCTCube (60 B-scans)', 'CLOSER (all B-scans)']


def soften_for_fill(hex_color, model_label, darken_factor=0.82,
                     yellow_darken_factor=0.9, closer_name='CLOSER'):
    """
    Same idea as get_model_colors' darken/brighten logic, but tuned for
    large filled heatmap cells rather than LaTeX text: raw palette colors
    that look fine as a small boxplot patch or a line/marker read as too
    bright/loud once they fill an entire grid cell edge-to-edge, so we
    mute them a bit here. Yellow gets darkened more (otherwise looks
    washed-out/pale at cell size); CLOSER is left basically as-is since
    it's meant to be the visually dominant color in this figure.
    """
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    is_yellow = 0.11 <= h <= 0.19
    is_closer = closer_name.lower() in model_label.lower()
    if is_closer:
        pass  # keep CLOSER's color as the palette defines it
    elif is_yellow:
        v *= yellow_darken_factor
        s = min(1.0, s * 1.1)
    else:
        v *= darken_factor
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (r2, g2, b2)


# Build the color map from your existing palette utility, covering every
# model-name string that appears in any panel (so substring matching in
# get_palette resolves each variant, e.g. 'CLOSER (60 B-scans)' vs 'CLOSER'),
# then soften each color for use as a large heatmap cell fill.
_all_model_strings = list(dict.fromkeys(all_models + vol_models + retr_models))
_raw_model_colors = get_palette(_all_model_strings)
model_colors = {
    name: soften_for_fill(hexcol, name) for name, hexcol in _raw_model_colors.items()
}

# ============================================================
# Per-raw-dataset-key metadata, used to build row labels dynamically at
# plot time (see build_grouped_ylabels below), rather than baking a fixed
# label string per key. This is what makes the "show the disease group
# only on the first row of each consecutive block" behavior correct
# regardless of sort order -- a static string can't know whether it will
# end up first or not in a given panel.
#
#   dataset_group  : category key used both for sorting (sort_by_group)
#                    and for looking up the human-readable group text
#                    (via group_display) shown in bold.
#   dataset_detail : the "(N classes)" / "N lesions" / "N layers" bit
#                    shown alongside the group text -- None if there's
#                    nothing to add (e.g. single-class segmentation-only
#                    entries with no class count).
#   dataset_name   : plain display name, shown on the second (non-bold)
#                    line, always.
#
# Any raw key missing from these dicts falls back to its raw string
# unchanged, so a forgotten entry is visible (ugly) rather than silently
# dropped.
# ============================================================
dataset_group = {
    'AMD_DME_3D': 'AMD/DME', 'AMD_DME_3D_lesions': 'AMD/DME',
    'BanglaOCT2025': 'AMD', 'MARIO_T1': 'AMD',
    'MMC_AMD': 'AMD', 'Duke_iAMD': 'AMD', 'Duke_iAMD_labeled_nosclera_70_layers': 'AMD',
    'AMD_SD': 'AMD', 'AMD_SD_vol_layers': 'AMD', 'AMD_SD_vol_lesions': 'AMD',
    'AROI': 'AMD', 'AROI_nosclera_layers': 'AMD', 'AROI_nosclera_lesions': 'AMD',  # unverified, please check
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
    'RETOUCH': 'AMD/DME/RVO', 'RETOUCH_lesions': 'AMD/DME/RVO',  # from memory, please verify
    'OCTA500': 'Multi-disease', 'OCTID': 'Multi-disease', 'OCTDL': 'Multi-disease',
    'OCT_C8': 'Multi-disease', 'OCTAVE': 'Multi-disease',
    'OCTAVE_layers': 'Multi-disease', 'OCTAVE_lesions': 'Multi-disease',
}

# Human-readable bold text for each category key above.
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

dataset_detail = {
    # Classification: class counts
    'AMD_DME_3D': '2 classes', 'BanglaOCT2025': '3 classes', 'GAMMA': '3 classes',
    'Harvard_Glaucoma': '2 classes', 'M3Ret': '4 classes',
    'UMNandDukeSrinivasan_to_Noor_cross': '3 classes',
    'Noor_to_UMNandDukeSrinivasan_cross': '3 classes',
    'OCT_MD_MS': '2 classes', 'OIMHS': '2 classes', 'OLIVES': '2 classes',
    'POAGD': '2 classes', 'MARIO_T1': '4 classes', 'GOALS': '2 classes',
    'UMN': '2 classes', 'MMC_AMD': '4 classes', 'Noor_Eye_Hospital': '3 classes',
    'OLIVES_baseline': '2 classes', 'OAEFD_DME': '2 classes', 'OAEFD_DR': '3 classes',
    'Kermany_correct': '4 classes', 'OCTA500': '2 classes',  # binary "Diseased" merges CNV/DR/AMD
    'Duke_iAMD': '2 classes', 'OCTID': '5 classes', 'OCTDL': '7 classes',
    'OCT_C8': '8 classes', 'OCTAVE': '6 classes',
    # Segmentation: structure counts
    'AMD_DME_3D_lesions': '2 lesions', 'AMD_SD_vol_layers': '1 layer',
    'AMD_SD_vol_lesions': '4 lesions', 'AROI_nosclera_layers': '3 layers',
    'AROI_nosclera_lesions': '3 lesions', 'Duke_DME_nosclera_layers': '7 layers',
    'Duke_DME_nosclera_lesions': '1 lesion', 'GOALS_layers': '3 layers',
    'OCTAVE_layers': '4 layers', 'OCTAVE_lesions': '6 lesions',
    'OIMHS_layers': '2 layers', 'OIMHS_lesions': '2 lesions',
    'RETOUCH_lesions': '3 lesions', 'RVO_Lesion_lesions': '2 lesions',
    'UMN_lesions': '1 lesion', 'Duke_iAMD_labeled_nosclera_70_layers': '2 layers',
    # Segmentation-only entries with nothing to add here:
    'RVO_ME': None, 'AMD_SD': None, 'AROI': None, 'Duke_DME': None, 'RETOUCH': None,
}

dataset_name = {
    'AMD_DME_3D': 'AMD-DME-3D', 'AMD_DME_3D_lesions': 'AMD-DME-3D',
    'BanglaOCT2025': 'BanglaOCT\n2025', 'GAMMA': 'GAMMA',
    'Harvard_Glaucoma': 'Harvard\nGlaucoma', 'M3Ret': 'M3Ret',
    'UMNandDukeSrinivasan_to_Noor_cross': 'NEH*', # \u2190 UMN\n+Duke-Sr',
    'Noor_to_UMNandDukeSrinivasan_cross': 'UMN+Duke-\nSrinivasan*', #\n \u2190 NEH',
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

# Canonical group order -- edit to taste (this is the order categories
# will appear top-to-bottom in every panel).
group_order = [
    'AMD/DME', 'AMD', 'DME', 'DR', 'DR/DME',
    'Glaucoma', 'MS', 'RVO', 'AMD/DME/RVO', 'MH', 'Multi-disease',
]


def sort_by_group(raw_index):
    """Sort raw dataset keys by category (per group_order), then
    alphabetically within each category. Keys missing from dataset_group
    are pushed to the end, sorted alphabetically, rather than silently
    dropped -- so a forgotten entry is visible (out of place) instead of
    invisible."""
    def sort_key(k):
        cat = dataset_group.get(k)
        cat_rank = group_order.index(cat) if cat in group_order else len(group_order)
        return (cat_rank, k)
    return sorted(raw_index, key=sort_key)


def _mathtext_bold(text):
    """Wrap `text` for matplotlib mathtext bold rendering: spaces must be
    escaped (mathtext otherwise collapses them), and a literal hyphen
    needs \\text{-} or it's parsed as a minus sign."""
    escaped = text.replace(' ', r'\ ').replace('-', r'\text{-}')
    return r'$\bf{' + escaped + '}$'


def pad_detail(detail, min_len=12):
    """Right-pad `detail` with spaces so it's at least `min_len` characters,
    for more visually consistent widths across rows (e.g. '1 layer' vs
    '6 lesions' vs '8 classes'). Leaves None untouched."""
    if '1 layer' in detail:
        min_len = min_len + 1
    if detail is None:
        return None
    return detail.rjust(min_len)


def build_grouped_ylabels(sorted_keys):
    """
    Build y-axis tick labels for a panel's rows, given the *already
    sorted* raw dataset keys for that panel: bold disease/category text
    on top (shown in full only on the first row of each consecutive
    group -- just the '(detail)' part on subsequent rows of the same
    group), plain dataset name below, always.
    """
    labels = []
    prev_cat = object()  # sentinel that can't equal any real category
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
    """Draw a horizontal line above each row index in break_rows
    (0-indexed), extending from x_start (negative = into the label
    margin, left of the plot) across the full row width."""
    for row in break_rows:
        ax.plot(
            [x_start, ax.get_xlim()[1]], [row - 0.5, row - 0.5],
            color='black', linewidth=1.0, clip_on=False,
            transform=ax.get_yaxis_transform(),
        )

def draw_group_box(ax, x_start, y_top, y_bottom):
    """Single vertical line at x_start, spanning from row y_top to row
    y_bottom (0-indexed), useful together with draw_group_lines to close
    off a box around the label margin."""
    ax.plot(
        [x_start, x_start], [y_top - 0.5, y_bottom + 0.5],
        color='black', linewidth=1.0, clip_on=False,
        transform=ax.get_yaxis_transform(),
    )

# ============================================================
# Loaders: raw CSV -> pivoted (Dataset x Model) DataFrame of mean metric
# ============================================================

def load_seed_level(path, metric='MCC', dataset_col='Dataset', model_col='Model'):
    """
    For CSVs with one row per (Dataset, Model, Run/seed) -- e.g. 2d_raw_results.csv,
    3d_raw_results.csv, chd_raw_results.csv. Averages `metric` over seeds.
    """
    df = pd.read_csv(path)
    pivot = df.groupby([dataset_col, model_col])[metric].mean().unstack(model_col)
    return pivot


def load_retrieval(path, metric='mAP'):
    """
    retrieval_raw_results.csv is already one row per (model_name, dataset_name),
    no seeds to average over.
    """
    df = pd.read_csv(path)
    pivot = df.pivot(index='dataset_name', columns='model_name', values=metric)
    return pivot


def load_segmentation(path, metric='Dice'):
    """
    sota_raw_seg_results.csv (and equivalents) has one row per (ID, Class, Model),
    with 'Dataset' already encoding dataset + structure type (e.g.
    'AMD_DME_3D_lesions'). Average over ID and Class (ignoring NaNs, which mark
    cases where both prediction and ground truth were empty for that class/slice).
    """
    df = pd.read_csv(path)
    pivot = df.groupby(['Dataset', 'Model'])[metric].mean().unstack('Model')
    return pivot


# ============================================================
# Title helper: '<name> (<metric>)', name bold, metric non-bold & smaller,
# placed immediately after the bold name (not independently re-centered --
# the pair sits slightly left of true axes-center since only the bold name
# was centered before the metric was appended, which is a fine tradeoff
# for how small the parenthetical is).
# ============================================================

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


# ============================================================
# Plotting (same encoding as before: color = winning model, opacity = margin
# within the row, numbers in every cell, winner's number bigger/white/bold).
# ============================================================

def draw_panel(ax, pivot_df, title, models_list, metric, show_ylabels=True,
                xtick_labels=None, ylabels=None):
    # Keep only the requested models, in the requested order; drop any
    # dataset (row) that's missing a value for one of them.
    pivot_df = pivot_df.reindex(columns=models_list).dropna(how='any')
    tasks = list(pivot_df.index)
    data = pivot_df.to_numpy()

    # Winning model per row: pick CLOSER on a tie (within floating-point
    # tolerance) rather than whichever model happens to come first in the
    # column order, so CLOSER is always highlighted when it's tied for best.
    closer_col_names = [m for m in models_list if 'CLOSER' in m]
    best_idx = np.zeros(len(tasks), dtype=int)
    for i in range(len(tasks)):
        row = data[i]
        row_max = row.max()
        is_tied_for_best = np.isclose(row, row_max, atol=1e-9)
        tied_models = [j for j in range(len(models_list)) if is_tied_for_best[j]]
        # If CLOSER (or a CLOSER variant) is among the tied-for-best models,
        # highlight that one; otherwise default to the first tied index.
        closer_tied = [
            j for j in tied_models if models_list[j] in closer_col_names
        ]
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
    """Load a pivot, sort its rows by disease/task group, and build the
    matching dynamic y-labels in one step -- keeping the pivot's index as
    raw dataset keys (labels are passed to draw_panel separately, not
    baked into the DataFrame)."""
    pivot = loader_fn(*loader_args, **loader_kwargs)
    sorted_keys = sort_by_group(pivot.index)
    pivot = pivot.loc[sorted_keys]
    ylabels = build_grouped_ylabels(sorted_keys)
    return pivot, ylabels

def draw_group_boxes(ax, groups, x_start=-0.23, y_pad=0.1):
    """
    Draw one vertical box-line per group, given a list of (start_row,
    end_row) tuples (0-indexed, inclusive). y_pad shrinks each line
    slightly inward from the exact row boundaries -- same idea as your
    0.1 / 5.2 vs. 3.75 / 6.9 offsets, just computed automatically from
    each group's own start/end instead of typed by hand per group.
    """
    for start, end in groups:
        draw_group_box(ax, x_start, y_top=start + y_pad, y_bottom=end - y_pad)


def main():
    # ---- Edit BASE_PATH above to point at your files' directory ----
    vol_pivot, vol_ylabels = _load_sort_label(
        load_seed_level, BASE_PATH / '3d_raw_results.csv', metric='MCC')
    retr_pivot, retr_ylabels = _load_sort_label(
        load_retrieval, BASE_PATH / 'retrieval_raw_results.csv', metric='mAP')
    # Divide retrieval results by 100 to match the scale of the other metrics (0-1)
    retr_pivot /= 100.0
    cd_pivot, cd_ylabels = _load_sort_label(
        load_seed_level, BASE_PATH / 'chd_raw_results.csv', metric='MCC')
    bscan_pivot, bscan_ylabels = _load_sort_label(
        load_seed_level, BASE_PATH / '2d_raw_results.csv', metric='MCC')
    seg_pivot, seg_ylabels = _load_sort_label(
        load_segmentation, BASE_PATH / 'sota_raw_seg_results.csv', metric='Dice')

    full_models = all_models

    fig = plt.figure(figsize=(11, 13))
    # Left column: volume diagnosis + retrieval on top, segmentation
    # spanning that same width below. Right column: change detection
    # (compact) directly above B-scan diagnosis -- since they share one
    # column, they come out the same width automatically. Left/right
    # outer columns are equal width, so segmentation ends up the same
    # width as change detection / B-scan diagnosis too.
    gs_outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.55)

    n_vol = len(vol_pivot.reindex(columns=vol_models).dropna(how='any'))
    n_retr = len(retr_pivot.reindex(columns=retr_models).dropna(how='any'))
    n_cd = len(cd_pivot.reindex(columns=full_models).dropna(how='any'))
    n_seg = len(seg_pivot.reindex(columns=full_models).dropna(how='any'))
    n_bscan = len(bscan_pivot.reindex(columns=full_models).dropna(how='any'))

    # Left column: top row (volume diagnosis, retrieval) sized to the
    # taller of the two; bottom row (segmentation) sized to its own row
    # count. A smaller hspace than before pulls segmentation up closer to
    # the row above it, using the freed space rather than changing either
    # panel's actual size.
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_outer[0],
        height_ratios=[max(n_vol, n_retr) * 1.15, n_seg * 1.15],
        width_ratios=[1, 1],
        hspace=0.2, wspace=0.9,
    )
    ax_vol = fig.add_subplot(gs_left[0, 0])
    ax_retr = fig.add_subplot(gs_left[0, 1])
    ax_seg = fig.add_subplot(gs_left[1, :])

    # Right column: change detection gets just enough height for its own
    # row(s) with a small margin; B-scan diagnosis gets the rest. Same
    # tightened hspace as the left column, pulling B-scan diagnosis up
    # closer to change detection.
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[1],
        height_ratios=[n_cd + 0.6, n_bscan * 1.15],
        hspace=0.2,
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


    draw_group_boxes(
        ax_vol,
        [(0, 3), (4, 4), (5, 6), (7, 9), (10, 10), (11, 11), (12, 13)],
        x_start=-0.64,
        y_pad=0.16
    )

    draw_group_boxes(
        ax_cd,
        [(0, 0)],
        x_start=-0.22,
        y_pad=0.16
    )

    draw_group_boxes(
        ax_seg,
        [(0, 1), (2, 6), (7, 8), (9, 9), (10, 10), (11, 11), (12, 13), (14, 15)],
        x_start=-0.22,
        y_pad=0.16
    )

    draw_group_boxes(
        ax_bscan,
        [(0, 4), (5, 7), (8, 8), (9, 9), (10, 11), (12, 15), (16, 16), (17, 17), (18, 22)],
        x_start=-0.22,
        y_pad=0.16
    )

    legend_labels = [
        r'$\bf{' + m + '}$' if m == 'CLOSER' else m
        for m in all_models
    ]

    handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m]) for m in all_models]
    fig.legend(
        handles, legend_labels, loc='lower center', ncol=len(all_models),
        bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=9,
        title='Row color = winning model on that disease group',
    )

    # handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m]) for m in all_models]
    # fig.legend(
    #     handles, all_models, loc='lower center', ncol=len(all_models),
    #     bbox_to_anchor=(0.5, 0), frameon=False, fontsize=9,
    #     title='Row color = winning model on that task',
    # )
    # fig.suptitle('Performance across all datasets, grouped by task type', fontsize=13, y=1.0)
    fig.savefig(SAVE_PATH / 'gen_results' / 'heatmap_real_data.svg', dpi=300, bbox_inches='tight')
    fig.savefig(SAVE_PATH / 'gen_results' / 'heatmap_real_data.pdf', dpi=300, bbox_inches='tight')
    print('Saved heatmap_real_data.svg/pdf')


if __name__ == '__main__':
    main()
