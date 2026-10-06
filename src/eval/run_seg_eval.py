import argparse
from pathlib import Path
import json

import tqdm
import numpy as np
from skimage import io
from skimage.transform import resize
# from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
import pandas as pd

from mutils.misc import SortingHelpFormatter
from visualization.plot_utils import LESION_CLASSES



def get_args():
    parser = argparse.ArgumentParser(
        'Evaluate segmentation results',
        formatter_class=SortingHelpFormatter
    )
    parser.add_argument(
        '-d', '--datasets_path', type=str, default='./_datasets/Segmentation/',
        help='Path to the datasets directory. (default: %(default)s)'
    )
    parser.add_argument(
        '-m', '--model_path', type=str, required=True,
        help='Path to the trained model directory. It expects a subdirectory'
            ' "preds" with the predictions. (required)'
    )
    parser.add_argument(
        '-e', '--external', type=str, default=None,
        help='Name of the external dataset on which the model made the '
        ' predictions. (default: %(default)s)'
    )
    parser.add_argument(
        '--test_alternative', type=str, default=None,
        help='Name of the alternative test set on which the model made the '
        ' predictions. (default: %(default)s)'
    )
    parser.add_argument(
        '--ignore_bg', action='store_true',
        help='Ignore the background class when computing metrics.'
            ' (default: %(default)s)'
    )
    parser.add_argument('--no_ignore_bg', dest='ignore_bg', action='store_false')
    parser.set_defaults(ignore_bg=True)
    parser.add_argument(
        '--empty_sets_nan', action='store_true',
        help='Return NaN if the prediction OR the ground truth is empty.'
            ' (default: %(default)s)'
    )
    parser.add_argument('--no_empty_sets_nan', dest='empty_sets_nan', action='store_false')
    parser.set_defaults(empty_sets_nan=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def dice_score(y_pred, y_true, empty_sets_nan=True):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    sum_true = np.sum(y_true)
    sum_pred = np.sum(y_pred)
    if sum_true == 0 and sum_pred == 0:
        return np.nan if empty_sets_nan else 1.0
    intersection = np.sum(y_true * y_pred)
    return 2.0 * intersection / (sum_true + sum_pred + 1e-6)


def iou_score(y_pred, y_true, empty_sets_nan=True):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    sum_true = np.sum(y_true)
    sum_pred = np.sum(y_pred)
    if sum_true == 0 and sum_pred == 0:
        return np.nan if empty_sets_nan else 1.0
    intersection = np.sum(y_true * y_pred)
    union = sum_true + sum_pred - intersection
    return intersection / (union + 1e-6)


# def get_edges(mask):
#     """Extract surface/edge pixels using binary erosion + XOR, matching MONAI."""
#     eroded = binary_erosion(mask)
#     return mask ^ eroded  # XOR gives the boundary pixels


def get_edges(mask):
    """Extract surface pixels using 2D erosion applied slice-by-slice."""
    if mask.ndim == 3:
        # Use 2D structuring element to avoid erasing thin structures
        struct = generate_binary_structure(2, 1)  # 2D cross
        struct_3d = struct[np.newaxis]  # shape (1, 3, 3) — no erosion across slices
        eroded = binary_erosion(mask, structure=struct_3d)
    else:
        eroded = binary_erosion(mask)
    return mask ^ eroded


def hausdorff_distance_95(y_pred, y_true, percentile=95):
    """
    y_pred, y_true: binary numpy arrays, any shape (2D or 3D)
    Matches MONAI's compute_hausdorff_distance implementation.
    """
    edges_pred = get_edges(y_pred)
    edges_true = get_edges(y_true)

    # Get coordinates of edge pixels
    coords_pred = np.argwhere(edges_pred)
    coords_true = np.argwhere(edges_true)

    if len(coords_pred) == 0 or len(coords_true) == 0:
        return np.nan

    # Compute pairwise distances
    tree_true = cKDTree(coords_true)
    tree_pred = cKDTree(coords_pred)

    # Directed distances
    dist_pred_to_true, _ = tree_true.query(coords_pred)
    dist_true_to_pred, _ = tree_pred.query(coords_true)

    # Symmetric HD95
    hd95 = max(
        np.percentile(dist_pred_to_true, percentile),
        np.percentile(dist_true_to_pred, percentile)
    )
    return hd95


def volume_hausdorff_distance(y_pred, y_true, percentile=95, empty_sets_nan=True):
    unique_pred = np.unique(y_pred)
    unique_true = np.unique(y_true)

    if unique_true.size == 1 and unique_pred.size == 1 and unique_true[0] == unique_pred[0]:
        return 0.0
    if unique_true.size == 1 or unique_pred.size == 1:
        if empty_sets_nan:
            return np.nan
        else:
            return float(np.sqrt(sum(s**2 for s in y_true.shape)))

    hd95_value = hausdorff_distance_95(
        y_pred.astype(bool),
        y_true.astype(bool),
        percentile
    )
    return hd95_value


def get_edges_2d(mask):
    """Extract surface pixels of a single 2D mask using binary erosion + XOR."""
    eroded = binary_erosion(mask)
    return mask ^ eroded


def hausdorff_distance_95_2d(y_pred, y_true, percentile=95):
    """
    y_pred, y_true: binary 2D numpy arrays (single B-scan).
    Assumes both masks are non-degenerate (checked by the caller).
    """
    edges_pred = get_edges_2d(y_pred)
    edges_true = get_edges_2d(y_true)

    coords_pred = np.argwhere(edges_pred)
    coords_true = np.argwhere(edges_true)

    if len(coords_pred) == 0 or len(coords_true) == 0:
        # Degenerate boundary (e.g., single-pixel mask eroded to nothing)
        return np.nan

    tree_true = cKDTree(coords_true)
    tree_pred = cKDTree(coords_pred)

    dist_pred_to_true, _ = tree_true.query(coords_pred)
    dist_true_to_pred, _ = tree_pred.query(coords_true)

    hd95 = max(
        np.percentile(dist_pred_to_true, percentile),
        np.percentile(dist_true_to_pred, percentile),
    )
    return hd95


def slice_hausdorff_distance_95(y_pred_slice, y_true_slice, percentile=95, empty_sets_nan=True):
    """
    HD95 for a single 2D slice, with degenerate-mask handling consistent
    with dice_score/iou_score (checks equality of constant masks, not just
    'both constant').
    """
    sum_true = y_true_slice.sum()
    sum_pred = y_pred_slice.sum()

    if sum_true == 0 and sum_pred == 0:
        # Both empty: no boundary defined for this class in this slice.
        return np.nan if empty_sets_nan else 0.0

    if sum_true == 0 or sum_pred == 0:
        # One empty, the other not: maximal error for this slice.
        if empty_sets_nan:
            return np.nan
        else:
            return float(np.sqrt(sum(s ** 2 for s in y_true_slice.shape)))  # slice diagonal

    return hausdorff_distance_95_2d(
        y_pred_slice.astype(bool),
        y_true_slice.astype(bool),
        percentile,
    )


def volume_hausdorff_distance_2d(y_pred, y_true, percentile=95, empty_sets_nan=True):
    """
    Volume-level HD95, computed as the mean of per-B-scan (2D) HD95 values.

    y_pred, y_true: arrays of shape (num_slices, H, W) for a volume, or
    (H, W) for a single slice.
    """
    if y_true.ndim == 2:
        return slice_hausdorff_distance_95(y_pred, y_true, percentile, empty_sets_nan)

    slice_values = np.array([
        slice_hausdorff_distance_95(y_pred[i], y_true[i], percentile, empty_sets_nan)
        for i in range(y_true.shape[0])
    ], dtype=float)

    if np.all(np.isnan(slice_values)):
        return np.nan
    return float(np.nanmean(slice_values))


def get_boundary_positions(mask, surface='top'):
    """
    For a 2D binary mask (H, W), return, per column (A-scan), the row index
    of the top (or bottom) foreground pixel, and a boolean mask of which
    columns actually have a defined boundary (i.e., non-empty column).
    """
    H, _ = mask.shape
    has_fg = mask.any(axis=0)
    if surface == 'top':
        idx = np.argmax(mask, axis=0)
    elif surface == 'bottom':
        idx = H - 1 - np.argmax(mask[::-1, :], axis=0)
    else:
        raise ValueError(f"Unknown surface: '{surface}'. Use 'top' or 'bottom'.")
    return idx, has_fg


def slice_mean_absolute_distance(pred_slice, true_slice, surface='top', empty_sets_nan=True):
    """
    Mean absolute distance (MAD), in pixels, between predicted and ground-truth
    boundary of a class region, for a single 2D slice. Boundary is defined
    per-column (A-scan) as the row index of the top (or bottom) foreground pixel.
    """
    H, _ = true_slice.shape
    pos_true, valid_true = get_boundary_positions(true_slice, surface)
    pos_pred, valid_pred = get_boundary_positions(pred_slice, surface)

    if not valid_true.any() and not valid_pred.any():
        return np.nan if empty_sets_nan else 0.0
    if not valid_true.any():
        # GT has no instance of this class in this slice, but prediction does.
        return np.nan if empty_sets_nan else float(H)

    # Evaluate only columns where GT boundary is defined; penalize columns
    # where the prediction missed the boundary entirely.
    distances = np.where(
        valid_pred,
        np.abs(pos_pred.astype(float) - pos_true.astype(float)),
        float(H),
    )
    return float(distances[valid_true].mean())


def volume_mean_absolute_distance(y_pred, y_true, surface='both', empty_sets_nan=True):
    """
    Volume-level MAD, computed as the mean of per-B-scan (2D) MAD values.
    surface='both' averages the top- and bottom-boundary MAD per slice
    (appropriate for a class defined as the region between two surfaces).
    """
    def _slice_mad(pred_slice, true_slice):
        if surface == 'both':
            mad_top = slice_mean_absolute_distance(pred_slice, true_slice, 'top', empty_sets_nan)
            mad_bottom = slice_mean_absolute_distance(pred_slice, true_slice, 'bottom', empty_sets_nan)
            vals = [v for v in (mad_top, mad_bottom) if not np.isnan(v)]
            return float(np.mean(vals)) if vals else np.nan
        return slice_mean_absolute_distance(pred_slice, true_slice, surface, empty_sets_nan)

    if y_true.ndim == 2:
        return _slice_mad(y_pred, y_true)

    slice_values = np.array(
        [_slice_mad(y_pred[i], y_true[i]) for i in range(y_true.shape[0])],
        dtype=float,
    )
    if np.all(np.isnan(slice_values)):
        return np.nan
    return float(np.nanmean(slice_values))



def is_lesion_class(class_name):
    return class_name in LESION_CLASSES



def print_mean_metrics(results_df):
    print('  Dice: {:.2f}'.format(results_df['Dice'].mean() * 100))
    print('  IoU: {:.2f}'.format(results_df['IoU'].mean() * 100))
    print('  HD95: {:.2f}'.format(results_df['HD95'].mean()))
    if 'HD952D' in results_df.columns:
        print('  HD952D: {:.2f}'.format(results_df['HD952D'].mean()))
    if 'MAD' in results_df.columns:
        print('  MAD (layers only): {:.2f}'.format(results_df['MAD'].mean()))


def translate_to_dukeiamd_from_aroi(y_pred, y_true):
    '''Convert from AROI to Duke iAMD semantic classes.
    Duke iAMD semantic classes:
        {
            "0": "Invalid",
            "51": "Above ILM",
            "102": "ILM-Inner RPEDC",
            "153": "Inner RPEDC-Outer BM",
            "204": "Below BM"
        }
    AROI semantic classes:
        {
            "0": "Above ILM",
            "23": "ILM-IPL/INL",
            "46": "IPL/INL-RPE",
            "69": "RPE-BM",
            "92": "Under BM",
            "115": "Cyst",
            "138": "PED",
            "161": "SRF"
        }
    '''
    mapping = {
        0: 51,
        23: 102,
        46: 102,
        69: 153,
        92: 204,
        # Map to invalid
        115: 0,
        138: 0,
        161: 0,
    }
    y_pred_dukeiamd = np.vectorize(mapping.get)(y_pred)
    y_true_dukeiamd = y_true.copy()
    # Assign invalid class to the pixels whose classes are not in the
    #   Duke iAMD semantic classes and were assigned to the invalid
    #   class.
    y_true_dukeiamd[y_pred_dukeiamd == 0] = 0
    return y_pred_dukeiamd, y_true_dukeiamd


def main():
    args = get_args()
    model_path = Path(args.model_path)
    if args.external is not None:
        results_fn = model_path / f'results_{args.external}.csv'
    elif args.test_alternative is not None:
        results_fn = model_path / f'results_{args.test_alternative}.csv'
    else:
        results_fn = model_path / 'results.csv'
    print('Target results file:', results_fn)
    if not args.overwrite and results_fn.exists():
        print(f'Results already exist in "{model_path}". Use --overwrite to overwrite.')
        exit(0)
    datasets_path = Path(args.datasets_path)
    if args.external is not None:
        preds_path = model_path / f'{args.external}_predictions'
        dataset = args.external
        print(f'Evaluating on external dataset: {args.external}')
        if not (datasets_path / dataset / 'test').exists():
            gt_masks_path = datasets_path / dataset / 'semseg'
        else:
            gt_masks_path = datasets_path / dataset / 'test' / 'semseg'
        source_dataset = model_path.parent.name
    else:
        if args.test_alternative is not None:
            test_name = args.test_alternative
        else:
            test_name = 'test'
        preds_path = model_path / f'{test_name}_predictions'
        dataset = model_path.parent.name
        gt_masks_path = datasets_path / dataset / test_name / 'semseg'
        source_dataset = dataset
    print(f'Evaluating on dataset: {dataset} (source dataset: {source_dataset})')

    if dataset.startswith('Duke_iAMD') and source_dataset.startswith('AROI'):
        print('Using translator from AROI to Duke iAMD.')
        translator = translate_to_dukeiamd_from_aroi
    else:
        translator = lambda x, y: (x, y)

    if not preds_path.exists():
        raise ValueError(f'Path "{preds_path}" does not exist.')

    if not gt_masks_path.exists():
        raise ValueError(f'Path "{gt_masks_path}" does not exist.')

    with open(datasets_path / dataset / 'INFO.json', 'r') as f:
        info = json.load(f)

    sem_classes = {}
    for _k, v in info.items():
        sem_classes[v['value']] = v['label']

    print('Semantic classes:')
    print(json.dumps(sem_classes, indent=4))

    volumes = {}
    for gt_mask_fn in gt_masks_path.iterdir():
        slice_num = None
        last_underscore = gt_mask_fn.stem.rfind('_')
        if last_underscore == -1:
            scan_id = gt_mask_fn.stem
            slice_num = 0
        else:
            scan_id = gt_mask_fn.stem[:last_underscore]
        if scan_id not in volumes:
            volumes[scan_id] = {}
        if slice_num is None:
            slice_num = int(gt_mask_fn.stem[last_underscore+1:])
        if slice_num not in volumes[scan_id]:
            volumes[scan_id][slice_num] = gt_mask_fn.stem

    # Order slices
    for scan_id, slice_nums in volumes.items():
        volumes[scan_id] = [volumes[scan_id][i] for i in sorted(slice_nums.keys())]

    fg_classes = []
    invalid_classes = []
    for sc in sem_classes:
        if 'invalid' in sem_classes[sc].lower():
            invalid_classes.append(sc)
        elif not (args.ignore_bg and (
            'bg' in sem_classes[sc].lower()
            or 'background' in sem_classes[sc].lower()
            or 'above ilm' in sem_classes[sc].lower()
        )):
            fg_classes.append(sc)

    print('Foreground classes:', fg_classes)
    print('Invalid classes:', invalid_classes)

    rows = []
    for scan_id, slices in tqdm.tqdm(volumes.items()):
        print(scan_id)
        gt = []
        pred = []
        for slice_id in slices:
            gt_mask_fn = gt_masks_path / (slice_id + '.png')
            gt_mask = io.imread(gt_mask_fn)
            gt.append(gt_mask)
            pred_mask_fn = preds_path / (slice_id + '.png')
            pred_mask = io.imread(pred_mask_fn)
            pred.append(pred_mask)
        gt = np.array(gt)
        pred = np.array(pred)
        # print(f'  gt shape: {gt.shape}, pred shape: {pred.shape}')
        # print(f'  gt unique: {np.unique(gt)}, pred unique: {np.unique(pred)}')
        # exit()
        if gt.shape != pred.shape:
            # pred = resize(pred, gt.shape, order=0, preserve_range=True)
            pred = resize(pred, gt.shape, order=0, preserve_range=True, anti_aliasing=False).astype(gt.dtype)
        pred, gt = translator(pred, gt)
        for sc in invalid_classes:
            pred[gt == sc] = sc
        for sc in fg_classes:
            sc_gt = gt == sc
            sc_pred = pred == sc
            sc_dice = dice_score(sc_pred, sc_gt, args.empty_sets_nan)
            sc_iou = iou_score(sc_pred, sc_gt, args.empty_sets_nan)
            sc_hd95 = volume_hausdorff_distance(sc_pred, sc_gt, 95, args.empty_sets_nan)
            sc_hd952d = volume_hausdorff_distance_2d(sc_pred, sc_gt, 95, args.empty_sets_nan)
            class_name = sem_classes[sc]
            sc_mad = (
                volume_mean_absolute_distance(sc_pred, sc_gt, 'both', args.empty_sets_nan)
                if not is_lesion_class(class_name)
                else np.nan
            )
            rows.append({
                'ID': scan_id,
                'Class': sem_classes[sc],
                'Dice': sc_dice,
                'IoU': sc_iou,
                'HD95': sc_hd95,
                'HD952D': sc_hd952d,
                'MAD': sc_mad,
            })

    results_df = pd.DataFrame(rows)
    print('\nAverage results:')
    print_mean_metrics(results_df)

    # if dataset in ['Duke_DME', 'AROI']:
    if dataset in []:
        # Save separate results for layers and lesions
        lesion_classes = [
            # Duke_DME:
            'Fluid',
            # AROI:
            'Cyst', 'PED', 'SRF'
        ]
        results_layers_df = results_df[~results_df['Class'].isin(lesion_classes)]
        print('\nAverage results (layers):')
        print_mean_metrics(results_layers_df)
        results_lesions_df = results_df[results_df['Class'].isin(lesion_classes)]
        print('\nAverage results (lesions):')
        print_mean_metrics(results_lesions_df)
        suffix = results_fn.stem.replace('results', '')
        with open(model_path / f'results_layers{suffix}.csv', 'w') as f:
            results_layers_df.to_csv(f, index=False)
        with open(model_path / f'results_lesions{suffix}.csv', 'w') as f:
            results_lesions_df.to_csv(f, index=False)
    else:
        # Save all results in a single file
        with open(results_fn, 'w') as f:
            results_df.to_csv(f, index=False)
    print(f'\nResults saved to "{model_path}" path.')



if __name__ == '__main__':
    main()

