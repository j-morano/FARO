"""
Evaluate segmentation at the B-scan level (Dice + HD95 2D) and record
ground-truth lesion area, for lesion-size-stratified analysis.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tqdm
from skimage import io
from skimage.transform import resize
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from mutils.misc import SortingHelpFormatter
from visualization.plot_utils import LESION_CLASSES


def get_args():
    parser = argparse.ArgumentParser(
        'B-scan-level segmentation evaluation with GT lesion area',
        formatter_class=SortingHelpFormatter,
    )
    parser.add_argument('-d', '--datasets_path', type=str, default='./_datasets/Segmentation/')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('--test_name', type=str, default='test')
    parser.add_argument('--empty_sets_nan', action='store_true')
    parser.add_argument('--no_empty_sets_nan', dest='empty_sets_nan', action='store_false')
    parser.set_defaults(empty_sets_nan=True)
    parser.add_argument('--out', type=str, default=None)
    return parser.parse_args()


def is_lesion_class(class_name):
    return class_name in LESION_CLASSES


def dice_score_2d(y_pred, y_true, empty_sets_nan=True):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    sum_true = np.sum(y_true)
    sum_pred = np.sum(y_pred)
    if sum_true == 0 and sum_pred == 0:
        return np.nan if empty_sets_nan else 1.0
    intersection = np.sum(y_true * y_pred)
    return 2.0 * intersection / (sum_true + sum_pred + 1e-6)


def get_edges_2d(mask):
    eroded = binary_erosion(mask)
    return mask ^ eroded


def hausdorff_distance_95_2d(y_pred, y_true, percentile=95):
    edges_pred = get_edges_2d(y_pred)
    edges_true = get_edges_2d(y_true)
    coords_pred = np.argwhere(edges_pred)
    coords_true = np.argwhere(edges_true)
    if len(coords_pred) == 0 or len(coords_true) == 0:
        return np.nan
    tree_true = cKDTree(coords_true)
    tree_pred = cKDTree(coords_pred)
    dist_pred_to_true, _ = tree_true.query(coords_pred)
    dist_true_to_pred, _ = tree_pred.query(coords_true)
    return max(
        np.percentile(dist_pred_to_true, percentile),
        np.percentile(dist_true_to_pred, percentile),
    )


def slice_hausdorff_95(y_pred_slice, y_true_slice, percentile=95, empty_sets_nan=True):
    sum_true = y_true_slice.sum()
    sum_pred = y_pred_slice.sum()
    if sum_true == 0 and sum_pred == 0:
        return np.nan if empty_sets_nan else 0.0
    if sum_true == 0 or sum_pred == 0:
        if empty_sets_nan:
            return np.nan
        return float(np.sqrt(sum(s ** 2 for s in y_true_slice.shape)))
    return hausdorff_distance_95_2d(
        y_pred_slice.astype(bool), y_true_slice.astype(bool), percentile
    )


def main():
    args = get_args()
    datasets_path = Path(args.datasets_path)
    model_path = Path(args.model_path)

    gt_masks_path = datasets_path / args.dataset / args.test_name / 'semseg'
    preds_path = model_path / f'{args.test_name}_predictions'

    if not gt_masks_path.exists():
        raise ValueError(f'Path "{gt_masks_path}" does not exist.')
    if not preds_path.exists():
        raise ValueError(f'Path "{preds_path}" does not exist.')

    with open(datasets_path / args.dataset / 'INFO.json', 'r') as f:
        info = json.load(f)
    sem_classes = {v['value']: v['label'] for v in info.values()}
    lesion_classes = {sc: name for sc, name in sem_classes.items() if is_lesion_class(name)}
    print('Lesion classes:', lesion_classes)

    gt_fns = sorted(gt_masks_path.iterdir())
    rows = []
    for gt_fn in tqdm.tqdm(gt_fns, desc='Evaluating B-scans'):
        slice_id = gt_fn.stem
        gt_mask = io.imread(gt_fn)

        pred_fn = preds_path / (slice_id + '.png')
        if not pred_fn.exists():
            print(f'Warning: no prediction for "{slice_id}", skipping.')
            continue
        pred_mask = io.imread(pred_fn)

        if pred_mask.shape != gt_mask.shape:
            pred_mask = resize(
                pred_mask, gt_mask.shape, order=0,
                preserve_range=True, anti_aliasing=False,
            ).astype(gt_mask.dtype)

        for sc, class_name in lesion_classes.items():
            gt_bin = gt_mask == sc
            pred_bin = pred_mask == sc
            dice = dice_score_2d(pred_bin, gt_bin, args.empty_sets_nan)
            hd952d = slice_hausdorff_95(pred_bin, gt_bin, 95, args.empty_sets_nan)
            gt_area = int(gt_bin.sum())
            rows.append({
                'Slice': slice_id,
                'Class': class_name,
                'Dice': dice,
                'HD952D': hd952d,
                'GT_Area': gt_area,
            })

    results_df = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else model_path / 'bscan_results.csv'
    results_df.to_csv(out_path, index=False)
    print(f'\nSaved {len(results_df)} rows to "{out_path}"')
    print(results_df.groupby('Class')[['Dice', 'HD952D']].agg(['mean', 'std', 'count']))


if __name__ == '__main__':
    main()
