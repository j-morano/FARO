from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize



RES_PATH = Path("/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/seg/SOTA_comparison_new/0/")
DATA_PATH = Path("/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Segmentation/")

SAVE_DIR = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/visualization/__plots_feat/seg/qualitative')
SAVE_DIR.mkdir(exist_ok=True)



models = {
    "miragev1-large_convnext_w": "MIRAGE",
    "octcube_convnext_w": "OCTCube",
    "PB-Mv2-Le_convnext_w": "CLOSER",
    "retfound_convnext_w": "RETFound",
    "urfound_convnext_w": "UrFound",
    "visionfm-base_convnext_w": "VisionFM",
}


def load_gt_cached(gt_file, gt_cache):
    if gt_file not in gt_cache:
        gt_cache[gt_file] = io.imread(gt_file)
    return gt_cache[gt_file]


def multi_class_dice(gt, pred, class_values, ignore_index=-1):
    class_values = np.array([v for v in class_values if v != ignore_index])
    gt_bin = (gt[..., None] == class_values)   # [H, W, C]
    pred_bin = (pred[..., None] == class_values)
    intersection = (gt_bin & pred_bin).sum(axis=(0, 1))
    sums = gt_bin.sum(axis=(0, 1)) + pred_bin.sum(axis=(0, 1))
    scores = np.where(sums == 0, np.nan, 2.0 * intersection / (sums + 1e-6))
    return np.nanmean(scores)


mapping_aroi_dukeiamd = {
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


def process_dataset(dataset_path):
    """Processes one dataset directory, returns a list of result dicts."""
    results = []
    gt_cache = {}
    dataset = dataset_path
    info_fn = DATA_PATH / dataset.name / "INFO.json"
    with open(info_fn, "r") as f:
        info = json.load(f)
    mapping = {v["label"]: v["value"] for v in info.values()}

    for model in sorted(dataset.iterdir()):
        if model.name not in models.keys():
            continue
        print(f"Processing {dataset.name}/{model.name}")
        prediction_dirs = []
        predictions_dir = model / "test_all_predictions"
        if not predictions_dir.exists():
            predictions_dir = model / "test_predictions"
        prediction_dirs.append((predictions_dir, dataset.name, mapping))

        for d in sorted(model.iterdir()):
            if d.name == "Duke_iAMD_labeled_nosclera_70_predictions":
                duke_info_fn = DATA_PATH / "Duke_iAMD_labeled_nosclera_70" / "INFO.json"
                with open(duke_info_fn, "r") as f:
                    duke_info = json.load(f)
                duke_mapping = {v["label"]: v["value"] for v in duke_info.values()}
                prediction_dirs.append((d, "Duke_iAMD_labeled_nosclera_70", duke_mapping))

        for predictions_dir, dataset_name, c_mapping in prediction_dirs:
            for pred_file in sorted(predictions_dir.iterdir()):
                gt_file = DATA_PATH / dataset_name / "test" / "semseg" / pred_file.name
                if not gt_file.exists():
                    gt_file = DATA_PATH / dataset_name / "semseg" / pred_file.name
                gt = load_gt_cached(gt_file, gt_cache)

                if np.count_nonzero(gt) < 0.07 * gt.size:
                    continue

                pred = io.imread(pred_file)
                if pred.shape != gt.shape:
                    pred = resize(pred, gt.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                if dataset_name == "Duke_iAMD_labeled_nosclera_70":
                    pred = np.vectorize(mapping_aroi_dukeiamd.get)(pred).astype(np.uint8)

                score = multi_class_dice(gt, pred, class_values=list(c_mapping.values()))
                results.append({
                    "dataset": dataset_name,
                    "model": model.name,
                    "file": pred_file.name,
                    "dice": score,
                })
    return results


def find_best_cases(max_workers=8):
    dataset_paths = sorted(RES_PATH.iterdir())
    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_dataset, d): d for d in dataset_paths}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                dataset_results = future.result()
                all_results.extend(dataset_results)
                print(f"Finished {dataset.name} ({len(dataset_results)} rows)")
            except Exception as e:
                print(f"ERROR processing {dataset.name}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(SAVE_DIR / "results_test_all.csv", index=False)
    print(f"Saved to {SAVE_DIR / 'results_test_all.csv'}")



def _colorize_prediction(gt, pred, bscan, remap=False):
    # Resize pred to match gt if needed
    if pred.shape != gt.shape:
        pred = resize(
            pred,
            gt.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)

    if remap:
        nonzero_counts = np.sum(gt > 0, axis=0)
        min_pixels = 5
        nonzero_cols = np.where(nonzero_counts > min_pixels)[0]
        if len(nonzero_cols) > 0:
            col_start = nonzero_cols[0]
            col_end = nonzero_cols[-1] + 1
            gt = gt[:, col_start:col_end]
            pred = pred[:, col_start:col_end]
            bscan = bscan[:, col_start:col_end]
        pred = np.vectorize(mapping_aroi_dukeiamd.get)(pred).astype(np.uint8)
        pred[pred == 51] = 0
        gt[gt == 51] = 0

    # Allocate rgb AFTER any cropping, using the (possibly new) shape
    H, W = gt.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Make brighter: move all intensities > 0 to the range 100-255.
    mapping = {0: 0}
    for i in range(1, 256):
        mapping[i] = int(100 + (i / 255) * (255 - 100))
    gt = np.vectorize(mapping.get)(gt).astype(np.uint8)
    pred = np.vectorize(mapping.get)(pred).astype(np.uint8)

    gt_bg = (gt == 0)
    pred_bg = (pred == 0)
    correct = (gt == pred) & ~gt_bg  # true positive, correct class
    false_neg = gt_bg == False       # placeholder, refined below
    false_neg = (~gt_bg) & pred_bg   # GT has structure, pred says background
    false_pos = gt_bg & (~pred_bg)   # GT background, pred says structure
    wrong_class = (~gt_bg) & (~pred_bg) & (gt != pred)  # both non-bg, different classes

    # True positives: grayscale value = class index (using gt's class value)
    rgb[correct] = np.stack([gt[correct]] * 3, axis=-1)

    # False negatives (missed structure): red
    rgb[false_neg] = [255, 0, 0]

    # False positives (predicted structure where none exists): cyan
    rgb[false_pos] = [0, 255, 255]

    # Wrong class (structure present in both, but mismatched class): violet
    rgb[wrong_class] = [148, 0, 211]

    return rgb, gt, bscan



def get_best_samples_per_image():
    MY_MODEL = "PB-Mv2-Le_convnext_w"

    results_df = pd.read_csv(SAVE_DIR / "results_test_all.csv")

    # Pivot so each model becomes a column, one row per (dataset, file)
    pivot = results_df.pivot(index=["dataset", "file"], columns="model", values="dice").reset_index()

    model_cols = [c for c in pivot.columns if c not in ["dataset", "file"]]
    other_models = [c for c in model_cols if c != MY_MODEL]

    pivot["best_other"] = pivot[other_models].max(axis=1)
    pivot["gap"] = pivot[MY_MODEL] - pivot["best_other"]

    # Keep only images where CLOSER is the best
    pivot = pivot[pivot[MY_MODEL] > pivot["best_other"]]

    top3_per_dataset = (
        pivot.sort_values("gap", ascending=False)
        .groupby("dataset", observed=True)
        .head(3)
        .sort_values(["dataset", "gap"], ascending=[True, False])
    )

    save_fn = SAVE_DIR / "top3_gap_per_image.csv"
    top3_per_dataset.to_csv(save_fn, index=False)
    print(f"Saved to {save_fn}")
    return top3_per_dataset


def produce_figures():
    top3_fn = SAVE_DIR / "top3_gap_per_image.csv"
    top3_df = pd.read_csv(top3_fn)
    print(top3_df)

    # Load per-model dice scores so we can look up the value for
    # each specific (dataset, file, model) when saving filenames.
    results_df = pd.read_csv(SAVE_DIR / "results_test_all.csv")
    dice_lookup = results_df.set_index(["dataset", "file", "model"])["dice"]

    for _, row in top3_df.iterrows():
        dataset = row["dataset"]
        file_name = row["file"]
        print(f"Producing figure for {dataset}/{file_name}")

        if "Duke_iAMD" in dataset:
            gt_path = DATA_PATH / dataset / "semseg"
            remap = True
        else:
            gt_path = DATA_PATH / dataset / "test" / "semseg"
            remap = False

        gt_file = gt_path / file_name
        assert gt_file.exists(), gt_file
        gt = io.imread(gt_file)

        bscan_fn = gt_file.parent.parent / "bscan" / file_name
        c_save_dir = SAVE_DIR / f"{dataset}_{Path(file_name).stem}"
        c_save_dir.mkdir(exist_ok=True)
        gt_save_fn = c_save_dir / "ground_truth.png"
        bscan = io.imread(bscan_fn)

        gt_saved = False
        for model in models.keys():
            if "Duke_iAMD" in dataset:
                pred_path = RES_PATH / "AROI_nosclera" / model / "Duke_iAMD_labeled_nosclera_70_predictions"
            else:
                pred_path = RES_PATH / dataset / model / "test_predictions"
            pred_file = pred_path / file_name
            assert pred_file.exists(), pred_file
            pred = io.imread(pred_file)
            color_pred, gt_norm, bscan_norm = _colorize_prediction(gt, pred, bscan, remap=remap)
            if not gt_saved:
                io.imsave(gt_save_fn, gt_norm, check_contrast=False)
                io.imsave(c_save_dir / "bscan.png", bscan_norm)
                gt_saved = True

            try:
                dice_val = dice_lookup.loc[(dataset, file_name, model)]
                dice_str = f"{dice_val:.3f}"
            except KeyError:
                dice_str = "NA"

            save_fn = c_save_dir / f"{model}_dice{dice_str}.png"
            io.imsave(save_fn, color_pred, check_contrast=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, help="Function to run")
    args = parser.parse_args()

    globals()[args.function]()
