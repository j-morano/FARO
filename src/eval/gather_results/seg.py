from pathlib import Path
from collections import OrderedDict
import argparse

import pandas as pd

from visualization.plot_utils import LESION_CLASSES

BASE_PATH = Path('/home/morano/SW/Documents/My_Papers/MIRAGEv2/v2/MIRAGEv2 (CLOSER) (Version 740)/complete_results/seg')

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

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='sota')
args = parser.parse_args()

if args.experiment == 'sota':
    model_mapping = OrderedDict({
        "visionfm-base_convnext_w": "VisionFM",
        "octcube_convnext_w": "OCTCube",
        "retfound_convnext_w": "RETFound",
        "urfound_convnext_w": "UrFound",
        "miragev1-large_convnext_w": "MIRAGE",
        "PB-Mv2-Le_convnext_w": "CLOSER",
    })

elif args.experiment == 'ablation':
    model_mapping = OrderedDict({
        "ibot-base_linear_w": "iBOT",
        "nonPB-MAE-only_linear_w": "MAE",
        "nonPB-MIRAGE_linear_w": "MIRAGE",
        "PB-Mv2-Le_linear_w": "CLOSER",
    })


def get_split_dataset_name(dataset_name, class_name):
    """Split dataset into '_layers' / '_lesions' based on class name."""
    if class_name in LESION_CLASSES:
        return f"{dataset_name}_lesions"
    return f"{dataset_name}_layers"


def gather_seg_results(dir, metrics):
    """
    Gathers raw per-(ID, Class) rows across all datasets/models, preserving
    full granularity exactly as in the source results*.csv files (ID, Class,
    Dice, IoU, HD95, HD952D, MAD, ...), just tagging each row with Dataset
    (split into _layers/_lesions) and Model.
    """
    all_rows = []

    for dataset in sorted(dir.iterdir()):
        if not dataset.is_dir() or (dataset.name not in INCLUDE_DATASETS and INCLUDE_DATASETS):
            continue
        for model_key, model_label in model_mapping.items():
            model_label_clean = model_label.replace('\n', ' ')
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
                df['SplitDataset'] = df['Class'].apply(
                    lambda c: get_split_dataset_name(dataset_name, c)
                )
                df['Dataset'] = df['SplitDataset']
                df['Model'] = model_label_clean
                df = df.drop(columns=['SplitDataset'])

                all_rows.append(df)

    if not all_rows:
        print("No data found.")
        return None

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df = full_df[full_df['Model'].isin([v.replace('\n', ' ') for v in model_mapping.values()])]

    # Sanity check: same number of rows per model
    for model_label in model_mapping.values():
        clean_name = model_label.replace('\n', ' ')
        count = (full_df['Model'] == clean_name).sum()
        print(f"Model: {clean_name}, Rows: {count}")

    return full_df


if args.experiment == 'ablation':
    print("ABLATION MODELS")
    seg_dir = '/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/seg/Ablation_new_paper_lin/0/'
else:
    print("SOTA MODELS")
    seg_dir = '/mnt/Data/SSHFS/msc_server/MIRAGE_future/eval/__output/seg/SOTA_comparison_new/0/'

seg_dir = Path(seg_dir)

# Gather every metric present in your results CSVs (adjust if you only
# want specific ones, e.g. ['Dice', 'HD95'])
metrics = ['Dice', 'IoU', 'HD95', 'HD952D', 'MAD']

full_df = gather_seg_results(seg_dir, metrics)

if full_df is not None:
    save_fn = BASE_PATH / f"{args.experiment}_raw_seg_results.csv"
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(save_fn, index=False)
    print(f"\nSaved {len(full_df)} rows to {save_fn}")
