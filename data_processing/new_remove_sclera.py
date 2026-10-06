from pathlib import Path
import json
import shutil

from skimage import io
from matplotlib import pyplot as plt


stage = 1

if stage == 0:

    BASE_PATH = Path("/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Segmentation")

    SCLERA_NAMES = {
        "AROI": "Under BM",
        "Duke_DME": "BM",
        "Duke_iAMD_labeled": "Below BM",
    }
    NEW_VALUES = {
        "AROI": 0,
        "Duke_DME": 0,
        "Duke_iAMD_labeled": 51,
    }
    '''INFO.json format:
    {
        "0": {
            "value": 0,
            "label": "Background"
        },
        ...
    }
    '''

    for dataset in SCLERA_NAMES.keys():
        print('-'*70)
        dataset_path = BASE_PATH / dataset
        new_dataset_path = BASE_PATH / f"{dataset}_nosclera"
        info_fn = dataset_path / "INFO.json"
        with open(info_fn, "r") as f:
            info = json.load(f)
        print(f"INFO.json for {dataset}: {info}")
        new_info = {}
        value_to_remove = None
        for key, value in info.items():
            if value["label"] == SCLERA_NAMES[dataset]:
                print(f"Removing sclera label {value['label']} from {dataset}")
                value_to_remove = value["value"]
                continue
            new_info[key] = value
        assert value_to_remove is not None
        new_info_fn = new_dataset_path / "INFO.json"
        print("New info:", new_info)
        print("Value to remove:", value_to_remove)

        split_list = ["train", "val", "test"]
        if dataset == "Duke_iAMD_labeled":
            split_list = [""]
        for split in split_list:
            split_path = dataset_path / split
            assert split_path.exists()
            split_path.mkdir(parents=True, exist_ok=True)
            tgt_bscan_path = new_dataset_path / split / "bscan"
            if tgt_bscan_path.exists():
                shutil.rmtree(tgt_bscan_path)
            shutil.copytree(split_path / "bscan", tgt_bscan_path)
            for semseg_fn in sorted((split_path / "semseg").iterdir()):
                print(f"Processing {semseg_fn}...")
                semseg = io.imread(semseg_fn)
                new_semseg = semseg.copy()
                new_semseg[semseg == value_to_remove] = NEW_VALUES[dataset]
                # Debug
                # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                # ax[0].imshow(semseg)
                # ax[0].set_title("Original")
                # ax[1].imshow(new_semseg)
                # ax[1].set_title("Sclera removed")
                # plt.show()
                tgt_path = new_dataset_path / split / "semseg" / semseg_fn.name
                tgt_path.parent.mkdir(parents=True, exist_ok=True)
                io.imsave(tgt_path, new_semseg)
        with open(new_info_fn, "w") as f:
            json.dump(new_info, f, indent=4)



elif stage == 1:
    # Remove images from Duke_iAMD_labeled with less than 50% of the
    #   pixels labeled
    threshold = 70
    path_dir = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Segmentation/Duke_iAMD_labeled_nosclera')
    if threshold == 50:
        suff = 'cleaned'
    else:
        suff = f'{threshold}'
    tgt_dir = path_dir.parent / f"Duke_iAMD_labeled_nosclera_{suff}"
    tgt_dir.mkdir(exist_ok=True)
    shutil.copy(path_dir / "INFO.json", tgt_dir / "INFO.json")
    copied = 0
    total = 0
    for semseg_fn in sorted((path_dir / "semseg").iterdir()):
        semseg = io.imread(semseg_fn)
        print(f"Processing {semseg_fn.name}...")
        print(f"    Shape: {semseg.shape}")
        num_pixels = semseg.size
        num_labeled_pixels = (semseg != 0).sum()
        perc_labeled = num_labeled_pixels / num_pixels * 100
        print(f"    Percentage of labeled pixels: {perc_labeled:.2f}%")
        if perc_labeled > threshold:
            tgt_path = tgt_dir / "semseg" / semseg_fn.name
            tgt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(semseg_fn, tgt_path)
            # Also copy corresponding b-scan
            bscan_fn = path_dir / "bscan" / semseg_fn.name
            tgt_bscan_path = tgt_dir / "bscan" / bscan_fn.name
            tgt_bscan_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(bscan_fn, tgt_bscan_path)
            copied += 1
        total += 1
    print(f"Copied {copied}/{total} images with more than {threshold}% labeled pixels.")





