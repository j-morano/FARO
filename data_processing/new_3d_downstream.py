from pathlib import Path
import argparse
import shutil
from glob import glob
import random
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from collections import OrderedDict

from skimage import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from scipy.io import loadmat
from scipy.ndimage import label



BASE_DATA_PATH = Path('/mnt/Data/')


BASE_PATH_2D = Path('/home/morano/SW/MIRAGE.git/main/_datasets/Classification/')
TGT_BASE_PATH = Path('/home/morano/SW/MIRAGE.git/main/_datasets/Classification_3D/')
TGT_SEG_DIR = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Segmentation')


def convert_to_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to grayscale using luminosity method
        return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype('uint8')
    elif len(image.shape) == 2:
        # Image is already grayscale
        return image
    else:
        raise ValueError("Unsupported image format")



def oimhs():
    path_dir = BASE_DATA_PATH / 'OIMHS_dataset/Images'
    tgt_path = TGT_BASE_PATH / 'OIMHS'
    # Remove all the dirs inside tgt_path
    if tgt_path.exists():
        for child in tgt_path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for split in (BASE_PATH_2D / 'OIMHS').iterdir():
        for cls_name in split.iterdir():
            print(split.name, cls_name.name)
            for img_fn in cls_name.iterdir():
                parts = img_fn.stem.split('_')
                patient_id = parts[0]
                eye_id = parts[1]
                eye_dir = path_dir / f'{eye_id}'
                imgs = list(eye_dir.glob('*.png'))
                assert imgs and len(imgs) > 0
                scan_nums = sorted([int(Path(img_fn).stem) for img_fn in imgs])
                volume = []
                volume_seg = []
                for scan_num in scan_nums:
                    img_fn = eye_dir / f'{scan_num}.png'
                    img = io.imread(img_fn)
                    # seg is right half, MH is red
                    # print(img.shape)
                    bscan = img[:, :img.shape[1] // 2]
                    seg = img[:, img.shape[1] // 2:]
                    bscan = convert_to_grayscale(bscan)
                    volume.append(bscan)
                    volume_seg.append(seg)
                tgt_fn = tgt_path / split.stem / cls_name.stem / f'{patient_id}_{eye_id}.npz'
                print(' ', tgt_fn)
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tgt_fn, vol=np.array(volume), seg=np.array(volume_seg))


def poagd():
    path_dir = BASE_DATA_PATH / 'POAG_detection/glaucoma_detection'
    tgt_base = TGT_BASE_PATH / 'POAGD'
    for split in (BASE_PATH_2D / 'POAGD').iterdir():
        for cls_name in split.iterdir():
            print(split.name, cls_name.name)
            for img_fn in cls_name.iterdir():
                volume_fn = path_dir / f'{img_fn.stem}.npy'
                volume = np.load(volume_fn)
                print(f'Processing {img_fn.stem} with shape {volume.shape}')
                tgt_fn = tgt_base / split.stem / cls_name.stem / f'{img_fn.stem}.npz'
                print(' ', tgt_fn)
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tgt_fn, vol=volume)



def umn():
    print('Processing UMN dataset')
    base_path = BASE_DATA_PATH / 'FoundOPTIMA_Downstream' / 'UMN'
    data_fn = 'UMN_Dataset.mat'
    target_path = TGT_BASE_PATH / 'UMN'
    target_path.mkdir(exist_ok=True)
    # IMPORTANT: the B-scans are not ordered.

    for cls_name in base_path.iterdir():
        if not cls_name.is_dir():
            continue
        dataset = loadmat(cls_name / data_fn)
        print(dataset.keys())
        all_subjects = dataset['AllSubjects']
        manual_fluid_1 = dataset['ManualFluid1']
        manual_fluid_2 = dataset['ManualFluid2']
        for i in range(len(all_subjects[0])):
            patient = str(i).zfill(3)
            print(all_subjects[0][i].shape)
            class_name = cls_name.stem.lower()
            volume = all_subjects[0][i].transpose(2, 0, 1)
            volume_mf1 = manual_fluid_1[0][i].transpose(2, 0, 1)
            volume_mf2 = manual_fluid_2[0][i].transpose(2, 0, 1)
            assert volume.shape == volume_mf1.shape == volume_mf2.shape
            idd = f'{class_name}_{patient}'
            tgt_fn = target_path / class_name / f'{idd}.npz'
            print(' ', tgt_fn)
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tgt_fn, vol=volume, seg1=volume_mf1, seg2=volume_mf2)


def harvardglaucoma():
    base_path = BASE_DATA_PATH / 'FoundOPTIMA_Downstream/Harvard_Glaucoma_Detection/Harvard Glaucoma Detection and Progression with 1000 Samples (Harvard-GDP1000)/Bscan'
    tgt_base = TGT_BASE_PATH / 'Harvard_Glaucoma'
    for split in (BASE_PATH_2D / 'Harvard_Glaucoma').iterdir():
        for cls_name in split.iterdir():
            for img_fn in cls_name.iterdir():
                # print(split.stem, cls_name.stem, img_fn.stem.split('_')[1])
                idd = img_fn.stem.split('_')[1]
                fn = f'data_{idd}.npz'
                volume = np.load(base_path / fn)['bscans']
                save_path = tgt_base / split.stem / cls_name.stem / fn
                print(' ', save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(save_path, vol=volume)


def remove_white_part(img):
    """
    Args:
        img: HxW numpy array, range 0-255
    """
    # Pad image with 1s
    padded = np.pad(img, ((16, 16), (16, 16)), mode='constant', constant_values=255)
    # plt.imshow(padded, cmap='gray')
    # plt.show()
    # Get biggest connected component of pixels with value > 250
    mask = padded > 250
    labeled, num_features = label(mask)
    largest_component = 0
    largest_size = 0
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if size > largest_size:
            largest_size = size
            largest_component = i
    # Put to 0
    padded[labeled == largest_component] = 0
    # Remove padding
    # plt.imshow(padded, cmap='gray')
    # plt.show()
    return padded[16:-16, 16:-16]


def dukes():
    base_path = BASE_DATA_PATH / 'FoundOPTIMA_Downstream' / 'Duke_Srinivasan' / 'Publication_Dataset'
    target_path = TGT_BASE_PATH / 'DukeSrinivasan'
    class_names_map = {
        'NORMAL': 'Normal',
        'AMD': 'AMD',
        'DME': 'DME',
    }
    for volume_fn in base_path.iterdir():
        print(volume_fn)
        for cls_ in class_names_map:
            if cls_ in volume_fn.stem:
                cls_name = class_names_map[cls_]
                break
        c_base_path = (volume_fn / 'TIFFs' / '8bitTIFFs')
        img_fns = c_base_path.glob('*.tif*')
        img_nums = sorted([int(fn.stem) for fn in img_fns])
        volume = []
        for img_num in img_nums:
            img_fn = c_base_path / f'0{img_num}.tif'
            img = io.imread(img_fn)
            img = convert_to_grayscale(img)
            img = remove_white_part(img)
            volume.append(img)
        volume = np.array(volume)
        print(f'Volume shape: {volume.shape}')
        tgt_fn = target_path / cls_name / f'{volume_fn.stem}.npz'
        print(' ', tgt_fn)
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tgt_fn, vol=volume)




def noor():
    base_path = BASE_DATA_PATH / 'FoundOPTIMA_Downstream' / 'Noor_Eye_Hospital' / 'Data'
    target_path = TGT_BASE_PATH / 'NoorEyeHospital'
    for cls_name in base_path.iterdir():
        if cls_name.stem == 'NORMAL':
            cls_name_stem = 'Normal'
        else:
            cls_name_stem = cls_name.stem
        for case in cls_name.iterdir():
            img_fns = list(case.glob('*.TIFF')) + list(case.glob('*.PNG'))
            assert img_fns and len(img_fns) > 0
            nums = []
            for img_fn in img_fns:
                num = int(re.search(r"(\d+)", img_fn.stem).group(1))
                img = io.imread(img_fn)
                img = convert_to_grayscale(img)
                nums.append((num, img))
            nums = sorted(nums, key=lambda x: x[0])
            volume = []
            for num, img in nums:
                volume.append(img)
            volume = np.array(volume)
            print(f'Volume shape: {volume.shape}')
            table = str.maketrans({'(': None, ')': None, ' ': '_'})
            fn = case.stem.translate(table)
            tgt_fn = target_path / cls_name_stem / f'{fn}.npz'
            print(' ', tgt_fn)
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tgt_fn, vol=volume)


def noor_split_like_2d():
    path_2d = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Classification/Noor_Eye_Hospital')
    path_3d = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Classification_3D/NoorEyeHospital')
    tgt_path_3d = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Classification_3D/Noor_Eye_Hospital')
    splits = {}
    for split in path_2d.iterdir():
        if not split.is_dir():
            continue
        if split.stem not in splits:
            splits[split.stem] = {}
        for cls_name in split.iterdir():
            if not cls_name.is_dir():
                continue
            if cls_name.stem not in splits[split.stem]:
                splits[split.stem][cls_name.stem] = set()
            for img_fn in cls_name.iterdir():
                last_underscore_pos = img_fn.stem.rfind('_')
                fid = img_fn.stem[:last_underscore_pos]
                splits[split.stem][cls_name.stem].add(fid)
    print(splits)
    for split in splits:
        for cls_name in splits[split]:
            for fid in splits[split][cls_name]:
                tgt_fn = tgt_path_3d / split / cls_name / f'{fid}.npz'
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path_3d / cls_name / f'{fid}.npz', tgt_fn)



def umnduketonoor():
    # TODO
    pass



def octmanualdelineations():
    import eyepy
    random.seed(256)

    if (TGT_BASE_PATH / 'OCT_MD_MS_SEG').exists():
        for child in (TGT_BASE_PATH / 'OCT_MD_MS_SEG').iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    # https://iacl.ece.jhu.edu/index.php?title=Resources
    path = Path('/mnt/Data/OCT_Manual_Delineations-2018_June_29_b/OCT_Manual_Delineations-2018_June_29')
    # info_fn = path / 'demographics-2018_June_29.csv'
    # info = pd.read_csv(info_fn)
    dataset = {
        'Healthy': {},
        'Multiple_Sclerosis': {}
    }
    # ID,Age,Gender,Eye,Diagnosis
    for img_fn in (path / 'vol').iterdir():
        id_ = img_fn.stem.split('_')[0]
        ev = eyepy.import_heyex_vol(img_fn)
        vol = ev.data
        # plt.imshow(vol[0], cmap='gray'); plt.show()
        seg_fn = path / 'delineation' / f'{img_fn.stem}.mat'
        seg = None
        try:
            seg1 = loadmat(seg_fn)['bd_pts']
            print('seg1', seg1.shape)
            seg = seg1
        except KeyError:
            pass
        try:
            seg2 = loadmat(seg_fn)['control_pts']
            print('seg2', seg2.shape)
            seg = seg2
        except KeyError:
            pass
        if id_.startswith('hc'):
            label = 'Healthy'
        elif id_.startswith('ms'):
            label = 'Multiple_Sclerosis'
        print(f'{id_} {vol.shape}, label: {label}\n')
        dataset[label][id_] = {
            'vol': vol,
            'label': label,
            'seg': seg,
        }
    # split into train/val/test (33/33/33)
    for label in dataset:
        ids = list(dataset[label].keys())
        random.shuffle(ids)
        val_start = int(len(ids) * 0.40)
        test_start = int(len(ids) * 0.70)
        for idx, id_ in enumerate(ids):
            if idx < val_start:
                split = 'train'
            elif idx < test_start:
                split = 'val'
            else:
                split = 'test'
            vol = dataset[label][id_]['vol']
            seg = dataset[label][id_]['seg']
            tgt_fn = TGT_BASE_PATH / 'OCT_MD_MS_SEG' / split / label / f'{id_}.npz'
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            print(' ', tgt_fn)
            np.savez_compressed(tgt_fn, vol=vol, seg=seg)



def banglaoct2025():
    base_path_dir = Path('/mnt/Data/BanglaOCT2025/BanglaOCT2025_Raw_130_Images_Only')
    tgt_path = TGT_BASE_PATH / 'BanglaOCT2025'
    for split in (BASE_PATH_2D / 'BanglaOCT2025').iterdir():
        for cls_name in split.iterdir():
            for img_fn in cls_name.iterdir():
                print(split.stem, cls_name.stem, img_fn.stem)
                parts = img_fn.stem.split('_')
                patient = parts[0] + '_' + parts[1] + '_' + parts[2]
                print(f'Patient: {patient}')
                c_path = base_path_dir / cls_name.stem / patient
                assert c_path.exists()
                # All those that match this pattern: oct_c_[ddd].bmp,
                #   d being a digit
                bscan_fns = sorted(c_path.glob('oct_c_[0-9][0-9][0-9].bmp'))
                print(f'Found {len(bscan_fns)} B-scans for patient {patient}')
                volume = []
                for bscan_fn in bscan_fns:
                    bscan = io.imread(bscan_fn)
                    bscan_gray = convert_to_grayscale(bscan)
                    volume.append(bscan_gray)
                volume = np.array(volume)
                print(f'Volume shape: {volume.shape}')
                save_fn = tgt_path / split.stem / cls_name.stem / f'{patient}.npz'
                print(' ', save_fn)
                save_fn.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(save_fn, vol=volume)


def get_ordered_bscans(dir: Path):
    extensions = ['.bmp', '.png', '.tif', '.tiff', '.jpeg', '.jpg']
    # Add their uppercase versions
    extensions += [ext.upper() for ext in extensions]
    bscans = []
    for img_fn in dir.iterdir():
        if img_fn.suffix in extensions:
            match = re.search(r'(\d+)(?!.*\d)', img_fn.stem)
            if match:
                num = int(match.group(1))
                bscans.append((num, img_fn))
            else:
                raise ValueError
    return sorted(bscans, key=lambda x: x[0])


def get_ordered_bscans_list(bscan_fns: list):
    bscans = []
    for fn in bscan_fns:
        stem = Path(fn).stem
        match = re.search(r'(\d+)(?!.*\d)', stem)
        if match:
            num = int(match.group(1))
            bscans.append((num, fn))
        else:
            raise ValueError
    return sorted(bscans, key=lambda x: x[0])


def octa500():
    path_dir = Path('/mnt/Data/OCTA-500/')
    info = pd.read_excel(path_dir / 'Text labels.xlsx')
    print(info.head())
    random.seed(512)
    # NOTE: Very few cases in total. Merging diseased
    # NORMAL has 160 volumes
    # CNV has 5 volumes
    # DR has 29 volumes
    # AMD has 6 volumes
    # ID    Sex OS/OD  Age Disease
    # 10301   F    OD   12  NORMAL
    if (TGT_BASE_PATH / 'OCTA500').exists():
        for child in (TGT_BASE_PATH / 'OCTA500').iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for split in (BASE_PATH_2D / 'OCTA500').iterdir():
        for cls_name in split.iterdir():
            print(split.name, cls_name.name)
            for img_fn in cls_name.iterdir():
                print('  ', img_fn)
                parts = img_fn.stem.split('_')
                vol_dir = path_dir / 'dataset_3m' / 'OCT' / parts[0]
                assert vol_dir.exists()
                bscans = get_ordered_bscans(vol_dir)
                volume = []
                gt_artery = io.imread(path_dir / 'dataset_3m' / 'Label' / 'GT_Artery' / f'{parts[0]}.bmp')
                gt_vein = io.imread(path_dir / 'dataset_3m' / 'Label' / 'GT_Vein' / f'{parts[0]}.bmp')
                for num, bscan_fn in bscans:
                    print('    ', bscan_fn)
                    bscan = io.imread(bscan_fn)
                    bscan_gray = convert_to_grayscale(bscan)
                    volume.append(bscan_gray)
                volume = np.array(volume)
                print(f'   Volume shape: {volume.shape}')
                tgt_fn = TGT_BASE_PATH / 'OCTA500' / split.stem / cls_name.stem / f'{parts[0]}.npz'
                print(' ', tgt_fn)
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tgt_fn, vol=volume, gt_artery=gt_artery, gt_vein=gt_vein)


def m3ret():
    path_dir = Path('/mnt/Data/M3Ret/')
    # info_fn = path_dir / 'labels_data_split/oct-macular-v4.6.xlsx'
    # info_train = pd.read_excel(info_fn, sheet_name='train-1216')
    # info_val = pd.read_excel(info_fn, sheet_name='val-406')
    # info_test = pd.read_excel(info_fn, sheet_name='test-406')
    if (TGT_BASE_PATH / 'M3Ret').exists():
        for child in (TGT_BASE_PATH / 'M3Ret').iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for split in (BASE_PATH_2D / 'M3Ret').iterdir():
        for cls_name in split.iterdir():
            print(split.name, cls_name.name)
            for img_fn in cls_name.iterdir():
                # print('  ', img_fn)
                parts = img_fn.stem.split('_')
                pat_id = parts[0]
                laterality = parts[1]
                name = parts[2] + '_' + parts[3]
                img_pattern = path_dir / 'OCTImages-Macular-v3.4' / f'{str(pat_id).zfill(10)}*' / laterality / 'study1' / f'{name}*'
                img_fns = glob(str(img_pattern))
                assert img_fns and len(img_fns) > 0
                bscan_fns = get_ordered_bscans_list(img_fns)
                print(f'   Found {len(bscan_fns)} B-scans for patient {pat_id}, laterality {laterality}, name {name}')
                volume = []
                for num, bscan_fn in bscan_fns:
                    # print('    ', bscan_fn)
                    bscan = io.imread(bscan_fn)
                    bscan_gray = convert_to_grayscale(bscan)
                    volume.append(bscan_gray)
                volume = np.array(volume)
                print(f'   Volume shape: {volume.shape}')
                tgt_fn = TGT_BASE_PATH / 'M3Ret' / split.stem / cls_name.stem / f'{pat_id}_{laterality}.npz'
                print(' ', tgt_fn)
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tgt_fn, vol=volume)


def convert_image_list_to_volume(bscan_fns: list):
    volume = []
    for num, bscan_fn in bscan_fns:
        bscan = io.imread(bscan_fn)
        bscan_gray = convert_to_grayscale(bscan)
        volume.append(bscan_gray)
    return np.array(volume)



def olives_baseline():
    path_dir = Path('/mnt/Data/FoundOPTIMA_Downstream/OLIVES/OLIVES/')
    tgt_base = TGT_BASE_PATH / 'OLIVES_baseline'
    if (tgt_base).exists():
        for child in (tgt_base).iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for split in (BASE_PATH_2D / 'OLIVES_baseline').iterdir():
        for cls_name in split.iterdir():
            print(split.name, cls_name.name)
            for img_fn in cls_name.iterdir():
                parts = img_fn.stem.split('_')
                # find pos of last '_'
                final_underscore_pos = img_fn.stem.rfind('_')
                tgt_stem = img_fn.stem[:final_underscore_pos]
                if 'Prime' in img_fn.stem:
                    study = parts[2] + '_' + parts[3]
                    patient = parts[4]
                    visit = parts[5]
                    laterality = parts[6]
                    vol_dir = path_dir / study / patient / visit / laterality
                    pattern = r'^[0-9]+\.(png|tif)$'
                else:
                    # 0249_0019_TREX_DME_GILA_0249GOS_V1_OS_TREXA_000023.png
                    study = parts[2] + ' ' + parts[3]
                    whatever = parts[4]
                    patient = parts[5]
                    visit = parts[6]
                    laterality = parts[7]
                    vol_dir = path_dir / study / whatever / patient / visit / laterality
                    pattern = r'^TREX.*_[0-9]{6}\.(tif|jpeg)$'
                fns = list(vol_dir.glob('*.*'))
                bscan_fns = [fn for fn in fns if re.match(pattern, fn.name)]
                # print(vol_dir)
                # Count files in that dir
                # num_files = len(list(vol_dir.glob('*.*')))
                bscan_fns = get_ordered_bscans_list(bscan_fns)
                # print(bscan_fns)
                print(f'Found {len(bscan_fns)} files for study {study}, patient {patient}, visit {visit}, laterality {laterality}')
                volume = convert_image_list_to_volume(bscan_fns)
                print(f'Volume shape: {volume.shape}')
                tgt_fn = tgt_base / split.stem / cls_name.stem / f'{tgt_stem}.npz'
                print(' ', tgt_fn)
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tgt_fn, vol=volume)


def gamma():
    path_dir = Path('/mnt/Data/GAMMA/Glaucoma_grading/training/multi-modality_images')
    tgt_base = TGT_BASE_PATH / 'GAMMA'
    for split in (BASE_PATH_2D / 'GAMMA').iterdir():
        for cls_name in split.iterdir():
            print(split.name, cls_name.name)
            for img_fn in cls_name.iterdir():
                vol_id = img_fn.stem.split('_')[0]
                print('  ', vol_id)
                vol_dir = path_dir / vol_id / vol_id
                bscan_fns = get_ordered_bscans(vol_dir)
                print(f'   Found {len(bscan_fns)} B-scans for volume {vol_id}')
                volume = convert_image_list_to_volume(bscan_fns)
                print(f'   Volume shape: {volume.shape}')
                tgt_fn = tgt_base / split.stem / cls_name.stem / f'{vol_id}.npz'
                print(' ', tgt_fn)
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tgt_fn, vol=volume)


def ninec():
    path_dir = Path('/mnt/Data/OPTIMA9C_3D/')
    tgt_base = TGT_BASE_PATH / '9C'
    for split in path_dir.iterdir():
        if '__debug' in split.stem:
            continue
        for cls_name in split.iterdir():
            print(split.name, cls_name.name)
            for fsid in cls_name.iterdir():
                print('  ', fsid.stem)
                vol = np.load(fsid / 'oct.npy')
                print(f'   Volume shape: {vol.shape}')
                tgt_fn = tgt_base / split.stem / cls_name.stem / f'{fsid.stem}.npz'
                print(' ', tgt_fn)
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tgt_fn, vol=vol)


def process_single_volume(row, split, tgt_base, tgt_base_2d):
    fsid = row['FileSetId']
    label = row['label']
    print(f'{split} {fsid} {label}')
    vol_fn = Path(row['sample_path'])
    assert vol_fn.exists(), vol_fn
    vol_dcm = pydicom.dcmread(vol_fn)
    vol_dcm.SamplesPerPixel = 1
    vol = vol_dcm.pixel_array
    if vol.dtype == np.uint16 and vol.max() > 255:
        vol = vol / 256
    vol = vol.astype(np.uint8)
    central_bscan = vol[vol.shape[0] // 2]
    tgt_fn = tgt_base / split / label / f'{fsid}.npz'
    tgt_fn_2d = tgt_base_2d / split / label / f'{fsid}.png'
    tgt_fn.parent.mkdir(parents=True, exist_ok=True)
    tgt_fn_2d.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tgt_fn, vol=vol)
    io.imsave(tgt_fn_2d, central_bscan)


def ninec_complete():
    train_fn = '__info/OPTIMA_15abr_all_vendors_central_bscans_nocompression_cnvtypes_filtered_9_classes_nodup.csv'
    test_fn = '__info/OPTIMA_15abr_all_vendors_central_bscans_nocompression_cnvtypes_test_filtered_9_classes_nodup.csv'
    val_fn = '__info/OPTIMA_15abr_all_vendors_central_bscans_nocompression_cnvtypes_val_filtered_9_classes_nodup.csv'
    # Format: ,sample_path,FileSetId,label,label_int,n_frames
    splits_info = {
        'train': pd.read_csv(train_fn),
        'test': pd.read_csv(test_fn),
        'val': pd.read_csv(val_fn),
    }
    tgt_base = TGT_BASE_PATH / '9C_complete'
    tgt_base_2d = BASE_PATH_2D / '9C_complete'
    for split, info in splits_info.items():
        print(f"Starting split: {split}")

        # Prepare a fixed version of the function with base paths pre-set
        worker = partial(
            process_single_volume,
            split=split,
            tgt_base=tgt_base,
            tgt_base_2d=tgt_base_2d
        )

        # Use ProcessPoolExecutor for CPU-heavy tasks like DICOM decompression
        # It defaults to the number of processors on your machine
        with ProcessPoolExecutor() as executor:
            # Convert dataframe rows to a list of dicts for the executor
            rows = [row for _, row in info.iterrows()]
            results = list(executor.map(worker, rows))

        print(f"Finished split: {split}")



def remove_under(path: Path):
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def octave():
    path_dir = Path('/mnt/Data/OCTAVE/nnUNet_raw/Dataset001_OCTAVE/imagesTr/')
    tgt_base = TGT_BASE_PATH / 'OCTAVE'
    remove_under(tgt_base)
    for split in (BASE_PATH_2D / 'OCTAVE').iterdir():
        for cls_name in split.iterdir():
            print(split.name, cls_name.name)
            for img_fn in cls_name.iterdir():
                parts = img_fn.stem.split('_')
                disease = parts[0]
                iid = parts[1]
                vol_fn = path_dir / f'OCTAVE_{disease}_{iid}_0000.tif'
                vol = io.imread(vol_fn)
                label_fn = path_dir.parent / 'labelsTr' / f'OCTAVE_{disease}_{iid}.tif'
                label = io.imread(label_fn)
                print(f'Volume shape: {vol.shape}')
                print(f'Label shape: {label.shape}')
                assert vol.shape == label.shape
                new_vol = np.zeros_like(vol)
                for i in range(vol.shape[0]):
                    new_vol[i] = remove_white_part(vol[i])
                tgt_fn = tgt_base / split.stem / cls_name.stem / f'{disease}_{iid}.npz'
                print(' ', tgt_fn)
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(tgt_fn, vol=new_vol, seg=label)


def pinnacle_crora():
    import utils
    import json

    random.seed(1024)

    conn = utils.get_optima_connection()

    info_fn = '__info/PINNACLE_BSL_cRORAatEnd_byMoritz_with_conversion.csv'
    info = pd.read_csv(info_fn)
    print(info.head())
    # Unnamed: 0,VRCPatId,Position,FileSetId,HRF,thick DLS,SDD,thin DLS,OPL,WEDGE,iRORA,REFR,HYPO,VITELLI,cRORA,last_visit,cnv_conversion_visit,cnv_conversion,last_visit_day,cnv_conversion_day
    # To save: HRF, thick DLS, SDD, thin DLS, OPL, WEDGE, iRORA, REFR, HYPO, VITELLI
    proc_info = {}
    errors = []
    for idx, row in info.iterrows():
        fsid = row['FileSetId']
        label = 'cRORA' if row['cRORA'] == 1 else 'no_cRORA'
        try:
            oct_fn, _ = utils.get_oct_fn_and_metadata(fsid, conn)
        except KeyError:
            print(f'Error processing {fsid}: FileSetId not found in database')
            errors.append(fsid)
            continue
        print(f'{fsid} {label}')
        features = [
            row['HRF'], row['thick DLS'], row['SDD'], row['thin DLS'], row['OPL'],
            row['WEDGE'], row['iRORA'], row['REFR'], row['HYPO'], row['VITELLI']
        ]
        features = np.array(features, dtype=np.uint8)
        patient = row['VRCPatId']
        if patient not in proc_info:
            proc_info[patient] = []
        proc_info[patient].append((fsid, label, features, oct_fn))

    with open('__info/pinnacle_crora_errors.json', 'w') as f:
        json.dump(errors, f, indent=4)

    num_patients = len(proc_info)
    print(f'Number of patients: {num_patients}')
    patients = list(proc_info.keys())
    random.shuffle(patients)
    val_start = int(num_patients * 0.60)
    test_start = int(num_patients * 0.80)
    for idx, patient in enumerate(patients):
        if idx < val_start:
            split = 'train'
        elif idx < test_start:
            split = 'val'
        else:
            split = 'test'
        for fsid, label, features, oct_fn in proc_info[patient]:
            vol_dcm = pydicom.dcmread(oct_fn)
            vol_dcm.SamplesPerPixel = 1
            vol = vol_dcm.pixel_array
            print(vol.shape, vol.dtype, vol.min(), vol.max())
            if vol.dtype == np.uint16 and vol.max() > 255:
                vol = vol / 256
            vol = vol.astype(np.uint8)
            tgt_fn = TGT_BASE_PATH / 'Pinnacle_cRORA' / split / label / f'{fsid}.npz'
            print(' ', tgt_fn)
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tgt_fn, vol=vol, features=features)



def rvo_lesion_seg():
    random.seed(16)
    path_dir = Path('/mnt/Data/RVO-Lesion/Image_Seg/')
    tgt_dir = TGT_SEG_DIR / 'RVO_Lesion'
    fns = (path_dir / 'images').glob('*.jpg')
    splits = {
        'train': set(),
        'test': set(),
    }
    # test_fn = path_dir / 'test.txt'
    for split in splits:
        with open(path_dir / f'{split}.txt', 'r') as f:
            for line in f:
                splits[split].add(line.strip().split('_')[0])
    print(f'Found {len(splits["train"])} training volumes and {len(splits["test"])} test volumes')
    overlap = splits['train'].intersection(splits['test'])
    assert not overlap, f'Overlap between train and test volumes: {overlap}'
    # Create new val dataset using train dataset, len = 30 (same as
    #   test).
    val_split = random.sample(list(splits['train']), 30)
    splits['train'] = splits['train'] - set(val_split)
    splits['val'] = set(val_split)
    overlap = splits['train'].intersection(splits['val'])
    overlap |= splits['test'].intersection(splits['val'])
    assert not overlap, f'Overlap between train and val volumes: {overlap}'
    print(f'Using {len(splits["train"])} training volumes, {len(splits["val"])} validation volumes and {len(splits["test"])} test volumes')
    vol_ids = {}
    for fn in fns:
        vol_id, bscan_id = fn.stem.split('_')
        if vol_id not in vol_ids:
            vol_ids[vol_id] = []
        vol_ids[vol_id].append(bscan_id)
    print(f'Found {len(vol_ids)} volumes')
    for vol_id, bscan_ids in vol_ids.items():
        vol_id_int = int(vol_id)
        print(f'Processing volume {vol_id} with {len(bscan_ids)} B-scans')
        for bscan_id in sorted(bscan_ids, key=lambda x: int(x)):
            print(f'  {bscan_id}')
            bscan_id_int = int(bscan_id)
            fn_stem = f'{vol_id_int:04d}_{bscan_id_int:03d}.png'
            img = io.imread(path_dir / 'images' / f'{vol_id}_{bscan_id}.jpg')
            img = convert_to_grayscale(img)
            seg = io.imread(path_dir / 'masks' / f'{vol_id}_{bscan_id}.png')
            print(f'    Image shape: {img.shape}, max/min: {img.max()}/{img.min()}')
            print(f'    Segmentation shape: {seg.shape}, unique: {np.unique(seg)}')
            # 1: SRF, 2: IRF
            new_seg = np.zeros_like(seg)
            new_seg[seg == 1] = 127  # SRF
            new_seg[seg == 2] = 255  # IRF
            print(f'    New segmentation shape: {new_seg.shape}, unique: {np.unique(new_seg)}')
            # plt.imshow(img, cmap='gray')
            # plt.imshow(new_seg, alpha=0.5)
            # plt.show()
            split_name = None
            for split, vol_ids in splits.items():
                if vol_id in vol_ids:
                    split_name = split
            assert split_name is not None
            img_fn = tgt_dir / split_name / 'bscan' / fn_stem
            seg_fn = tgt_dir / split_name / 'semseg' / fn_stem
            img_fn.parent.mkdir(parents=True, exist_ok=True)
            seg_fn.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(img_fn, img)
            io.imsave(seg_fn, new_seg)
    info = {
        "0": {
            "value": 0,
            "label": "Background"
        },
        "1": {
            "value": 127,
            "label": "SRF",
        },
        "2": {
            "value": 255,
            "label": "IRF"
        },
    }
    with open(tgt_dir / 'INFO.json', 'w') as f:
        json.dump(info, f, indent=4)


def octave_seg():
    data_dir = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Classification_3D/OCTAVE')
    tgt_dir = TGT_SEG_DIR / 'OCTAVE'
    M = 19
    # From /mnt/Data/OCTAVE/nnUNet_raw/Dataset001_OCTAVE
    # Retinal Pigment Epithelium (RPE)
    # Retina (RET)
    # Choroid and Sclera (CHO)
    # Vitreous (VIT)
    # Subretinal Material (SRM)
    # Posterior Hyaloid Membrane (HYA)
    # Retrohyaloid Space (RHS)
    # Epiretinal Membrane (ERM)
    # Sub-ERM Space (SES)
    # Hypertransmission Defect (HTD)
    # Artifact (ART)
    # Retinal/Subretinal Fluid (FLU)
    # Hyperreflective Material (HRM)
    labels = OrderedDict({
        "Background": (0, 0),
        "CHO": (7, 0),
        "VIT": (8, 0),
        "ART": (11, 0),
        "SRM": (1, 1 * M),
        "HRM": (2, 2 * M),
        "FLU": (3, 3 * M),
        "HTD": (4, 4 * M),
        "RPE": (5, 5 * M),
        "RET": (6, 6 * M),
        "HYA": (9, 9 * M),
        "RHS": (10, 10 * M),
        "ERM": (12, 12 * M),
        "SES": (13, 13 * M),
    })
    info = {}
    ingore = {"CHO", "ART", "VIT"}
    i = 0
    for label_name, (label_value, new_value) in labels.items():
        if label_name in ingore:
            continue
        info[str(i)] = {
            "value": new_value,
            "label": label_name,
        }
        i += 1
    print(json.dumps(info, indent=4))
    for split in (data_dir).iterdir():
        for cls_name in split.iterdir():
            print(split.name, cls_name.name)
            for vol_fn in cls_name.iterdir():
                data = np.load(vol_fn)
                vol = data['vol']
                seg = data['seg']
                print(f'Processing {vol_fn.stem} with volume shape {vol.shape} and segmentation shape {seg.shape}')
                print(vol.dtype, vol.min(), vol.max())
                print(np.unique(seg))
                for label_name, (label_value, new_value) in labels.items():
                    seg[seg == label_value] = new_value
                seg = seg.astype(np.uint8)
                print(f'After remapping, unique segmentation values: {np.unique(seg)}')
                for i in range(vol.shape[0]):
                    img_fn = tgt_dir / split.stem /  'bscan' / f'{vol_fn.stem}_{i:03d}.png'
                    seg_fn = tgt_dir / split.stem /  'semseg' / f'{vol_fn.stem}_{i:03d}.png'
                    img_fn.parent.mkdir(parents=True, exist_ok=True)
                    seg_fn.parent.mkdir(parents=True, exist_ok=True)
                    io.imsave(img_fn, vol[i])
                    io.imsave(seg_fn, seg[i])
    with open(tgt_dir / 'INFO.json', 'w') as f:
        json.dump(info, f, indent=4)


def umn_seg():
    random.seed(32)
    path_dir = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Classification_3D/UMN')
    scans = {}
    info = {
        "0": {
            "value": 0,
            "label": "Background"
        },
        "1": {
            "value": 255,
            "label": "Fluid/cyst",
        },
    }
    for cls_name in path_dir.iterdir():
        scans[cls_name.name] = cls_name.iterdir()
    # Split using 60 20 20
    splits = {
        'train': set(),
        'val': set(),
        'test': set(),
    }
    for cls_name, scan_fns in scans.items():
        scan_fns = list(scan_fns)
        random.shuffle(scan_fns)
        num_scans = len(scan_fns)
        val_start = int(num_scans * 0.60)
        test_start = int(num_scans * 0.80)
        for idx, scan_fn in enumerate(scan_fns):
            if idx < val_start:
                splits['train'].add(scan_fn.stem)
            elif idx < test_start:
                splits['val'].add(scan_fn.stem)
            else:
                splits['test'].add(scan_fn.stem)
    for split, scan_fns in splits.items():
        print(f'{split}: {len(scan_fns)} scans')
    tgt_dir = TGT_SEG_DIR / 'UMN'
    for cls_name in path_dir.iterdir():
        print(cls_name.name)
        for vol_fn in cls_name.iterdir():
            data = np.load(vol_fn)
            vol = data['vol']
            seg1 = data['seg1']
            seg1[seg1 == 1] = 255
            seg2 = data['seg2']
            seg2[seg2 == 1] = 255
            print(f'Processing {vol_fn.stem} with volume shape {vol.shape} and segmentation shape {seg1.shape}')
            print('  ', vol.dtype, vol.min(), vol.max())
            print('  ', np.unique(seg1))
            print('  ', np.unique(seg2))
            # plt.imshow(vol[0], cmap='gray')
            # plt.imshow(seg1[0], alpha=0.5)
            # plt.imshow(seg2[0], alpha=0.5)
            # plt.show()
            c_split = None
            for split, scan_fns in splits.items():
                if vol_fn.stem in scan_fns:
                    c_split = split
            assert c_split is not None
            print(f'   Assigned to split: {c_split}')
            for i in range(vol.shape[0]):
                c_seg1 = seg1[i]
                c_seg2 = seg2[i]
                assert len(np.unique(c_seg1)) <= 2 and len(np.unique(c_seg2)) <= 2
                # assert len(np.unique(c_seg1)) == len(np.unique(c_seg2))
                if len(np.unique(c_seg1)) == 1 or len(np.unique(c_seg2)) == 1:
                    # Only background, skip
                    print(f'    Skipping B-scan {i} because it contains only background')
                    continue
                name = f'{vol_fn.stem}_{i:03d}.png'
                img_fn = tgt_dir / c_split / 'bscan' / name
                seg1_fn = tgt_dir / c_split / 'seg1' / name
                seg2_fn = tgt_dir / c_split / 'seg2' / name
                img_fn.parent.mkdir(parents=True, exist_ok=True)
                seg1_fn.parent.mkdir(parents=True, exist_ok=True)
                seg2_fn.parent.mkdir(parents=True, exist_ok=True)
                io.imsave(img_fn, vol[i])
                io.imsave(seg1_fn, seg1[i])
                io.imsave(seg2_fn, seg2[i])
    with open(tgt_dir / 'INFO.json', 'w') as f:
        json.dump(info, f, indent=4)


def convert_retouch_seg(seg):
    mapping = {
        1: 85,
        2: 170,
        3: 255,
    }
    for k, v in mapping.items():
        seg[seg == k] = v
    seg = seg.astype(np.uint8)
    return seg


def retouch_test_seg():
    skip_empty = False
    path_dir = Path('/optima/exchange/RETOUCH/')
    tgt_dir = TGT_SEG_DIR / 'RETOUCH' / 'test_all'
    for dir in (path_dir / 'TestSet-MUW').iterdir():
        print(dir)
        # oct.mhd
        # oct.raw
        # reference.mhd
        # reference.raw
        oct_fn = dir / 'oct.mhd'
        # oct_raw_fn = dir / 'oct.raw'
        seg_fn = dir / 'reference.mhd'
        # seg_raw_fn = dir / 'reference.raw'
        vol = io.imread(oct_fn)
        seg = io.imread(seg_fn)
        seg_2_fn = Path(str(seg_fn).replace('TestSet-MUW', 'TestSet-Radboud'))
        seg_2 = io.imread(seg_2_fn)
        seg = convert_retouch_seg(seg)
        seg_2 = convert_retouch_seg(seg_2)
        print('   vol  ', vol.shape, vol.dtype, vol.min(), vol.max())
        print('   seg  ', seg.shape, seg.dtype, seg.min(), seg.max())
        print('   seg_2', seg_2.shape, seg_2.dtype, seg_2.min(), seg_2.max())
        for i in range(vol.shape[0]):
            if skip_empty and (len(np.unique(seg[i])) <= 1 or len(np.unique(seg_2[i])) <= 1):
                # Only background, skip
                print(f'    Skipping B-scan {i} because it contains only background in one of the segmentations')
                continue
            c_fn = f'{dir.stem}_{i:03d}.png'
            img_fn = tgt_dir / 'bscan' / c_fn
            seg_fn = tgt_dir / 'semseg' / c_fn
            seg_2_fn = tgt_dir / 'semseg_Radboud' / c_fn
            img_fn.parent.mkdir(parents=True, exist_ok=True)
            seg_fn.parent.mkdir(parents=True, exist_ok=True)
            seg_2_fn.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(img_fn, vol[i])
            io.imsave(seg_fn, seg[i])
            io.imsave(seg_2_fn, seg_2[i])
    # NOTE: INFO is already in the dataset



def amd_dme_cls_seg():
    random.seed(64)
    path_dir = Path('/mnt/Data/AMD_DME_3D_Dataset')
    base = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets')
    cls_2d_tgt_dir = base / 'Classification' / 'AMD_DME_3D'
    cls_3d_tgt_dir = base / 'Classification_3D' / 'AMD_DME_3D'
    seg_tgt_dir = base / 'Segmentation' / 'AMD_DME_3D'

    paths = {
        'AMD': path_dir / 'AMD_labeled',
        'DME': path_dir / 'DME_labeled',
    }
    '''Seg info:
        AMD: 255: PED
        DME: 255: IRF -> 127
    '''
    # First split files into train/val/test using 50/25/25 split at volume level
    fns = {}
    for cls_name in paths:
        c_path = paths[cls_name]
        assert isinstance(c_path, Path)
        for vol_fn in (c_path / 'input').iterdir():
            stem = vol_fn.stem[:5]
            # At the case level
            if cls_name not in fns:
                fns[cls_name] = set()
            fns[cls_name].add(stem)
    print(f'Found {len(fns["AMD"])} AMD cases and {len(fns["DME"])} DME cases')
    splits = {
        'train': set(),
        'val': set(),
        'test': set(),
    }
    for cls_name in fns:
        vol_fns = list(fns[cls_name])
        random.shuffle(vol_fns)
        num_vols = len(vol_fns)
        val_start = int(num_vols * 0.50)
        test_start = int(num_vols * 0.75)
        for idx, vol_fn in enumerate(vol_fns):
            if idx < val_start:
                splits['train'].add(vol_fn)
            elif idx < test_start:
                splits['val'].add(vol_fn)
            else:
                splits['test'].add(vol_fn)
    print(f'Train split: {len(splits["train"])} cases')
    print(f'Val split: {len(splits["val"])} cases')
    print(f'Test split: {len(splits["test"])} cases')

    for cls_name in paths:
        c_path = paths[cls_name]
        assert isinstance(c_path, Path)
        for vol_fn in (c_path / 'input').iterdir():
            print(cls_name, vol_fn)
            seg_fn =  c_path / 'label' / f'{vol_fn.stem}_mask.tif'
            assert seg_fn.exists(), seg_fn
            vol = io.imread(vol_fn)
            seg = io.imread(seg_fn)
            print(f'   Volume: {vol.shape}, {vol.dtype}, min/max: {vol.min()}/{vol.max()}')
            print(f'   Segmentation: {seg.shape}, {seg.dtype}, unique values: {np.unique(seg)}')
            split_name = None
            for split, sample_stems in splits.items():
                for sample_stem in sample_stems:
                    if vol_fn.stem.startswith(sample_stem):
                        split_name = split
                        break
                if split_name is not None:
                    break
            assert split_name is not None, f'Could not find split for volume {vol_fn.stem}'
            print(f'   Assigned to split: {split_name}')
            middle = vol.shape[0] // 2
            central_bscan = vol[middle]
            # plt.imshow(central_bscan, cmap='gray')
            # plt.imshow(seg[middle], alpha=0.5)
            # plt.show()
            if cls_name == 'DME':
                seg[seg == 255] = 127
            cls_2d_fn = cls_2d_tgt_dir / split_name / cls_name / f'{vol_fn.stem}.png'
            cls_3d_fn = cls_3d_tgt_dir / split_name / cls_name / f'{vol_fn.stem}.npz'
            cls_2d_fn.parent.mkdir(parents=True, exist_ok=True)
            cls_3d_fn.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(cls_2d_fn, central_bscan)
            np.savez_compressed(cls_3d_fn, vol=vol, seg=seg)
            for i in range(vol.shape[0]):
                if len(np.unique(seg[i])) <= 1:
                    # Only background, skip
                    print(f'    Skipping B-scan {i} because it contains only background')
                    continue
                img_fn = seg_tgt_dir / split_name / 'bscan' / f'{vol_fn.stem}_{i:03d}.png'
                seg_i_fn = seg_tgt_dir / split_name / 'semseg' / f'{vol_fn.stem}_{i:03d}.png'
                img_fn.parent.mkdir(parents=True, exist_ok=True)
                seg_i_fn.parent.mkdir(parents=True, exist_ok=True)
                io.imsave(img_fn, vol[i])
                io.imsave(seg_i_fn, seg[i])



def amd_sd_seg():
    random.seed(64)
    current_tgt_save_dir = TGT_SEG_DIR / 'AMD_SD_vol'
    path_dir = Path('/mnt/Data/AMD-SD/images')
    train_fn = path_dir.parent / 'training.txt'
    val_fn = path_dir.parent / 'validation.txt'
    splits = {}
    for split, fn in [('train', train_fn), ('test', val_fn)]:
        with open(fn, 'r') as f:
            splits[split] = set(Path(line.strip()).stem for line in f)
    print(f'Found {len(splits["train"])} training samples and {len(splits["test"])} test samples')
    # Now get 30% of the training ones and put them in val
    train_samples = list(splits['train'])
    volume_ids = set()
    for train_sample in train_samples:
        volume_id = train_sample.split('_')[0]
        volume_ids.add(volume_id)
    volume_ids = list(volume_ids)
    print(f'Found {len(volume_ids)} unique volume IDs in training samples')
    random.shuffle(volume_ids)
    # val_from_train = set(train_samples[:int(len(train_samples) * 0.30)])
    val_vol_ids = set(volume_ids[:int(len(volume_ids) * 0.30)])
    val_from_train = set()
    for train_sample in train_samples:
        volume_id = train_sample.split('_')[0]
        if volume_id in val_vol_ids:
            val_from_train.add(train_sample)
    splits['train'] = splits['train'] - val_from_train
    splits['val'] = val_from_train
    print(
        f'After split, {len(splits["train"])} training samples,'
        f' {len(splits["val"])} validation samples'
        f' and {len(splits["test"])} test samples'
    )
    # Assert all of them are disjoint
    overlap = splits['train'].intersection(splits['val'])
    overlap |= splits['train'].intersection(splits['test'])
    overlap |= splits['val'].intersection(splits['test'])
    assert not overlap, f'Overlap between splits: {overlap}'

    mapping = {
        # SRF: red
        # IRF: green
        # PED: blue
        # IS/OS: magenta
        # SHRM: yellow
        (255, 0, 0): 50,  # SRF
        (0, 255, 0): 100,  # IRF
        (0, 0, 255): 150,  # PED
        (255, 0, 255): 200,  # IS/OS
        (255, 255, 0): 250,  # SHRM
        (255, 255, 255): 0,  # Weird borders after thresholding
    }
    for vol_fn in path_dir.iterdir():
        for bs_fn in vol_fn.iterdir():
            print(bs_fn)
            stem = bs_fn.stem
            split_name = None
            for split, sample_stems in splits.items():
                if stem in sample_stems:
                    split_name = split
            assert split_name is not None, f'Could not find split for sample {stem}'
            img = io.imread(bs_fn)
            half_width = img.shape[1] // 2
            bscan = img[:, :half_width]
            bscan = convert_to_grayscale(bscan)
            seg = img[:, half_width:]
            # Binarize seg
            seg[seg > 127] = 255
            seg[seg <= 127] = 0
            new_seg = np.zeros(seg.shape[:2], dtype=np.uint8)
            for color, value in mapping.items():
                mask = np.all(seg == color, axis=-1)
                new_seg[mask] = value
            assert set(np.unique(new_seg)).issubset(set(mapping.values()))
            # plt.imshow(bscan, cmap='gray')
            # plt.imshow(new_seg, alpha=0.5)
            # plt.show()
            vol_id, bscan_id = stem.split('_')
            save_name = f'{int(vol_id):04d}_{int(bscan_id):03d}.png'
            img_fn = current_tgt_save_dir / split_name / 'bscan' / save_name
            seg_fn = current_tgt_save_dir / split_name / 'semseg' / save_name
            img_fn.parent.mkdir(parents=True, exist_ok=True)
            seg_fn.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(img_fn, bscan)
            io.imsave(seg_fn, new_seg)
    info = {
        "0": {
            "value": 0,
            "label": "Background"
        },
        "1": {
            "value": 50,
            "label": "SRF",
        },
        "2": {
            "value": 100,
            "label": "IRF",
        },
        "3": {
            "value": 150,
            "label": "PED",
        },
        "4": {
            "value": 200,
            "label": "IS/OS",
        },
        "5": {
            "value": 250,
            "label": "SHRM",
        },
    }
    with open(current_tgt_save_dir / 'INFO.json', 'w') as f:
        json.dump(info, f, indent=4)
    with open(current_tgt_save_dir / 'IMPORTANT.txt', 'w') as f:
        f.write(
            'IMPORTANT: The splits in this dataset are at the volume level.'
            '\nThe previous version of this dataset had splits at the B-scan'
            ' level, which is not correct.'
        )


# TODO: IMPORTANT: ADD NEW FUNCTIONS FOR AMD_DME_3D, RETOUCH, AND UMN
#   FOR USING ALL THE B-SCANS IN THE TEST SET, SINCE THE EVALUATIONS
#   ARE AT THE VOLUME LEVEL.
def amd_dme_cls_seg_test_all():
    prev_dir = TGT_SEG_DIR / 'AMD_DME_3D'
    test_vol_ids = set()
    for fn in (prev_dir / 'test' / 'bscan').iterdir():
        test_vol_ids.add(fn.stem.split('_')[0])
    print(f'Found {len(test_vol_ids)} unique test volume IDs')

    path_dir = Path('/mnt/Data/AMD_DME_3D_Dataset')
    base = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets')
    seg_tgt_dir = base / 'Segmentation' / 'AMD_DME_3D' / 'test_all'

    paths = {
        'AMD': path_dir / 'AMD_labeled',
        'DME': path_dir / 'DME_labeled',
    }
    '''Seg info:
        AMD: 255: PED
        DME: 255: IRF -> 127
    '''
    # First split files into train/val/test using 50/25/25 split at volume level
    for cls_name in paths:
        c_path = paths[cls_name]
        assert isinstance(c_path, Path)
        for vol_fn in (c_path / 'input').iterdir():
            vol_id = vol_fn.stem
            if vol_id not in test_vol_ids:
                continue
            print(cls_name, vol_fn)
            seg_fn =  c_path / 'label' / f'{vol_fn.stem}_mask.tif'
            assert seg_fn.exists(), seg_fn
            vol = io.imread(vol_fn)
            seg = io.imread(seg_fn)
            print(f'   Volume: {vol.shape}, {vol.dtype}, min/max: {vol.min()}/{vol.max()}')
            print(f'   Segmentation: {seg.shape}, {seg.dtype}, unique values: {np.unique(seg)}')
            # plt.imshow(central_bscan, cmap='gray')
            # plt.imshow(seg[middle], alpha=0.5)
            # plt.show()
            if cls_name == 'DME':
                seg[seg == 255] = 127
            for i in range(vol.shape[0]):
                img_fn = seg_tgt_dir / 'bscan' / f'{vol_fn.stem}_{i:03d}.png'
                seg_i_fn = seg_tgt_dir / 'semseg' / f'{vol_fn.stem}_{i:03d}.png'
                img_fn.parent.mkdir(parents=True, exist_ok=True)
                seg_i_fn.parent.mkdir(parents=True, exist_ok=True)
                io.imsave(img_fn, vol[i])
                io.imsave(seg_i_fn, seg[i])


def umn_seg_test_all():
    prev_dir = TGT_SEG_DIR / 'UMN'
    test_vol_ids = set()
    for fn in (prev_dir / 'test' / 'bscan').iterdir():
        last_underscore_idx = fn.stem.rfind('_')
        vol_id = fn.stem[:last_underscore_idx]
        test_vol_ids.add(vol_id)

    print(f'Found {len(test_vol_ids)} unique test volume IDs')
    path_dir = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Classification_3D/UMN')
    # Split using 60 20 20
    tgt_dir = TGT_SEG_DIR / 'UMN' / 'test_all'
    for cls_name in path_dir.iterdir():
        print(cls_name.name)
        for vol_fn in cls_name.iterdir():
            vol_id = vol_fn.stem
            if vol_id not in test_vol_ids:
                # print(f'Skipping volume {vol_id} because it is not in the test set')
                continue
            print(f'Processing volume {vol_id}')
            data = np.load(vol_fn)
            vol = data['vol']
            seg1 = data['seg1']
            seg1[seg1 == 1] = 255
            seg2 = data['seg2']
            seg2[seg2 == 1] = 255
            print(f'Processing {vol_fn.stem} with volume shape {vol.shape} and segmentation shape {seg1.shape}')
            print('  ', vol.dtype, vol.min(), vol.max())
            print('  ', np.unique(seg1))
            print('  ', np.unique(seg2))
            for i in range(vol.shape[0]):
                c_seg1 = seg1[i]
                c_seg2 = seg2[i]
                assert len(np.unique(c_seg1)) <= 2 and len(np.unique(c_seg2)) <= 2
                # assert len(np.unique(c_seg1)) == len(np.unique(c_seg2))
                name = f'{vol_fn.stem}_{i:03d}.png'
                img_fn = tgt_dir / 'bscan' / name
                seg1_fn = tgt_dir / 'semseg' / name
                seg2_fn = tgt_dir / 'semseg_2' / name
                img_fn.parent.mkdir(parents=True, exist_ok=True)
                seg1_fn.parent.mkdir(parents=True, exist_ok=True)
                seg2_fn.parent.mkdir(parents=True, exist_ok=True)
                io.imsave(img_fn, vol[i])
                io.imsave(seg1_fn, seg1[i])
                io.imsave(seg2_fn, seg2[i])



COLOR_TO_GRAY_OIMHS = {
    (0, 0, 0): 0,        # black: Background
    (0, 255, 0): 63,     # green: Retina
    (255, 255, 0): 126,  # yellow: Choroid
    (255, 0, 0): 189,    # red: Macular hole
    (0, 0, 255): 252,    # blue: IRF
}

def color_to_grayscale_oimhs(vol_rgb):
    """
    vol_rgb: [S, H, W, 3] uint8 array, each pixel one of the colors above.
    Returns: [S, H, W] uint8 grayscale volume.
    """
    S, H, W, _ = vol_rgb.shape
    gray = np.zeros((S, H, W), dtype=np.uint8)

    for rgb, gray_val in COLOR_TO_GRAY_OIMHS.items():
        mask = np.all(vol_rgb == np.array(rgb), axis=-1)  # [S, H, W]
        gray[mask] = gray_val

    return gray

def oimhs_seg():
    path = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Classification_3D/OIMHS')
    base_tgt_seg_dir = TGT_SEG_DIR / 'OIMHS'
    base_tgt_seg_dir.mkdir(exist_ok=True)
    with open(base_tgt_seg_dir / 'INFO.json', 'w') as f:
        json.dump({
            "0": {"value": 0, "label": "Background"},
            "1": {"value": 63, "label": "Retina"},
            "2": {"value": 126, "label": "Choroid"},
            "3": {"value": 189, "label": "Macular hole"},
            "4": {"value": 252, "label": "IRF"},
        }, f, indent=4)
    for split in path.iterdir():
        for cls_name in split.iterdir():
            for vol_fn in cls_name.iterdir():
                data = np.load(vol_fn)
                vol = data['vol']
                seg = data['seg']
                print(f'Processing {vol_fn.stem} with volume shape {vol.shape} and segmentation shape {seg.shape}')
                print('  ', vol.dtype, vol.min(), vol.max())
                print('  ', np.unique(seg))
                tgt_dir = base_tgt_seg_dir / split.stem
                cls_name_disp = cls_name.stem.replace('MacularHole', 'MH')
                print(seg.dtype, seg.shape, seg.min(), seg.max())
                segg = color_to_grayscale_oimhs(seg)
                print('  segg', segg.dtype, segg.shape, segg.min(), segg.max())
                for i in range(vol.shape[0]):
                    img_fn = tgt_dir / 'bscan' / f'{cls_name_disp}_{vol_fn.stem}_{i:03d}.png'
                    seg_fn = tgt_dir / 'semseg' / f'{cls_name_disp}_{vol_fn.stem}_{i:03d}.png'
                    # Translate into grayscale
                    img_fn.parent.mkdir(parents=True, exist_ok=True)
                    seg_fn.parent.mkdir(parents=True, exist_ok=True)
                    io.imsave(img_fn, vol[i])
                    io.imsave(seg_fn, segg[i])


def oct_md_ms_seg():
    path = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Classification_3D/OCT_MD_MS/')
    for split in sorted(path.iterdir()):
        for class_name in sorted(split.iterdir()):
            for fn in sorted(class_name.iterdir()):
                print(fn)
                data = np.load(fn, allow_pickle=True)
                # print(data.keys())
                oct = data['vol']
                seg = data['seg']
                print(seg.shape)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str, help='Function to execute')
    args = parser.parse_args()

    globals()[args.function]()
