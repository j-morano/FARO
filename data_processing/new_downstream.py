from pathlib import Path
import argparse
import shutil
from glob import glob
import random

from skimage import io
import numpy as np
# package for loading .mat
from scipy.io import loadmat
from scipy.ndimage import label
import pandas as pd
import matplotlib.pyplot as plt

from img_utils import remove_white_borders




BASE_DATA_PATH = Path('/mnt/Data/')


TGT_BASE_PATH = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Classification/')


def convert_to_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to grayscale using luminosity method
        return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype('uint8')
    elif len(image.shape) == 2:
        # Image is already grayscale
        return image
    else:
        raise ValueError("Unsupported image format")



def trghs22fpg_4():
    path_dir = BASE_DATA_PATH / 'trghs22fpg-4/Dataset/Dataset/Macula'
    tgt_path = TGT_BASE_PATH / 'trghs22fpg_4'
    for disease in path_dir.iterdir():
        print(f'{disease.name} has {len(list(disease.iterdir()))} patients')
        for patient in disease.iterdir():
            for eye in patient.iterdir():
                try:
                    bscan_fn = glob(str(eye / '*B-scan*.jpg'))[0]
                except IndexError:
                    try:
                        bscan_fn = glob(str(eye / 'original.jpg'))[0]
                    except IndexError:
                        print(f'No B-scan found for {eye}')
                        continue
                bscan = io.imread(bscan_fn)
                bscan_gray = convert_to_grayscale(bscan)
                tgt_fn = f'{disease.name}/{disease.name}_{patient.name}_{eye.name}.jpg'.replace(' ', '_')
                tgt_fn = tgt_path / tgt_fn
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                print(' ', tgt_fn)
                io.imsave(tgt_fn, bscan_gray)


'''NOTE:
For very large datasets like Kermany and C8, the idea for linear probing
is the following: keep the test set as it is, and split the original
validation set into train/val. To ensure that we are not cheating, the
images will be ordered by name, and the first 60% will be for training
and the rest for validation. In addition, we will clean both datasets
removing the white borders.
'''


def process_kermany_img(img_fn, tgt_path, disease, split):
    img = io.imread(img_fn)
    img = convert_to_grayscale(img)
    img = remove_white_borders(img)
    tgt_fn = tgt_path / split / disease.name / img_fn.name
    tgt_fn.parent.mkdir(parents=True, exist_ok=True)
    print(' ', tgt_fn)
    io.imsave(tgt_fn, img)


def process_kermany_like_dataset(tgt_path, path_dir):
    # First process test normally
    for disease in (path_dir / 'test').iterdir():
        for img_fn in disease.iterdir():
            process_kermany_img(img_fn, tgt_path, disease, 'test')
    for disease in (path_dir / 'val').iterdir():
        val_fns = sorted(glob(str(disease / '*.*')))
        split_idx = int(len(val_fns) * 0.6)
        for img_fn in val_fns[:split_idx]:
            img_fn = Path(img_fn)
            process_kermany_img(img_fn, tgt_path, disease, 'train')
        for img_fn in val_fns[split_idx:]:
            img_fn = Path(img_fn)
            process_kermany_img(img_fn, tgt_path, disease, 'val')


def kermany_correct():
    path_dir = BASE_DATA_PATH / 'Kermany_correct'
    tgt_path = TGT_BASE_PATH / 'Kermany_correct'
    process_kermany_like_dataset(tgt_path, path_dir)


def oct_c8():
    path_dir = BASE_DATA_PATH / 'OCT-C8/RetinalOCT_Dataset/RetinalOCT_Dataset'
    tgt_path = TGT_BASE_PATH / 'OCT_C8'
    process_kermany_like_dataset(tgt_path, path_dir)


def oimhs():
    random.seed(32)
    path_dir = BASE_DATA_PATH / 'OIMHS_dataset/Images'
    tgt_path = TGT_BASE_PATH / 'OIMHS'
    # Remove all the dirs inside tgt_path
    if tgt_path.exists():
        for child in tgt_path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    info = pd.read_excel('/home/morano/Bookmarks/Data/OIMHS_dataset/Demographics of the participants.xlsx')
    # Patient ID, Eye ID, Eye category, Age(years), Sex, Stage
    print(info.head())
    patient_ids = info['Patient ID'].unique()
    # Randomly split the patient ids into train/val/test (60/20/20)
    patient_ids = list(patient_ids)
    random.shuffle(patient_ids)
    split_idx_1 = int(len(patient_ids) * 0.5)
    split_idx_2 = int(len(patient_ids) * 0.75)
    train_ids = patient_ids[:split_idx_1]
    val_ids = patient_ids[split_idx_1:split_idx_2]
    test_ids = patient_ids[split_idx_2:]
    print(f'Train IDs: {train_ids}')
    print(f'Val IDs: {val_ids}')
    print(f'Test IDs: {test_ids}')
    # NOTE: stage of macular hole
    # NOTE: Grouping 1+2 and 3+4
    for volume in path_dir.iterdir():
        imgs = glob(str(volume / '*.png'))
        largest_mh = ('', 0)
        for img_fn in imgs:
            img = io.imread(img_fn)
            # seg is right half, MH is red
            # print(img.shape)
            all_seg = img[:, img.shape[1] // 2:]
            # print(all_seg.shape, np.unique(all_seg))
            # Get the MH segmentation, where the R is 255 and the other
            #   channels are 0.
            mh_seg = (all_seg == [255, 0, 0]).all(axis=-1)
            # plt.imshow(mh_seg); plt.show()
            area = mh_seg.sum()
            if area > largest_mh[1]:
                largest_mh = (img_fn, area)
        eye_id = volume.name
        patient_id = info[info['Eye ID'] == int(eye_id)]['Patient ID'].values[0]
        stage = info[info['Eye ID'] == int(eye_id)]['Stage'].values[0]
        if stage in [1, 2, 3]:
            stage = '123'
        elif stage in [4]:
            stage = '4'
        if patient_id in train_ids:
            split = 'train'
        elif patient_id in val_ids:
            split = 'val'
        elif patient_id in test_ids:
            split = 'test'
        else:
            raise ValueError
        img = io.imread(largest_mh[0])
        idx = Path(largest_mh[0]).stem
        width = img.shape[1]
        left_half = img[:, :width // 2]
        tgt_fn = tgt_path / split / f'MacularHole_{stage}' / f'{patient_id}_{eye_id}_{idx}.png'
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        print(' ', tgt_fn)
        io.imsave(tgt_fn, left_half)


def octandeyefundus():
    '''
    DME column specifies whether DME has been diagnosed.
        - 1: DME IS present.
        - 0: DME IS NOT present.
    DR column provides information about DR diagnosis and degree.
        - 0: NO DR present.
        - NPDR: Non-Proliferative Diabetic Retinopathy.
        - PDR: Proliferative Diabetic Retinopathy.
        - '-': unknown classification.
    Name,DME,DR,Size,Format
    1222_OD_o_2,0,0,1408X573,jpg
    For example: '1221_OD_o_3' corresponds to the 3rd sample picture of
        OCT image from the right eyeball of subject number 1221.
    NOTE: create 1 dataset for each label type.
    '''
    random.seed(64)
    path_dir = BASE_DATA_PATH / 'OCT-AND-EYE-FUNDUS-DATASET-main'
    tgt_path_dme = TGT_BASE_PATH / 'OAEFD_DME'
    tgt_path_dr = TGT_BASE_PATH / 'OAEFD_DR'
    info = pd.read_csv(path_dir / 'OCT.csv')
    print(info.head())
    imgs = list((path_dir / 'OCT').glob('*/*.*'))
    print(f'Found {len(imgs)} images')
    patients = set()
    for img_fn in imgs:
        patient = img_fn.stem.split('_')[0]
        patients.add(patient)
    patients = sorted(patients)
    print(f'Found {len(patients)} patients')
    random.shuffle(patients)
    train_idx = int(len(patients) * 0.45)
    val_idx = int(len(patients) * 0.70)

    dme_label_mapping = {0: 'No_DME', 1: 'DME'}
    dr_label_mapping = {'0': 'No_DR', 'NPDR': 'NPDR', 'PDR': 'PDR'}

    def get_imgs_from_patient(patient, all_imgs):
        patient_imgs = []
        for img_fn in all_imgs:
            if img_fn.stem.split('_')[0] == patient:
                patient_imgs.append(img_fn)
        return patient_imgs

    for idx, patient in enumerate(patients):
        if idx < train_idx:
            split = 'train'
        elif idx < val_idx:
            split = 'val'
        else:
            split = 'test'
        patient_imgs = get_imgs_from_patient(patient, imgs)
        print(f'Processing patient {patient} with {len(patient_imgs)} images')
        for img_fn in patient_imgs:
            fsid = img_fn.stem
            label_dme = info[info['Name'] == fsid]['DME'].values[0]
            label_dme_str = dme_label_mapping[label_dme]
            print(f'  Processing {img_fn.stem} with DME label {label_dme}')
            tgt_fn_dme = tgt_path_dme / split / label_dme_str / img_fn.name
            tgt_fn_dme.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_fn, tgt_fn_dme)
            label_dr = info[info['Name'] == fsid]['DR'].values[0]
            try:
                label_dr_str = dr_label_mapping[label_dr]
            except KeyError:
                continue
            tgt_fn_dr = tgt_path_dr / split / label_dr_str / img_fn.name
            tgt_fn_dr.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_fn, tgt_fn_dr)


def poagd():
    random.seed(4)
    path_dir = BASE_DATA_PATH / 'POAG_detection/glaucoma_detection'
    tgt_base = TGT_BASE_PATH / 'POAGD'
    patients = set()
    for img_fn in path_dir.iterdir():
        patient_id = img_fn.stem.split('-')[1]
        patients.add(patient_id)
    patients = sorted(patients)
    print(f'Found {len(patients)} patients')
    random.shuffle(patients)
    train_idx = int(len(patients) * 0.5)
    val_idx = int(len(patients) * 0.75)
    for idx, patient in enumerate(patients):
        if idx < train_idx:
            split = 'train'
        elif idx < val_idx:
            split = 'val'
        else:
            split = 'test'
        img_fns = sorted(path_dir.glob(f'*{patient}*.npy'))
        print(f'Processing patient {patient} with {len(img_fns)} images')
        for img_fn in img_fns:
            volume = np.load(img_fn)
            bscan = volume[volume.shape[0] // 2]
            disease = img_fn.stem.split('-')[0]
            # plt.imshow(bscan, cmap='gray'); plt.show()
            print(f'Processing {img_fn.stem} with shape {volume.shape}')
            tgt_fn = tgt_base / split / disease / img_fn.name.replace('.npy', '.png')
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(tgt_fn, bscan)


def umnduketonoor():
    random.seed(64)
    path_dir = Path('/home/morano/SW/MIRAGE.git/main/_datasets/Classification/UMNandDukeSrinivasan_to_Noor_cross/train')
    tgt_dir = path_dir.parent / 'val'
    for disease in path_dir.iterdir():
        imgs = list(disease.iterdir())
        num_imgs = len(imgs)
        print(f'{disease.name} has {num_imgs} images')
        # Randomly sample 25%
        num_sample = int(num_imgs * 0.33)
        random.shuffle(imgs)
        sample_imgs = imgs[:num_sample]
        for img_fn in sample_imgs:
            shutil.move(img_fn, tgt_dir / disease.name / img_fn.name)


def goals():
    random.seed(128)
    path_dir = BASE_DATA_PATH / 'FoundOPTIMA_Downstream/GOALS/'
    labels = pd.read_csv(path_dir / 'GC_GT.csv', dtype={'ImgName': str, 'GC_Label': int})
    tgt_base = TGT_BASE_PATH / 'GOALS'
    if tgt_base.exists():
        for child in tgt_base.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    print(labels.head())
    # ImgName,GC_Label
    imgs_dir = path_dir / 'Image'
    label_mapping = {0: 'Normal', 1: 'Glaucoma'}
    imgs = sorted(glob(str(imgs_dir / '*.*')))
    random.shuffle(imgs)
    val_start = int(len(imgs) * 0.4)
    test_start = int(len(imgs) * 0.7)
    for idx, img_fn in enumerate(imgs):
        fsid = Path(img_fn).stem
        print(fsid)
        label_int = labels[labels['ImgName'] == fsid]['GC_Label'].values[0]
        label = label_mapping[label_int]
        if idx < val_start:
            split = 'train'
        elif idx < test_start:
            split = 'val'
        else:
            split = 'test'
        tgt_fn = tgt_base / split / label / Path(img_fn).name
        tgt_fn.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_fn, tgt_fn)


def octmanualdelineations():
    import eyepy
    random.seed(256)

    for child in (TGT_BASE_PATH / 'OCT_MD_MS').iterdir():
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
        bscan = vol[vol.shape[0] // 2]
        if id_.startswith('hc'):
            label = 'Healthy'
        elif id_.startswith('ms'):
            label = 'Multiple_Sclerosis'
        print(f'{id_} {bscan.shape}, label: {label}\n')
        dataset[label][id_] = {
            'bscan': bscan,
            'label': label
        }
    # split into train/val/test (33/33/33)
    for label in dataset:
        ids = list(dataset[label].keys())
        random.shuffle(ids)
        val_start = int(len(ids) * 0.33)
        test_start = int(len(ids) * 0.66)
        for idx, id_ in enumerate(ids):
            if idx < val_start:
                split = 'train'
            elif idx < test_start:
                split = 'val'
            else:
                split = 'test'
            bscan = dataset[label][id_]['bscan']
            tgt_fn = TGT_BASE_PATH / 'OCT_MD_MS' / split / label / f'{id_}.png'
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(tgt_fn, bscan)


def banglaoct2025():
    path_dir = Path('/mnt/Data/BanglaOCT2025/BanglaOCT2025_Raw_130_Images_Only')
    random.seed(2)
    # NOTE: Very few images for validation and test in the original
    #   split. Recreating splits.
    if (TGT_BASE_PATH / 'BanglaOCT2025').exists():
        for child in (TGT_BASE_PATH / 'BanglaOCT2025').iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for disease in path_dir.iterdir():
        if not disease.is_dir():
            continue
        patients = list(disease.iterdir())
        random.shuffle(patients)
        val_start = int(len(patients) * 0.40)
        test_start = int(len(patients) * 0.70)
        for i, patient in enumerate(patients):
            if not patient.is_dir():
                continue
            bscan_fns = sorted(patient.glob('oct_c_*.bmp'))
            if not bscan_fns:
                print(f'No B-scans found for {patient}')
                continue
            middle_bscan_fn = bscan_fns[len(bscan_fns) // 2]
            print(f'{disease.name}/{patient.name} has {len(bscan_fns)} B-scans')
            bscan = io.imread(middle_bscan_fn)
            if i < val_start:
                split = 'train'
            elif i < test_start:
                split = 'val'
            else:
                split = 'test'
            bscan_gray = convert_to_grayscale(bscan)
            tgt_fn = TGT_BASE_PATH / 'BanglaOCT2025' / split / disease.stem / f'{patient.name}_{middle_bscan_fn.stem}.png'
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            print(' ', tgt_fn)
            io.imsave(tgt_fn, bscan_gray)


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
    # diseases = info['Disease'].unique()
    diseases = {'Normal': ['NORMAL'], 'DR': ['DR']}
    diseases = {'Normal': ['NORMAL'], 'Diseased': ['CNV', 'DR', 'AMD']}
    samples = {}
    for group_name, disease_group in diseases.items():
        disease_vols = []
        for disease in disease_group:
            disease_vols.extend(info[info['Disease'] == disease]['ID'].values)
        samples[group_name] = disease_vols
    data_dir = path_dir / 'dataset_3m' / 'OCT'
    for group_name, disease_vols in samples.items():
        random.shuffle(disease_vols)
        val_start = int(len(disease_vols) * 0.40)
        test_start = int(len(disease_vols) * 0.70)
        print(f'{group_name} has {len(disease_vols)}')
        for i, disease_vol in enumerate(disease_vols):
            print(f'  {disease_vol}')
            vol_dir = data_dir / str(disease_vol)
            print(f'    Looking for B-scans in {vol_dir}')
            img_fns = sorted(vol_dir.glob('*.bmp'))
            indices = sorted([int(fn.stem) for fn in img_fns])
            middle_idx = indices[len(indices) // 2]
            middle_bscan_fn = vol_dir / f'{middle_idx}.bmp'
            middle_bscan = io.imread(middle_bscan_fn)
            print(f'    Found B-scans with indices: {indices}')
            print(f'    Found {len(img_fns)} B-scans')
            if i < val_start:
                split = 'train'
            elif i < test_start:
                split = 'val'
            else:
                split = 'test'
            middle_bscan_gray = convert_to_grayscale(middle_bscan)
            tgt_fn = TGT_BASE_PATH / 'OCTA500' / split / group_name / f'{disease_vol}_{middle_bscan_fn.stem}.png'
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            print(' ', tgt_fn)
            io.imsave(tgt_fn, middle_bscan_gray)


def mmcamd():
    # https://github.com/search?q=repo%3Ali-xirong%2Fmmc-amd%20dryAMD&type=code
    path_dir = BASE_DATA_PATH / 'MMC-AMD'
    splits_dir = path_dir / 'mmc-amd-splitAP'
    oct_dir = path_dir / 'ImageData' / 'oct'
    if (TGT_BASE_PATH / 'MMC_AMD').exists():
        for child in (TGT_BASE_PATH / 'MMC_AMD').iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for subset in ['train', 'val', 'test']:
        split_fn = splits_dir / f'mmc-amd-splitAP-{subset}' / 'ImageSets' / 'oct.txt'
        with open(split_fn, 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            line = line.strip()
            print(line)
            disease_char = line.split('-')[1]
            if disease_char == 'h':
                disease = 'Normal'
            elif disease_char == 'd':
                disease = 'Dry_AMD'
            elif disease_char == 'w':
                disease = 'Wet_AMD'
            elif disease_char == 'p':
                disease = 'PCV'
            else:
                raise ValueError(f'Unknown disease char {disease_char} in line {line}')
            oct_fn = oct_dir / f'{line}.jpg'
            oct = io.imread(oct_fn)
            # Add random noise L in the bottom left corner to remove some
            #   identifying info from the image (e.g. scale bar)
            oct = convert_to_grayscale(oct)
            # Copy the crop next to the bottom left corner and add some
            #   noise. NOTE: do this to make sure all images from
            #   diffferent classes look the same, as some classes
            #   always had this thing and others not.
            noise = np.random.randint(0, 5, size=(64, 48), dtype=np.uint8)
            center_w = oct.shape[1] // 2
            crop = oct[-64:, center_w - 24:center_w + 24 ]
            oct[-64:, :48] = crop + noise
            assert oct_fn.exists(), f'File {oct_fn} does not exist'
            tgt_fn = TGT_BASE_PATH / 'MMC_AMD' / subset / disease / oct_fn.name
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            print(' ', tgt_fn)
            # shutil.copy(oct_fn, tgt_fn)
            io.imsave(tgt_fn, oct)


def m3ret():
    path_dir = Path('/mnt/Data/M3Ret/')
    info_fn = path_dir / 'labels_data_split/oct-macular-v4.6.xlsx'
    info_train = pd.read_excel(info_fn, sheet_name='train-1216')
    info_val = pd.read_excel(info_fn, sheet_name='val-406')
    info_test = pd.read_excel(info_fn, sheet_name='test-406')
    if (TGT_BASE_PATH / 'M3Ret').exists():
        for child in (TGT_BASE_PATH / 'M3Ret').iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    # print(info_train.keys())
    # Index(['patient ID', 'patient name', 'laterality', 'year', 'month', 'day',
    #        'ME', 'DR', 'glaucoma', 'cataract', 'optic_atrophy', 'birthday', 'age',
    #        'gender', 'Images'],
    #       dtype='str')
    for subset, info in zip(['train', 'val', 'test'], [info_train, info_val, info_test]):
        for idx, row in info.iterrows():
            if row['ME'] == 'ME' and row['DR'] == 'Negative':
                me_cls = 'ME'
            elif row['ME'] == 'ME' and row['DR'] == 'DR':
                me_cls = 'ME_DR'
            elif row['ME'] == 'Negative' and row['DR'] == 'DR':
                me_cls = 'DR'
            elif row['ME'] == 'Negative' and row['DR'] == 'Negative':
                me_cls = 'Neither'
            else:
                continue
            pat_id = row['patient ID']
            laterality = row['laterality']
            name = row['patient name']
            img_dir = path_dir / 'OCTImages-Macular-v3.4' / f'{str(pat_id).zfill(10)}+{name}' / laterality / 'study1'
            assert img_dir.exists(), f'Image directory {img_dir} does not exist'
            img_fns = sorted(img_dir.glob('volume_1_*.png'))
            bscan_ids = sorted([int(fn.stem.split('_')[-1]) for fn in img_fns])
            middle = bscan_ids[len(bscan_ids) // 2]
            middle_fn = img_dir / f'volume_1_{middle}.png'
            assert middle_fn.exists(), f'Middle B-scan {middle_fn} does not exist'
            tgt_fn = TGT_BASE_PATH / 'M3Ret' / subset / me_cls / f'{pat_id}_{laterality}_{middle_fn.stem}.png'
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            print(' ', tgt_fn)
            shutil.copy(middle_fn, tgt_fn)



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


def octave():
    random.seed(1024)
    path_dir = Path('/mnt/Data/OCTAVE/nnUNet_raw/Dataset001_OCTAVE/imagesTr/')
    data = {}
    for img_fn in path_dir.iterdir():
        if img_fn.suffix != '.tif':
            continue
        parts = img_fn.stem.split('_')
        disease = parts[1]
        iid = parts[2]
        print(f'Processing {img_fn.stem} with disease {disease} and id {iid}')
        if disease not in data:
            data[disease] = []
        data[disease].append(iid)
    # Split into train/val/test (40/30/30)
    splits = {}
    for disease, iids in data.items():
        random.shuffle(iids)
        val_start = int(len(iids) * 0.4)
        test_start = int(len(iids) * 0.7)
        splits[disease] = {
            'train': iids[:val_start],
            'val': iids[val_start:test_start],
            'test': iids[test_start:]
        }
    for disease, splits in splits.items():
        for split, iids in splits.items():
            for iid in iids:
                img_fn = path_dir / f'OCTAVE_{disease}_{iid}_0000.tif'
                # Read image and show shape
                img = io.imread(img_fn)
                print(f'Image shape: {img.shape}')
                # Take the middle B-scan
                middle = img.shape[0] // 2
                bscan = img[middle]
                # Convert to grayscale if needed
                bscan = convert_to_grayscale(bscan)
                bscan = remove_white_part(bscan)
                print(f'B-scan grayscale shape: {bscan.shape}')
                middle_str = str(middle).zfill(3)
                tgt_fn = TGT_BASE_PATH / 'OCTAVE' / split / disease / f'{disease}_{iid}_{middle_str}.png'
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                print(' ', tgt_fn)
                io.imsave(tgt_fn, bscan)


def mario():
    path_dir = Path('/home/morano/Bookmarks/Data/MARIO/Task_1/')
    tgt_base = TGT_BASE_PATH / 'MARIO_T1'
    pattern_fn = 'df_task1_{subset}.csv'
    class_mapping = {
        0: 'reduced',
        1: 'stable',
        2: 'worsened',
        3: 'other',
    }
    for subset in ['train', 'val', 'test']:
        info_fn = path_dir / pattern_fn.format(subset=subset)
        info = pd.read_csv(info_fn)
        img_dir = path_dir / subset
        # id_patient,side_eye,BScan,image_at_ti,image_at_ti+1,label,split_type,LOCALIZER_at_ti+1,LOCALIZER_at_ti,sex,age_at_ti+1,age_at_ti,num_current_visit_at_i+1,num_current_visit_at_i,delta_t,case
        for idx, row in info.iterrows():
            img_1_fn = Path(row['image_at_ti'])
            img_2_fn = Path(row['image_at_ti+1'])
            label_int = row['label']
            label_str = class_mapping[label_int]
            img_fn_1 = img_dir / img_1_fn
            img_fn_2 = img_dir / img_2_fn
            assert img_fn_1.exists(), img_fn_1
            assert img_fn_2.exists(), img_fn_2
            img_1 = io.imread(img_fn_1)
            img_2 = io.imread(img_fn_2)
            img_1 = convert_to_grayscale(img_1)
            img_2 = convert_to_grayscale(img_2)
            padding = np.zeros_like(img_1)
            img_combined = np.stack([img_1, img_2, padding], axis=-1)
            tgt_fn = tgt_base / subset / label_str / f'{img_1_fn.stem}_{img_2_fn.stem}.png'
            tgt_fn.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(tgt_fn, img_combined)
            print(' ', tgt_fn)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str, help='Function to execute')
    args = parser.parse_args()

    globals()[args.function]()
