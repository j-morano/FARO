from pathlib import Path
import argparse
import shutil

import numpy as np
import pandas as pd
from scipy.io import loadmat
from skimage import io

from factory import get_factory_adder
from img_utils import remove_white_borders



add_dataset, dataset_factory = get_factory_adder()


base_target_path = Path('/mnt/Data/OCT_Segmentation_Foundation_Model/OCT/')




@add_dataset
def remove_white_borders_test():
    fns = [
        '/mnt/Data/Kermany_3/CellData/OCT/train/DME/DME-30521-212.jpeg',
        '/mnt/Data/Kermany_3/CellData/OCT/train/DME/DME-30521-170.jpeg',
        '/mnt/Data/Kermany_3/CellData/OCT/train/DME/DME-30521-15.jpeg',
        '/mnt/Data/Kermany_3/CellData/OCT/train/DME/DME-30521-27.jpeg',
    ]
    for fn in fns:
        img = io.imread(fn)
        remove_white_borders(img)


@add_dataset
def kermany():
    print('Processing Kermany dataset')
    base_path = '/mnt/Data/Kermany_3/CellData/OCT/'
    base_path = Path(base_path)
    target_path = base_target_path / 'Kermany'
    target_path.mkdir(exist_ok=True)
    for sp in base_path.iterdir():
        for cp in sp.iterdir():
            class_name = cp.name.lower()
            for fn in cp.iterdir():
                idd = fn.stem.split('-')[1:]
                idd = '-'.join(idd)
                print('\t', class_name, idd)
                tgt_fn = target_path / f'{class_name}__{idd}.png'
                if not tgt_fn.exists():
                    img = io.imread(fn)
                    img = remove_white_borders(img)
                    io.imsave(tgt_fn, img)


@add_dataset
def gamma_train():
    print('Processing GAMMA train dataset')
    base_path = Path('/mnt/Data/GAMMA/Glaucoma_grading/training/')
    target_path = base_target_path / 'GAMMA_train'
    target_path.mkdir(exist_ok=True)
    glaucoma_grading_fn = base_path / 'glaucoma_grading_training_GT.xlsx'
    glaucoma_grading = pd.read_excel(glaucoma_grading_fn)
    # print(glaucoma_grading.head())
    images_path = base_path / 'multi-modality_images'
    for _i, row in glaucoma_grading.iterrows():
        idd = str(row['data']).zfill(4)
        c_path = images_path / idd / idd
        if row['non'] == 1:
            class_name = 'non-glaucoma'
        elif row['early'] == 1:
            class_name = 'glaucoma_early'
        elif row['mid_advanced'] == 1:
            class_name = 'glaucoma_mid_advanced'
        else:
            raise ValueError('Unknown class')
        for slice in c_path.iterdir():
            sidd = slice.stem.split('_')[0].zfill(3)
            ext = slice.suffix
            shutil.copy(slice, target_path / f'{class_name}__{idd}_{sidd}{ext}')
            print('\t', idd, class_name, sidd)


@add_dataset
def harvard_gdp1000():
    print('Processing Harvard GDP dataset')
    base_path = Path('/mnt/Data/FoundOPTIMA_Downstream/Harvard_Glaucoma_Detection/Harvard Glaucoma Detection and Progression with 1000 Samples (Harvard-GDP1000)/Bscan/')
    glaucoma_grading_fn = base_path.parent / 'ReadMe' / 'data_summary.csv'
    glaucoma_grading = pd.read_csv(glaucoma_grading_fn)
    target_path = base_target_path / 'Harvard_GDP1000'
    target_path.mkdir(exist_ok=True)
    print(glaucoma_grading.head())
    for _i, row in glaucoma_grading.iterrows():
        fn = row['filename']
        if row['glaucoma'] == 1:
            class_name = 'glaucoma'
        else:
            class_name = 'non-glaucoma'
        volume = np.load(base_path / (fn + '.npz'))
        bscans = volume['bscans']
        # print(bscans.shape)
        # plt.imshow(bscans[0])
        # plt.show()
        for i in range(bscans.shape[0]):
            i_disp = str(i).zfill(3)
            fn_disp = fn.split('_')[1]  # type: ignore
            print('\t', class_name, fn_disp, i_disp)
            io.imsave(target_path / f'{class_name}__{fn_disp}_{i_disp}.png', bscans[i])


@add_dataset
def duke_srinivasan():
    print('Processing Duke Srinivasan dataset')
    base_path = Path('/mnt/Data/FoundOPTIMA_Downstream/Duke_Srinivasan/Publication_Dataset/')
    target_path = base_target_path / 'Duke_Srinivasan'
    target_path.mkdir(exist_ok=True)
    for sample in base_path.iterdir():
        if 'AMD' in sample.name:
            class_name = 'dry_amd'
        elif 'DME' in sample.name:
            class_name = 'dme'
        elif 'NORMAL' in sample.name:
            class_name = 'normal'
        else:
            raise ValueError('Unknown class')
        idd = sample.stem
        for fn in (sample / 'TIFFs' / '8bitTIFFs').iterdir():
            sidd = fn.stem.zfill(3)
            print('\t', class_name, idd, sidd)
            img = io.imread(fn)
            img = remove_white_borders(img)
            io.imsave(target_path / f'{class_name}__{idd}_{sidd}.png', img)


@add_dataset
def ne_hut():
    print('Processing NE HUT dataset')
    base_path = Path('/mnt/Data/NEH_UT/')
    target_path = base_target_path / 'NE_HUT'
    target_path.mkdir(exist_ok=True)
    data_info = pd.read_csv(base_path / 'data_information.csv')
    print(data_info.head())
    # Patient ID, Class Eye, B-scan, Label, Directory
    for _i, row in data_info.iterrows():
        class_name = row['Class'] + '--' + row['Label']
        class_name = class_name.lower()
        idd = str(row['Patient ID']).zfill(3) + '-' + row['Eye']
        sidd = str(row['B-scan']).zfill(3)
        print('\t', class_name, idd, sidd)
        subpath = str(row['Directory'])
        fn = base_path / 'NEH_UT_2021RetinalOCTDataset' / subpath
        if not fn.exists():
            subpath = subpath.replace('NOrmal', 'Normal')
            fn = base_path / 'NEH_UT_2021RetinalOCTDataset' / subpath

        img = io.imread(fn)
        io.imsave(target_path / f'{class_name}__{idd}_{sidd}.png', img)


@add_dataset
def noor_eh():
    print('Processing Noor EH (Eye Hospital) dataset')
    base_path = Path('/mnt/Data/FoundOPTIMA_Downstream/Noor_Eye_Hospital/Data/')
    target_path = base_target_path / 'Noor_EH'
    target_path.mkdir(exist_ok=True)
    for c in base_path.iterdir():
        class_name = c.stem.lower()
        for volume in c.iterdir():
            idd = volume.stem.replace('(' , '').replace(')', '')
            parts = idd.split(' ')
            diagnosis = parts[0]
            number = parts[1]
            idd = f'{diagnosis}_{number.zfill(3)}'
            for fn in volume.iterdir():
                sidd = (
                    fn.stem
                    .replace('(' , '')
                    .replace(')', '')
                    .replace(' ', '')
                    .replace('Image', '')
                ).zfill(3)
                if 'DME' in idd and sidd == '001':
                    continue
                print('\t', class_name, idd, sidd)
                img = io.imread(fn)
                io.imsave(target_path / f'{class_name}__{idd}_{sidd}.png', img)


@add_dataset
def oct_c8():
    base_path = Path('/mnt/Data/OCT-C8/RetinalOCT_Dataset/RetinalOCT_Dataset/')
    target_path = base_target_path / 'OCT_C8'
    target_path.mkdir(exist_ok=True)
    remove_white_for = ['cnv', 'dme', 'drusen', 'normal']
    for subset in base_path.iterdir():
        for c in subset.iterdir():
            class_name = c.stem.lower()
            remove_white = class_name in remove_white_for
            for fn in c.iterdir():
                idd = fn.stem
                # Find first '_' and duplicate it '__'
                # u_idx = idd.index('_')
                # idd = idd[:u_idx] + '_' + idd[u_idx:]
                print('\t', class_name, idd)
                if remove_white:
                    img = io.imread(fn)
                    img = remove_white_borders(img)
                    io.imsave(target_path / f'{class_name}__{idd}.png', img)
                else:
                    shutil.copy(fn, target_path / f'{class_name}__{idd}{fn.suffix}')


@add_dataset
def octdl():
    base_path = Path('/mnt/Data/FoundOPTIMA_Downstream/OCTDL/')
    target_path = base_target_path / 'OCTDL'
    target_path.mkdir(exist_ok=True)
    data_info = pd.read_csv(base_path / 'OCTDL_labels.csv')
    print(data_info.head())
    for i, row in data_info.iterrows():
        idd = str(i).zfill(4)
        disease = row['disease']
        subcategory = row['subcategory']
        condition = row['condition']
        fn = base_path / 'OCTDL' / disease / row['file_name']
        if disease == 'NO':
            disease = 'Normal'
        if subcategory == 'NO':
            subcategory = 'Normal'
        if condition == 'NO':
            condition = 'Normal'
        class_name = f'{disease}_{subcategory}-{condition}'
        class_name = class_name.lower()
        print('\t', class_name, idd)
        # assert fn.exists(), f'{fn} does not exist'
        shutil.copy(fn, target_path / f'{class_name}__{idd}{fn.suffix}')


@add_dataset
def octid():
    print('Processing OCTID dataset')
    base_path = Path('/mnt/Data/FoundOPTIMA_Downstream/OCTID/images')
    target_path = base_target_path / 'OCTID'
    target_path.mkdir(exist_ok=True)
    for s in base_path.iterdir():
        stem = s.stem
        if 'NORMAL' in stem:
            class_name = 'normal'
        elif 'MH' in stem:
            class_name = 'macular_hole'
        elif 'AMRD' in stem:
            class_name = 'amd'
        elif 'CSR' in stem:
            class_name = 'central_serous_retinopathy'
        elif 'DR' in stem:
            class_name = 'diabetic_retinopathy'
        else:
            raise ValueError('Unknown class')
        shutil.copy(s, target_path / f'{class_name}__{stem}{s.suffix}')


@add_dataset
def olives():
    base_path = Path('/mnt/Data/FoundOPTIMA_Downstream/OLIVES/OLIVES/')
    target_path = base_target_path / 'OLIVES'
    target_path.mkdir(exist_ok=True)
    class_name = 'dme'
    for study in (base_path / 'TREX DME').iterdir():
        for patient in study.iterdir():
            if not patient.is_dir():
                continue
            for visit in patient.iterdir():
                if not visit.is_dir():
                    continue
                for eye in visit.iterdir():
                    for fn in eye.iterdir():
                        if '000' not in fn.stem:
                            continue
                        sidd = str(int(fn.stem.split('_')[-1])).zfill(3)
                        idd = patient.stem + '_' + visit.stem + '_' + eye.stem + '_' + sidd
                        print(idd)
                        img = io.imread(fn)
                        io.imsave(target_path / f'{class_name}__{idd}.png', img)
    class_name = 'dr'
    for patient in (base_path / 'Prime_FULL').iterdir():
        if not patient.is_dir():
            continue
        for visit in patient.iterdir():
            if not visit.is_dir():
                continue
            for eye in visit.iterdir():
                for fn in eye.iterdir():
                    stem = fn.stem
                    # Check if stem is a number
                    try:
                        int(stem)
                    except ValueError:
                        continue
                    idd = patient.stem + '_' + visit.stem + '_' + eye.stem + '_' + fn.stem.zfill(3)
                    print(idd)
                    img = io.imread(fn)
                    io.imsave(target_path / f'{class_name}__{idd}.png', img)


@add_dataset
def poagd():
    print('Processing POAGD dataset')
    base_path = Path('/mnt/Data/POAG_detection/glaucoma_detection/')
    target_path = base_target_path / 'POAGD'
    target_path.mkdir(exist_ok=True)
    for fn in base_path.iterdir():
        data = np.load(fn)
        # print(data.shape)
        class_name = fn.stem.split('-')[0].lower()
        # plt.imshow(data[0], cmap='gray')
        # plt.show()
        for i in range(data.shape[0]):
            idx = fn.stem.index('-')
            idd = fn.stem[idx+1:]
            sidd = str(i).zfill(3)
            print('\t', class_name, idd, sidd)
            io.imsave(target_path / f'{class_name}__{idd}_{sidd}.png', data[i])



@add_dataset
def umn():
    print('Processing UMN dataset')
    base_path = Path('/mnt/Data/FoundOPTIMA_Downstream/UMN/')
    data_fn = 'UMN_Dataset.mat'
    target_path = base_target_path / 'UMN'
    target_path.mkdir(exist_ok=True)

    for c in base_path.iterdir():
        if not c.is_dir():
            continue
        dataset = loadmat(c / data_fn)
        all_subjects = dataset['AllSubjects']
        # print(all_subjects.shape)
        for i in range(len(all_subjects[0])):
            # print(all_subjects[0][i].shape)
            patient = str(i).zfill(3)
            for j in range(all_subjects[0][i].shape[-1]):
                sidd = str(j).zfill(3)
                class_name = c.stem.lower()
                bscan = all_subjects[0][i][..., j]
                idd = f'{class_name}_{patient}_{sidd}'
                print('\t', class_name, idd)
                io.imsave(target_path / f'{class_name}__{idd}.png', bscan)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset name', required=True)
    args = parser.parse_args()

    dataset_factory[args.dataset]()

