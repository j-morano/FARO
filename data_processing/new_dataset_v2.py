from pathlib import Path
import yaml
import lzma
import json
from collections import OrderedDict
import time
import multiprocessing
import shutil
import logging
import argparse
import gzip
import pickle
import cv2

import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import pydicom
from PIL import Image

from lesionlib import LesionSet
from img_utils import get_layer_maps_from_yml



########################################################################
# Constants


# '''Example metadata entry:
# '19302666-25-LASANJUAWVRORMAYEJARDWLOT+XMXSEKKJLUTWQANQMNJIBMCYANNWFUNPCKUJG+LARHIY': {'SourceId': 1899452, 'ScanId': 20650542, 'FileSetId': '19302666-25-LASANJUAWVRORMAYEJARDWLOT+XMXSEKKJLUTWQANQMNJIBMCYANNWFUNPCKUJG+LARHIY', 'Study': 'VIBES', 'Site': 'AT09000', 'VRCPatId': '25044', 'Vendor': 'Heidelberg Engineering', 'VRCTicket': 'VIBES_25044_20200908', 'VRCScan': 'VIBES_25044_780342_19_17_20200908084123_OD_35133_6', 'DataLocation': 'OPTDATA_VIBES', 'Disease': 'Unknown', 'StudyEye': '??', 'VisitType': 'UNKNOWN', 'VisitId': 981273, 'DayInStudy': 0, 'NrBScans': 19, 'PatientId': 282761, 'ExternalSourceId': '19302666-25-LASANJUAWVRORMAYEJARDWLOT', 'Sex': 'M', 'VisitTime': 1599561633, 'ScanSpacing_mus': 249.00900268554688, 'Position': 'OD', 'Name': 'Macular Cube 1024x19 30&deg; ART', 'Remark': '20210701__patid_35133__e2e_35133_6_progid__b00d0c6fb1bc40fc91888a547b1ddd4d__unrotate_0_gainoct_1.00_gainslo_1.02__converter_type__converter_date_', 'SourceTypeId': 1, 'CreationTime': '2021-10-31_03:16:02', 'ScanFolder': '/optima/data/OPTDATA_VIBES/19302666-25-LASANJUAWVRORMAYEJARDWLOT+XMXSEKKJLUTWQANQMNJIBMCYANNWFUNPCKUJG+LARHIY'}
# '''


# DATA
MAX_LAYERS = 12
LESION_MAPPING = OrderedDict({
    'HRF': 13,
    'Cyst': 14,
    'SRF': 15,
    'PED': 16,
    'SHRM': 17,
    'Pseudodrusen': 18,
    'ORT': 19,
})
VENDOR_MAPPING = {
    'Heidelberg Engineering': 'Spectralis',
    'Cirrus': 'Cirrus',
}

# PATHS
SEGS_PATH = Path('/optima/exchange/VIBES_SD-RetinaNet/')
BASE_SAVE_PARENT = Path('__new_dataset_v2')
DEBUG_PATH = BASE_SAVE_PARENT / '__debug'
# DATA_SAVE_PATH = Path('/home/optima/jsanchez/hpc-mnt/FullVIBES-v2/')
DATA_SAVE_PATH = Path('/mnt/Data/SSHFS/msc_raid/FullVIBES-v2/')
BSCANLAYER_SAVE_PATH = DATA_SAVE_PATH / 'v2_bscanlayermap'

# PROCESS
MAX_SLICES = 49
TGT_SIZE = (512, 512)
NUM_WORKERS = 16


########################################################################
# Basic functions overrides and config

# print = partial(print, flush=True)

cv2.setNumThreads(1)

# Setup logging at the top of your script
logging.basicConfig(
    filename=BASE_SAVE_PARENT / '__layer_read_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(message)s'
)


########################################################################
# Functions


def split_to_slices(data):
    total = data.shape[0]
    if total > 50:
        start = 1
        end = total - 2
    else:
        start = 0
        end = total - 1
    num_slices = min(total, MAX_SLICES)
    selected_indices = np.linspace(start, end, num_slices, dtype=int)
    return selected_indices, total


def compress8(img):
    # Scale based on the data type's maximum possible value
    if img.dtype == np.uint16:
        img = img / 65535
    elif img.dtype == np.uint8:
        img = img / 255
    return (img * 255).astype(np.uint8)


def process_entries(entries):
    proc_id = multiprocessing.current_process().pid
    print(f'Process {proc_id} started with {len(entries)} entries.')
    samples = []
    total_processed = 0
    for fsid, oct_fn, nrbscans in entries:
        start = time.time()
        print(fsid)
        hashn = fsid.split('-')[2]
        first_dir = hashn[0:2]
        second_dir = hashn[2:3]

        # Create tgt dirs
        bscanlayer_save_parent = BSCANLAYER_SAVE_PATH / first_dir / second_dir / fsid
        bscanlayer_save_parent.mkdir(parents=True, exist_ok=True)

        existing_files = list(bscanlayer_save_parent.glob('*.png'))
        num_tgt_slices = min(nrbscans, MAX_SLICES)
        if len(existing_files) == num_tgt_slices:
            print(f'  All {num_tgt_slices} B-scans already processed, skipping.')
            continue

        oct_data = pydicom.dcmread(oct_fn)
        oct_data.SamplesPerPixel = 1  # Ensure single channel
        oct_array = oct_data.pixel_array

        oct_array = compress8(oct_array)
        indices, total = split_to_slices(oct_array)
        c_base_path = SEGS_PATH / first_dir / second_dir / fsid
        for bscan_id in indices:
            bscan = oct_array[bscan_id]
            # bscan_resized = bscan_pil.resize(TGT_SIZE, resample=Image.Resampling.BILINEAR)
            save_fn = f'{bscan_id:03d}.png'
            # bscan_resized.save(bscan_save_parent / save_fn)
            yml_path = c_base_path / 'layers' / f'{bscan_id:03d}.yml.xz'
            lesions_path = c_base_path / 'lesions' / f'{bscan_id:03d}.png'
            # assert yml_path.exists(), yml_path
            # assert lesions_path.exists(), lesions_path
            try:
                mask = get_layer_maps_from_yml(yml_path)
            except FileNotFoundError:
                logging.error(f"FSID: {fsid} | B-scan: {bscan_id}")
                print('    IMPORTANT: Layer file not found, skipping.')
                print('      YML Path:', yml_path)
                continue
            # shutil.copy(yml_path, original_save_parent_layers)
            # shutil.copy(lesions_path, original_save_parent_lesions)
            # print(
            #     '    Loaded layer mask with shape:',
            #     mask.shape,
            #     'and unique labels:',
            #     np.unique(mask)
            # )
            loaded_set = LesionSet.load(lesions_path)
            # print('    Loaded lesion set with', len(loaded_set.lesions), 'lesions.')
            lesion_mask = np.zeros_like(mask, dtype=np.uint8)
            for lesion in LESION_MAPPING.keys():
                # print('      Lesion:', lesion)
                c_lesion_mask = loaded_set[lesion].mask
                lesion_mask[c_lesion_mask > 0] = LESION_MAPPING[lesion]
            bscan_resized = cv2.resize(
                bscan,
                dsize=(TGT_SIZE[1], TGT_SIZE[0]),
                interpolation=cv2.INTER_LINEAR
            )
            mask_resized = cv2.resize(
                mask,
                dsize=(TGT_SIZE[1], TGT_SIZE[0]),
                interpolation=cv2.INTER_NEAREST
            )
            lesion_mask_resized = cv2.resize(
                lesion_mask,
                dsize=(TGT_SIZE[1], TGT_SIZE[0]),
                interpolation=cv2.INTER_NEAREST
            )
            # print(masks_resized.shape)
            image_to_save = np.stack(
                [bscan_resized, mask_resized, lesion_mask_resized],
                axis=-1
            ).astype(np.uint8)
            # print(image_to_save.shape, image_to_save.dtype)
            img_pil = Image.fromarray(image_to_save)
            img_pil.save(bscanlayer_save_parent / save_fn)
            # Append to samples the subdir for the file
            samples.append(str(bscanlayer_save_parent.relative_to(DATA_SAVE_PATH) / save_fn))
            if total_processed % 200 == 0:
                fig = Figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.imshow(image_to_save[:,:,0], cmap='gray')
                ax.imshow(image_to_save[:,:,1], cmap='tab20', alpha=0.3, vmin=0, vmax=19)
                ax.imshow(image_to_save[:,:,2], cmap='hot', alpha=0.3, vmin=0, vmax=19)
                # ax.set_title(f'FSID: {fsid} | Max Label: {np.max(mask)}')
                ax.set_title(f'{fsid}\nMax: {np.max(mask)}')
                ax.axis('off')
                fig.savefig(
                    DEBUG_PATH / f'{fsid}_{bscan_id}.jpg',
                    bbox_inches='tight',
                    dpi=50
                )
                plt.close(fig)
            total_processed += 1
        end = time.time()
        print(f'  Processed in {end - start:.2f} seconds.')
    with gzip.open(BASE_SAVE_PARENT / f'samples_{proc_id}.json.gz', 'wt') as f:
        json.dump(samples, f)



########################################################################
# Main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Recreate the FSID list even if it exists.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Override existing processed data.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode.'
    )
    args = parser.parse_args()

    if args.debug:
        print('DEBUG MODE ENABLED')
        NUM_WORKERS = 1

    # Clean log file at start
    open(BASE_SAVE_PARENT / '__layer_read_errors.log', 'w').close()

    # Create and clean dirs
    BASE_SAVE_PARENT.mkdir(exist_ok=True)
    for dir in [
        BSCANLAYER_SAVE_PATH,
        DEBUG_PATH
    ]:
        dir.mkdir(exist_ok=True)
        if args.overwrite:
            for subdir in dir.iterdir():
                if subdir.is_dir():
                    shutil.rmtree(subdir)
                else:
                    subdir.unlink()
    if args.overwrite:
        for fn in BASE_SAVE_PARENT.iterdir():
            if fn.name.startswith('samples_') and fn.name.endswith('.json.gz'):
                fn.unlink()

    with gzip.open(BASE_SAVE_PARENT / 'v2_all_metadata.pkl.gz', 'rb') as f:
        metadata = pickle.load(f)
    entries = []
    for fsid, cmetadata in metadata.items():
        entries.append((fsid, Path(cmetadata['oct_fn']), cmetadata['NrBScans']))

    chunks = np.array_split(entries, NUM_WORKERS)

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        pool.map(process_entries, chunks)
