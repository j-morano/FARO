from pathlib import Path
import yaml
import lzma
import json
from collections import OrderedDict
import time
import multiprocessing
import shutil
import logging
import gzip

import numpy as np
from skimage import io
from skimage.transform import resize
from matplotlib.figure import Figure

from lesionlib import LesionSet

# Setup logging at the top of your script
logging.basicConfig(
    filename='__v2_data_processing/__layer_read_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(message)s'
)
# Clean log file at start
open('__v2_data_processing/__layer_read_errors.log', 'w').close()

metadata_fn = Path('__v2_data_processing') / 'vibes_metadata.json.gz'
with gzip.open(metadata_fn, 'rt') as f:
    metadata = json.load(f)
'''Example entry:
'19302666-25-LASANJUAWVRORMAYEJARDWLOT+XMXSEKKJLUTWQANQMNJIBMCYANNWFUNPCKUJG+LARHIY': {'SourceId': 1899452, 'ScanId': 20650542, 'FileSetId': '19302666-25-LASANJUAWVRORMAYEJARDWLOT+XMXSEKKJLUTWQANQMNJIBMCYANNWFUNPCKUJG+LARHIY', 'Study': 'VIBES', 'Site': 'AT09000', 'VRCPatId': '25044', 'Vendor': 'Heidelberg Engineering', 'VRCTicket': 'VIBES_25044_20200908', 'VRCScan': 'VIBES_25044_780342_19_17_20200908084123_OD_35133_6', 'DataLocation': 'OPTDATA_VIBES', 'Disease': 'Unknown', 'StudyEye': '??', 'VisitType': 'UNKNOWN', 'VisitId': 981273, 'DayInStudy': 0, 'NrBScans': 19, 'PatientId': 282761, 'ExternalSourceId': '19302666-25-LASANJUAWVRORMAYEJARDWLOT', 'Sex': 'M', 'VisitTime': 1599561633, 'ScanSpacing_mus': 249.00900268554688, 'Position': 'OD', 'Name': 'Macular Cube 1024x19 30&deg; ART', 'Remark': '20210701__patid_35133__e2e_35133_6_progid__b00d0c6fb1bc40fc91888a547b1ddd4d__unrotate_0_gainoct_1.00_gainslo_1.02__converter_type__converter_date_', 'SourceTypeId': 1, 'CreationTime': '2021-10-31_03:16:02', 'ScanFolder': '/optima/data/OPTDATA_VIBES/19302666-25-LASANJUAWVRORMAYEJARDWLOT+XMXSEKKJLUTWQANQMNJIBMCYANNWFUNPCKUJG+LARHIY'}
'''

layer_path = Path('/optima/exchange/VIBES_SD-RetinaNet/')

base_path = Path('/mnt/Data/SSHFS/msc_server/Mini_Datasets/FullVIBES-v2/bscan/')
base_path = Path('/mnt/Data/SSHFS/msc_raid/FullVIBES-v2/bscan/')
save_path = base_path.parent / 'bscanlayermap'
save_path.mkdir(exist_ok=True)
original_save_path = save_path.parent / 'segs'
original_save_path.mkdir(exist_ok=True)

for entry in save_path.iterdir():
    shutil.rmtree(entry)
for entry in original_save_path.iterdir():
    shutil.rmtree(entry)

debug_path = Path('__v2_data_processing/__debug')
debug_path.mkdir(exist_ok=True)

for entry in debug_path.iterdir():
    entry.unlink()


# def get_layer_maps_from_yml_old(yml_path):
#     """
#     Parses the .yml.xz trace and renders it into a 2D segmentation mask.
#     Fills layers from top to bottom using broadcasting.
#     """
#     # 1. Load the compressed YAML
#     with lzma.open(yml_path, 'rt') as f:
#         data = yaml.safe_load(f)

#     width = data['info']['width']
#     height = data['info']['height']
#     # The layers are in a list: data['layers'] = [{'ILM': [...]}, {'RPE': [...]}, ...]
#     raw_layers = data['layers']
#     num_layers = len(raw_layers)

#     # 2. Extract strings into a numerical array (width, num_layers)
#     # We use a 2D array because your original logic was designed for 3D (num_bscans)
#     # but here we are processing one B-scan at a time.
#     layer_coords = np.zeros((width, num_layers), dtype=int)

#     for i, layer_dict in enumerate(raw_layers):
#         # Extract the values string (assuming one entry per layer dictionary)
#         layer_name = list(layer_dict.keys())[0]
#         val_str = layer_dict[layer_name][0]['values']
#         coords = np.fromstring(val_str, sep=',', dtype=int)
#         layer_coords[:, i] = coords

#     # 3. Create Meshgrid for the single B-scan
#     # xx: width, yy: height
#     yy, xx = np.meshgrid(
#         np.arange(height),
#         np.arange(width),
#         indexing='ij'
#     )

#     # 4. Fill the masks
#     # We initialize with 0 (Background)
#     mask = np.zeros((height, width), dtype=np.uint8)

#     # Apply the same logic: every subsequent layer overwrites the previous
#     # if it's "deeper" (greater Y value)
#     for i in range(num_layers):
#         # layer_coords[xx, i] gets the boundary height for that column
#         mask[yy >= layer_coords[xx, i]] = i + 1

#     return mask


def get_layer_maps_from_yml(yml_path):
    with lzma.open(yml_path, 'rt') as f:
        data = yaml.safe_load(f)

    width = data['info']['width']
    height = data['info']['height']
    raw_layers = data['layers']

    # 1. Pre-allocate the mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # 2. Vectorized Y-indices (the "column")
    y_indices = np.arange(height).reshape(-1, 1)

    # 3. Process layers
    for i, layer_dict in enumerate(raw_layers):
        layer_name = list(layer_dict.keys())[0]
        val_str = layer_dict[layer_name][0]['values']

        # np.fromstring is very CPU-efficient
        coords = np.fromstring(val_str, sep=',', dtype=int)

        # BROADCASTING: This is the CPU-fastest part.
        # It creates the boolean mask on the fly in the CPU registers.
        mask[y_indices >= coords] = i + 1

    return mask


# mask = get_layer_maps_from_yml(sample_path / 'layers' / '000.yml.xz')
# plt.imshow(mask, cmap='tab20')
# plt.show()

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

def process_entries(entries):
    total_processed = 0
    for entry in entries:
        start = time.time()
        print(entry)
        fsid = entry.name
        print('Processing fsid:', fsid)
        hashn = fsid.split('-')[2]
        first_dir = hashn[0:2]
        second_dir = hashn[2:3]
        bscan_entries = list(entry.iterdir())
        if len(bscan_entries) != min(metadata[fsid]['NrBScans'], 49):
            logging.error(f"FSID: {fsid} | B-scan: All")
            print(f' WARNING: Expected {metadata[fsid]["NrBScans"]} B-scans, found {len(bscan_entries)}.')
            print(f' Skipping FSID due to B-scan count mismatch.')
            continue
        original_save_parent = original_save_path / fsid
        original_save_parent.mkdir(exist_ok=True)
        original_save_parent_layers = original_save_parent / 'layers'
        original_save_parent_layers.mkdir(exist_ok=True)
        original_save_parent_lesions = original_save_parent / 'lesions'
        original_save_parent_lesions.mkdir(exist_ok=True)
        for bscan_entry in bscan_entries:
            bscan = io.imread(bscan_entry)
            # Spectralis__OD__000-002.png
            parts = bscan_entry.stem.split('__')
            device = parts[0]
            eye = parts[1]
            bscan_id = parts[-1].split('-')[0]
            if (device != VENDOR_MAPPING[metadata[fsid]['Vendor']]
                or eye != metadata[fsid]['Position']
            ):
                logging.error(f"FSID: {fsid} | B-scan: {bscan_id}")
                print(f'  Skipping B-scan id: {bscan_id} due to vendor/eye mismatch.')
            print(f'  ({total_processed}) B-scan id: {bscan_id}')
            c_base_path = layer_path / first_dir / second_dir / fsid
            yml_path = c_base_path / 'layers' / f'{bscan_id}.yml.xz'
            lesions_path = c_base_path / 'lesions' / f'{bscan_id}.png'
            # assert yml_path.exists(), yml_path
            # assert lesions_path.exists(), lesions_path
            try:
                mask = get_layer_maps_from_yml(yml_path)
            except FileNotFoundError:
                logging.error(f"FSID: {fsid} | B-scan: {bscan_id}")
                print('    IMPORTANT: Layer file not found, skipping.')
                continue
            shutil.copy(yml_path, original_save_parent_layers)
            shutil.copy(lesions_path, original_save_parent_lesions)
            print('    Loaded layer mask with shape:', mask.shape, 'and unique labels:', np.unique(mask))
            loaded_set = LesionSet.load(lesions_path)
            print('    Loaded lesion set with', len(loaded_set.lesions), 'lesions.')
            for lesion in LESION_MAPPING.keys():
                print('      Lesion:', lesion)
                lesion_mask = loaded_set[lesion].mask
                try:
                    mask[lesion_mask > 0] = LESION_MAPPING[lesion]
                except ValueError:
                    logging.error(f"FSID: {fsid} | B-scan: {bscan_id}")
                    print('        IMPORTANT: Error processing lesion mask, skipping this lesion.')
                    continue
            # lesion_mask = resize(lesion_mask, bscan.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            # print('    Loaded lesion mask with shape:', lesion_mask.shape, 'and unique labels:', np.unique(lesion_mask))
            mask = resize(mask, bscan.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            if total_processed % 100 == 0:
                fig = Figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.imshow(bscan, cmap='gray')
                ax.imshow(mask, cmap='tab20', alpha=0.3, vmin=0, vmax=19)
                # ax.set_title(f'FSID: {fsid} | Max Label: {np.max(mask)}')
                ax.set_title(f'{fsid}\nMax: {np.max(mask)}')
                ax.axis('off')
                fig.savefig(debug_path / f'{fsid}_{bscan_id}.jpg', bbox_inches='tight', dpi=60)
                fig.clear()
            save_parent = save_path / fsid
            save_parent.mkdir(exist_ok=True)
            save_fn = save_parent / bscan_entry.name
            io.imsave(save_fn, mask, check_contrast=False)
            total_processed += 1
        end = time.time()
        print(f'  Processed in {end - start:.2f} seconds.')



if __name__ == '__main__':
    entries = list(base_path.iterdir())

    num_workers = 12
    chunks = np.array_split(entries, num_workers)

    with multiprocessing.Pool(num_workers) as pool:
        pool.map(process_entries, chunks)
