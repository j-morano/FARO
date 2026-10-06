from pathlib import Path
import gzip
import pickle
import argparse
import joblib
import gc
import json
import shutil

import numpy as np
from tqdm import tqdm




def create_file(args):
    print("Creating the training databank from metadata...")

    fn = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/metadata_with_thicknesses.pkl.gz'

    with gzip.open(fn, 'rb') as f:
        metadata = pickle.load(f)

    '''Example:
    67115192-25-GDQTLYOJFBUWHARGKILBRPWFA+CRTAHULNZOVABDJOCRPJSSZWIMENVWCGMLVPZ+ESDGTA {'SourceId': 1822580, 'ScanId': 20573858, 'FileSetId': '67115192-25-GDQTLYOJFBUWHARGKILBRPWFA+CRTAHULNZOVABDJOCRPJSSZWIMENVWCGMLVPZ+ESDGTA', 'Study': 'VIBES', 'Site': 'AT09000', 'VRCPatId': '10008801', 'Vendor': 'Cirrus', 'VRCTicket': 'VS_ARGOS-8801_20170315', 'VRCScan': '1.2.276.0.75.2.2.42.896740711157.20170315134807813.253286026', 'DataLocation': 'OPTDATA_VIBES', 'Disease': 'Unknown', 'StudyEye': '??', 'VisitType': 'V01', 'VisitId': 949138, 'DayInStudy': 0, 'NrBScans': 128, 'PatientId': 265724, 'ExternalSourceId': '67115192-25-GDQTLYOJFBUWHARGKILBRPWFA', 'Sex': 'O', 'VisitTime': 1489582087, 'ScanSpacing_mus': 47.24399948120117, 'Position': 'OS', 'Name': 'Macular Cube 512x128', 'Remark': '20210513_vibes_cirr_SD-S2_ARGOS-8801_20170315134807_gainoct_1.16_gainslo_1.95', 'SourceTypeId': 1, 'CreationTime': '2021-05-20_11:00:36', 'ScanFolder': '/optima/data/OPTDATA_VIBES/67115192-25-GDQTLYOJFBUWHARGKILBRPWFA+CRTAHULNZOVABDJOCRPJSSZWIMENVWCGMLVPZ+ESDGTA', 'oct_fn': '/optima/data/OPTDATA_VIBES/67115192-25-GDQTLYOJFBUWHARGKILBRPWFA+CRTAHULNZOVABDJOCRPJSSZWIMENVWCGMLVPZ+ESDGTA/bscan.dcm', 'width_mm': 6.0119, 'height_mm': 2.0019, 'aspect_ratio': 3.0031, 'pixel_dims': (512, 1024), 'layers_thicknesses': {'001': '41.36~17.41|18.53~2.60|26.73~2.60|21.95~2.69|19.48~1.37|59.39~3.71|19.91~1.27|38.50~1.54|16.75~1.25|7.61~0.86|244.34~8.52', ... }}
    '''

    def get_norm_thicknesses_from_str_flat(input_str, eps=1e-6):
        # Epsilon to prevent division by zero
        layer_thicknesses = []

        for part in input_str.split('|'):
            avg_str, std_str = part.split('~')
            avg = max(0.0, float(avg_str)) # Ensure no negative thickness
            std = float(std_str)

            # 1. Mean Thickness: Use log1p because layer thickness
            # in um can range from 0 to 400+. log helps compress this range.
            norm_avg = float(np.log1p(avg))

            # 2. Variability: Use a proper Coefficient of Variation (CV)
            # If avg is 0, CV should also be 0 (no layer = no variation).
            if avg > eps:
                norm_std = std / avg
            else:
                norm_std = 0.0

            layer_thicknesses.extend([norm_avg, norm_std])

        return layer_thicknesses

    def get_sample_info(fsid, sample_metadata):
        laterality_str = sample_metadata['Position']
        num_bscans = sample_metadata['NrBScans']
        device_str = sample_metadata['Vendor']
        # Normalized version for width.
        #   Stats: width_mm: min=2.77, max=11.55
        width_mm_original = sample_metadata['width_mm']
        width_mm = (width_mm_original - 2.77) / (11.55 - 2.77)
        laterality = 0 if laterality_str == 'OD' else 1
        # Option A:
        # bscan_rel_number = int(Path(path).stem) / (num_bscans - 1)
        # Option B:
        # central_bscan = num_bscans // 2
        # bscan_number = int(Path(path).stem)
        # dist_to_center = abs(bscan_number - central_bscan) / central_bscan
        # Option C:
        central_bscan = (num_bscans - 1) / 2.0
        device = 0 if device_str == 'Heidelberg Engineering' else 1
        bscans_info = []
        for bscan_id, thicknesses_str in sample_metadata['layers_thicknesses'].items():
            bscan_number = int(bscan_id)
            dist_to_center = (bscan_number - central_bscan) / central_bscan
            dist_to_center = max(-1.0, min(1.0, dist_to_center))
            thicknesses = get_norm_thicknesses_from_str_flat(thicknesses_str)
            bscans_info.append((bscan_id, {
                'position': dist_to_center,
                'thicknesses': thicknesses,
            }))
        return {
            'laterality': laterality,
            'device': device,
            'width_mm': width_mm,
            'bscans': bscans_info,
        }

    training_databank = {
        # patient_id
        #   fileset_id
        #       relevant_metadata
        #       bscans_info
        #           bscan_id
        #               dist_to_center (normalized)
        #               thicknesses (normalized)
    }
    for k, v in tqdm(metadata.items()):
        patient = v['PatientId']
        if patient not in training_databank:
            training_databank[patient] = []
        sample_info = get_sample_info(k, v)
        training_databank[patient].append((k, sample_info))
        if args.debug:
            print(patient, training_databank[patient])

    # Remove patient level from training_databank and convert to list
    training_databank = list(training_databank.items())

    del metadata
    gc.collect()

    class ProgressFileWriter:
        def __init__(self, path):
            self.f = open(path, 'wb')
            self.pbar = tqdm(unit='B', unit_scale=True, desc="Saving")

        def write(self, data):
            self.f.write(data)
            self.pbar.update(len(data))

        def close(self):
            self.f.close()
            self.pbar.close()

    tgt_fn = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/training_databank.pkl'
    writer = ProgressFileWriter(tgt_fn)
    pickle.dump(training_databank, writer, protocol=4)
    writer.close()


def check_data(args):
    print("Checking the created training databank...")
    fn = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/training_databank.pkl'
    with open(fn, 'rb') as f:
        training_databank = pickle.load(f)

    print(f"Total patients: {len(training_databank)}")
    total_volumes = 0
    total_bscans = 0
    for patient_id, samples in training_databank:
        total_volumes += len(samples)
        for fileset_id, sample_info in samples:
            num_bscans = len(sample_info['bscans'])
            total_bscans += num_bscans
    print(f"Total volumes: {total_volumes}")
    print(f"Total bscans: {total_bscans}")

    # Print first volume
    patient_id, samples = training_databank[0]
    print(f"Patient ID: {patient_id}")
    fsid, volume = samples[0]
    print(f"Fileset ID: {fsid}")
    print(volume)


def mini_dataset(args):
    print("Creating a mini dataset for quick testing...")
    fn = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/training_databank.pkl'
    with open(fn, 'rb') as f:
        training_databank = pickle.load(f)
    mini_training_databank = []
    # Add only the first 5 patients
    for i in range(5):
        mini_training_databank.append(training_databank[i])
    tgt_fn = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/mini_training_databank.pkl'
    with open(tgt_fn, 'wb') as f:
        pickle.dump(mini_training_databank, f, protocol=4)

def copy_mini_dataset(args):
    src_dir = Path('/mnt/Data/SSHFS/msc_raid/FullVIBES-v2/v2_bscanlayermap')
    mtdb_fn = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/mini_training_databank.pkl'
    tgt_dir = Path('/home/morano/Bookmarks/Data/FullVIBES-v2/v2_new_bscanlayermap')
    with open(mtdb_fn, 'rb') as f:
        mini_training_databank = pickle.load(f)
    for patient_id, samples in mini_training_databank:
        for fsid, sample_info in samples:
            for bscan_id, bscan_info in sample_info['bscans']:
                part = fsid.split('-')[2]
                subdir_1 = part[:2]
                subdir_2 = part[2:3]
                src_fn = src_dir / subdir_1 / subdir_2 / fsid / f'{bscan_id}.png'
                tgt_fn = tgt_dir / subdir_1 / subdir_2 / fsid / f'{bscan_id}.png'
                tgt_fn.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_fn, tgt_fn)
    print("Mini dataset copied successfully.")


def translate_to_fast_format(args):
    input_pkl_path = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/training_databank.pkl'
    output_joblib_path = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/training_databank.joblib'
    # input_pkl_path = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/mini_training_databank.pkl'
    # output_joblib_path = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/mini_training_databank.joblib'

    print(f"Loading old format from {input_pkl_path}...")
    with open(input_pkl_path, 'rb') as f:
        old_db = pickle.load(f)

    # 1. Count totals for pre-allocation
    print("Calculating totals...")
    total_bscans = 0
    total_volumes = 0
    total_patients = len(old_db)
    for _, samples in old_db:
        total_volumes += len(samples)
        for _, vol_info in samples:
            total_bscans += len(vol_info['bscans'])

    print(f"Allocating: {total_patients} Patients, {total_volumes} Volumes, {total_bscans} B-scans...")

    # Heavy Data Arrays (Contiguous Memory)
    all_thicknesses = np.zeros((total_bscans, 22), dtype=np.float32)
    all_positions = np.zeros((total_bscans, 1), dtype=np.float32)
    all_bscan_ids = np.zeros((total_bscans,), dtype=np.int32)
    all_vol_meta = np.zeros((total_volumes, 3), dtype=np.float32)

    # Hierarchical Indexing
    patient_list = [] # Stores: {'id': p_id, 'vol_range': (start, end)}
    volume_list = []  # Stores: {'fsid': fsid, 'bscan_range': (start, end)}

    curr_bscan_idx = 0
    curr_vol_idx = 0

    print("Flattening data...")
    for patient_id, samples in tqdm(old_db):
        start_vol_ptr = curr_vol_idx

        for fsid, vol_info in samples:
            # 1. Store Volume Metadata [lat, dev, width]
            all_vol_meta[curr_vol_idx] = [
                vol_info['laterality'],
                vol_info['device'],
                vol_info['width_mm']
            ]

            # 2. Store B-scan Data
            start_bscan_ptr = curr_bscan_idx
            # Order bscans by first item in tuple
            bscans = sorted(vol_info['bscans'], key=lambda x: int(x[0]))
            for b_id, bscan_data in bscans:
                all_positions[curr_bscan_idx] = bscan_data['position']
                all_thicknesses[curr_bscan_idx] = bscan_data['thicknesses']
                all_bscan_ids[curr_bscan_idx] = int(b_id) # "001" -> 1
                curr_bscan_idx += 1

            # 3. Record volume range
            volume_list.append((fsid, start_bscan_ptr, curr_bscan_idx))
            curr_vol_idx += 1
            vol_info.pop('bscans') # Memory cleanup

        # 4. Record patient range
        patient_list.append((patient_id, start_vol_ptr, curr_vol_idx))

    del old_db
    gc.collect()

    print(f"Saving to {output_joblib_path}...")
    output = {
        'patients': patient_list,
        'volumes': volume_list,
        'thicknesses': all_thicknesses,
        'positions': all_positions,
        'bscan_ids': all_bscan_ids,
        'vol_meta': all_vol_meta
    }

    joblib.dump(output, output_joblib_path)
    print("Done!")



class DatabankNavigator:
    def __init__(self, path):
        print(f"Loading databank navigator from {path}...")
        # CRITICAL: mmap_mode='r' makes the huge arrays shareable across processes
        data = joblib.load(path, mmap_mode='r')

        self.patients = data['patients']      # List of (id, start, end)
        self.volumes = data['volumes']        # List of (fsid, start, end)
        self.thicknesses = data['thicknesses']
        self.positions = data['positions']
        self.bscan_ids = data['bscan_ids']
        self.vol_meta = data['vol_meta']

        # O(1) Lookups mapping ID to the index in our tuple lists
        self.pat_id_to_idx = {p[0]: i for i, p in enumerate(self.patients)}
        self.fsid_to_vol_idx = {v[0]: i for i, v in enumerate(self.volumes)}

    def get_fsids_by_patient(self, patient_id):
        if patient_id not in self.pat_id_to_idx:
            return []

        p_idx = self.pat_id_to_idx[patient_id]
        _, v_start, v_end = self.patients[p_idx]

        return [self.volumes[i][0] for i in range(v_start, v_end)]

    def get_full_volume_by_fsid(self, fsid, format_as_str=True):
        if fsid not in self.fsid_to_vol_idx:
            return None

        v_idx = self.fsid_to_vol_idx[fsid]
        _, b_start, b_end = self.volumes[v_idx]

        raw_ids = self.bscan_ids[b_start:b_end]
        bscan_ids = [f"{int(x):03d}" for x in raw_ids] if format_as_str else raw_ids

        return {
            'fsid': fsid,
            'meta': self.vol_meta[v_idx],
            'thicknesses': self.thicknesses[b_start:b_end],
            'positions': self.positions[b_start:b_end],
            'bscan_ids': bscan_ids,
            'count': b_end - b_start
        }

    def get_bscan_info(self, fsid, bscan_id):
        if fsid not in self.fsid_to_vol_idx:
            return None

        v_idx = self.fsid_to_vol_idx[fsid]
        _, b_start, b_end = self.volumes[v_idx]

        # Slice the ID array and use binary search (O(log N))
        vol_bscan_ids = self.bscan_ids[b_start:b_end]
        target_id = int(bscan_id)
        offset = np.searchsorted(vol_bscan_ids, target_id)

        if offset >= len(vol_bscan_ids) or vol_bscan_ids[offset] != target_id:
            return None

        global_idx = b_start + offset
        meta = self.vol_meta[v_idx]

        return {
            'thickness': self.thicknesses[global_idx],
            'pos': self.positions[global_idx],
            'bscan_id': f"{target_id:03d}",
            'laterality': meta[0],
            'device': meta[1],
            'width_mm': meta[2],
            'fsid': fsid
        }

    def get_random_bscan_from_patient(self, patient_id):
        p_idx = self.pat_id_to_idx[patient_id]
        _, v_start, v_end = self.patients[p_idx]

        v_idx = np.random.randint(v_start, v_end)
        _, b_start, b_end = self.volumes[v_idx]

        b_idx = np.random.randint(b_start, b_end)
        meta = self.vol_meta[v_idx]

        return {
            'thickness': self.thicknesses[b_idx],
            'pos': self.positions[b_idx],
            'bscan_id': f"{int(self.bscan_ids[b_idx]):03d}",
            'laterality': meta[0],
            'device': meta[1],
            'width_mm': meta[2],
            'fsid': self.volumes[v_idx][0]
        }


def check_new_data(args):
    databank_fn = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/training_databank.pkl'
    with open(databank_fn, 'rb') as f:
        mini_db = pickle.load(f)

    print('Loading the new data bank navigator...')
    # Pro-tip: Always use mmap_mode='r' for the navigator to save RAM
    navigator = DatabankNavigator('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/training_databank.joblib')

    for patient_id, samples in tqdm(mini_db):
        # print(f"Verifying Patient ID: {patient_id}")
        for fsid, sample_info in samples:
            # 1. Get data from the new navigator
            volume_info = navigator.get_full_volume_by_fsid(fsid)

            # 2. Sort the OLD bscans to match the sorted order in the translation script
            # In your old pkl, sample_info is the dict containing 'bscans', 'laterality', etc.
            old_bscans_sorted = sorted(sample_info['bscans'], key=lambda x: int(x[0]))

            # Add this inside the 'for fsid, sample_info in samples:' loop
            assert np.isclose(volume_info['meta'][0], sample_info['laterality']), f"Lat mismatch for {fsid}"
            assert np.isclose(volume_info['meta'][1], sample_info['device']), f"Dev mismatch for {fsid}"

            # 4. Deep verification of every B-scan
            for i, (b_id_old, b_data_old) in enumerate(old_bscans_sorted):
                # Check IDs (converted to int/string match)
                # print(f"Checking {fsid} B-scan {b_id_old}...")
                assert int(b_id_old) == int(volume_info['bscan_ids'][i]), \
                    f"ID mismatch at index {i}: {b_id_old} vs {volume_info['bscan_ids'][i]}"

                # Check Positions
                pos_old = b_data_old['position']
                pos_new = volume_info['positions'][i][0]
                assert np.isclose(pos_old, pos_new, atol=1e-5), \
                    f"Pos mismatch in {fsid} B-scan {b_id_old}: {pos_old} vs {pos_new}"

                # Check Thicknesses (the whole 22-length array)
                thick_old = b_data_old['thicknesses']
                thick_new = volume_info['thicknesses'][i]
                assert np.allclose(thick_old, thick_new, atol=1e-5), \
                    f"Thickness mismatch in {fsid} B-scan {b_id_old}"

    print("\n" + "="*30)
    print("SUCCESS: All data matches perfectly!")
    print("="*30)


def get_bscan_by_fsid(fsid, bscan_id, old_db):
    info = {}
    for patient_id, samples in old_db:
        for fsid_sample, sample_info in samples:
            if fsid_sample == fsid:
                info['laterality'] = sample_info['laterality']
                info['device'] = sample_info['device']
                info['width_mm'] = sample_info['width_mm']
                for bscan_id_sample, bscan_info in sample_info['bscans']:
                    if bscan_id_sample == bscan_id:
                        info['position'] = bscan_info['position']
                        info['thicknesses'] = bscan_info['thicknesses']
                        return info
    return None


def manual_check(args):
    fsid = '98540682-25-AILGABPJTSDSYAWNDTQVAKXBH+FJBGEVOCHHJWDIAMRHOTIGKIFZJQIOFSJMWNK+HVYIHJ'
    bscan_id = '004'
    mini_databank_fn = '/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/mini_training_databank.pkl'
    with open(mini_databank_fn, 'rb') as f:
        mini_db = pickle.load(f)
    navigator = DatabankNavigator('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/training_databank.joblib')
    bscan_info = navigator.get_bscan_info(fsid, bscan_id)
    # print(navigator.patients)
    print(f"Volume Meta: {bscan_info}")
    # Get the bscan info for the specific bscan_id
    bscan_idx = None
    vol_info_old = get_bscan_by_fsid(fsid, bscan_id, mini_db)
    print(f"Old Volume Meta {vol_info_old}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    globals()[args.function](args)
