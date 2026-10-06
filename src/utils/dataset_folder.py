import os
import os.path
from pathlib import Path
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, cast, Union
import gzip
import json
import copy
from itertools import chain
import pickle
import joblib

import torch
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import io
from skimage import transform as sk_transform
from torchvision.datasets.vision import VisionDataset

from multimae.parts.multimae_utils import pair
from utils.data_utils import pil_loader



def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_nonclass_dataset(
        directory: str,
        args,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    print(f"Making non-class dataset from {directory}")
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    target_dir = directory
    assert os.path.isdir(target_dir), target_dir
    for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                # Only include files that are in the fsids list
                if args.fsids is not None:
                    if Path(path).stem in args.fsids:
                        item = path, 0
                        instances.append(item)
                else:
                    item = path, 0
                    instances.append(item)
    return instances


def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt logs)
            both extensions and is_valid_file should not be passed.

    Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 logs in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def normalize_to_0_1(sample: np.ndarray) -> np.ndarray:
    """Normalize to 0-1 range data with any range (positive values)."""
    return (sample - np.min(sample)) / (np.max(sample) - np.min(sample))


# DEPRECATED
class MultiTaskDatasetFolder_OLD(VisionDataset):
    """A generic multi-task dataset loader where the samples are arranged in this way: ::

        root/task_a/class_x/xxx.ext
        root/task_a/class_y/xxy.ext
        root/task_a/class_z/xxz.ext

        root/task_b/class_x/xxx.ext
        root/task_b/class_y/xxy.ext
        root/task_b/class_z/xxz.ext

    Args:
        root (string): Root directory path.
        tasks (list): List of tasks as strings
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt logs)
            both extensions and is_valid_file should not be passed.

    Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            tasks: List[str],
            loader: Callable[[str], Any],
            args,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None,
    ) -> None:
        print('MultiTaskDatasetFolder_OLD is DEPRECATED')
        exit(0)
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.tasks = tasks
        self.args = args
        assert args is not None
        # classes, class_to_idx = self._find_classes(os.path.join(self.root, self.tasks[0]))

        prefixes = {} if prefixes is None else prefixes
        prefixes.update({task: '' for task in tasks if task not in prefixes})

        samples = {
            # task: make_dataset(os.path.join(self.root, f'{prefixes[task]}{task}'), class_to_idx, extensions, is_valid_file)
            task: make_nonclass_dataset(os.path.join(self.root, f'{prefixes[task]}{task}'), args, extensions, is_valid_file)
            for task in self.tasks
        }

        for task, task_samples in samples.items():
            if len(task_samples) == 0:
                msg = "Found 0 logs in subfolders of: {}\n".format(os.path.join(self.root, task))
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        # self.classes = classes
        # self.class_to_idx = class_to_idx
        self.samples = samples
        # self.targets = [s[1] for s in list(samples.values())[0]]

        # Select random subset of dataset if so specified
        if isinstance(max_images, int):
            total_samples = len(list(self.samples.values())[0])
            np.random.seed(0)
            permutation = np.random.permutation(total_samples)
            for task in samples:
                self.samples[task] = [self.samples[task][i] for i in permutation][:max_images]

        self.cache = {}
        self.ids = {}

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        target = None
        if index in self.cache:
            sample_dict, target = deepcopy(self.cache[index])
        else:
            sample_dict = {}
            # slice = None
            start_i = None
            start_j = None
            random_idx = None
            for task in self.tasks:
                path, target = self.samples[task][index]
                if path.endswith('.npy') or path.endswith('.npz'):
                    if task == 'layermaps':
                        sample = np.load(path)['layer_maps'].astype(int)
                    elif task == 'bscanlayermap':
                        sample = np.load(path).astype(int)
                    else:
                        sample = np.load(path).astype(np.float32) / 255.0
                    if sample.ndim == 2 and self.args.three_d:
                        if task == 'slo':
                            sample = sample[:, np.newaxis, :]
                        elif task == 'bscan':
                            sample = sample[np.newaxis, :, :]
                        else:
                            raise ValueError(f"Unknown task {task}")
                    elif sample.ndim == 3:
                        if self.args.input_size[task] != sample.shape:
                            # print(f"Resizing {task} from {sample.shape} to {self.args.input_size[task]}")
                            sample = resize(
                                sample,
                                self.args.input_size[task],
                                anti_aliasing=False,
                                preserve_range=True,
                                order=1
                            ).astype(sample.dtype)
                else:
                    sample = io.imread(path)
                    sample_type = sample.dtype
                    # target_size = None
                    # for _, value in self.args.input_size.items():
                    #     target_size = value
                    # if sample.shape != target_size:
                    #     if 'semseg' in task:
                    #         order = 0
                    #     else:
                    #         order = 1
                    #     sample = resize(
                    #         sample,
                    #         target_size,
                    #         anti_aliasing=False,
                    #         preserve_range=True,
                    #         order=order
                    #     ).astype(sample_type)
                    if 'semseg' in task:
                        # TODO: HACK: Remove this
                        # Convert to 0-indexed labels
                        for label, i in self.args.mapping.items():
                            sample[sample == label] = i
                        # Convert to one-hot encoding
                        # oh_sample = np.zeros((self.args.num_classes, *sample.shape), dtype=np.float32)
                        # for i in range(self.args.num_classes):
                        #     oh_sample[i, sample == i] = 1.0
                        # sample = oh_sample
                    else:
                        sample = normalize_to_0_1(sample)
                    # print(sample.shape, sample.min(), sample.max())
                    # sample = Image.fromarray(sample)
                    # else:
                    #     sample = pil_loader(path, convert_rgb=(task=='rgb'))
                # sample = sample.convert('P') if 'semseg' in task else sample
                sample_dict[task] = sample
                if index not in self.ids:
                    self.ids[index] = Path(path).stem
            # self.cache[index] = deepcopy((sample_dict, target))

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # for task in sample_dict:
        #     print(task, sample_dict[task].shape, sample_dict[task].min(), sample_dict[task].max())


        return sample_dict, target, self.ids[index]

    def __len__(self) -> int:
        return len(list(self.samples.values())[0])



class MultiTaskDatasetFolderOld(VisionDataset):
    """A generic multi-task dataset loader where the samples are arranged in this way: ::

        root/task_a/class_x/xxx.ext
        root/task_a/class_y/xxy.ext
        root/task_a/class_z/xxz.ext

        root/task_b/class_x/xxx.ext
        root/task_b/class_y/xxy.ext
        root/task_b/class_z/xxz.ext

    Args:
        root (string): Root directory path.
        tasks (list): List of tasks as strings
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt logs)
            both extensions and is_valid_file should not be passed.

    Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            tasks: List[str],
            loader: Callable[[str], Any],
            args,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert args is not None
        self.tasks = tasks
        self.args = args
        self.loader = loader
        self.extensions = extensions

        with gzip.open(args.filelist.format(version=args.version), 'rt', encoding='utf-8') as f:
            self.samples = json.load(f)
            print('🚨 Number of files in filelist:', len(self.samples))
            # Print first 5 elements and exit
            # for i, sample in enumerate(self.samples[:5]):
            #     print(i, sample)

        self.ids = {}

    # def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
    #     # """
    #     # Finds the class folders in a dataset.

    #     # Args:
    #     #     dir (string): Root directory path.

    #     # Returns:
    #     #     tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    #     # Ensures:
    #     #     No class is a subdirectory of another.
    #     # """
    #     # classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    #     # classes.sort()
    #     # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    #     # return classes, class_to_idx
    #     raise NotImplementedError

    def read_item(self, subpath: Union[Path, str]) -> Tuple[dict, dict]:
        sample_dict = {}
        path = Path(self.root, subpath)
        sample_all = np.array(Image.open(path))
        # 0: B-scan, 1: layers, 2: lesions
        for task in self.tasks:
            if task == 'bscan':
                sample = sample_all[:, :, 0].astype(np.float32) / 255.0
            elif task == 'bscanlayermap':
                sample = sample_all[:, :, 1].astype(int)
                # IMPORTANT: Treat the sclera as background
                sample[sample == 12] = 0
                # print('sample', sample.shape, np.unique(sample))
                sample_les = sample_all[:, :, 2].astype(int) - 1
                # print('sample_les', sample_les.shape, np.unique(sample_les))
                sample[sample_les > 0] = sample_les[sample_les > 0]
            sample_dict[task] = sample
        fsid = Path(path).parent.stem
        identifier = fsid + '_' + Path(path).stem
        # Cache metadata info
        if identifier not in self.ids:
            laterality_str = self.args.metadata[fsid]['Position']
            num_bscans = self.args.metadata[fsid]['NrBScans']
            device_str = self.args.metadata[fsid]['Vendor']
            # Normalized version for width.
            #   Stats: width_mm: min=2.77, max=11.55
            width_mm_original = self.args.metadata[fsid]['width_mm']
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
            bscan_number = int(Path(path).stem)
            dist_to_center = (bscan_number - central_bscan) / central_bscan
            dist_to_center = max(-1.0, min(1.0, dist_to_center))
            device = 0 if device_str == 'Heidelberg Engineering' else 1
            self.ids[identifier] = {
                'fsid': fsid,
                'laterality': laterality,
                'device': device,
                'width_mm': np.array([width_mm], dtype=np.float32),
                'position': np.array([dist_to_center], dtype=np.float32),
            }
        return sample_dict, copy.deepcopy(self.ids[identifier])

    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # target = None
        subpath = self.samples[index]
        sample_dict, info = self.read_item(subpath)
        if self.transform is not None:
            # Add laterality to sample_dict
            sample_dict['laterality'] = info['laterality']
            sample_dict = self.transform(sample_dict)
            # Remove from sample_dict and add to info
            info['laterality'] = sample_dict.pop('laterality')
            if 'lesion_presence' in sample_dict:
                 info['lesion_presence'] = sample_dict.pop('lesion_presence')
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return sample_dict, info

    def __len__(self) -> int:
        return len(self.samples)


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
            'laterality': int(self.vol_meta[v_idx][0]),
            'device': int(self.vol_meta[v_idx][1]),
            'width_mm': self.vol_meta[v_idx][2],
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
            'thicknesses': self.thicknesses[global_idx],
            'position': self.positions[global_idx],
            'bscan_id': f"{target_id:03d}",
            'laterality': int(meta[0]),
            'device': int(meta[1]),
            'width_mm': meta[2],
            'fsid': fsid
        }

    def get_random_bscan_from_patient_idx(self, patient_idx):
        _, v_start, v_end = self.patients[patient_idx]

        v_idx = np.random.randint(v_start, v_end)
        _, b_start, b_end = self.volumes[v_idx]

        b_idx = np.random.randint(b_start, b_end)
        meta = self.vol_meta[v_idx]

        return {
            'thicknesses': self.thicknesses[b_idx],
            'position': self.positions[b_idx],
            'bscan_id': f"{int(self.bscan_ids[b_idx]):03d}",
            'laterality': int(meta[0]),
            'device': int(meta[1]),
            'width_mm': meta[2],
            'fsid': self.volumes[v_idx][0]
        }

    def get_random_volume_from_patient_idx(self, patient_idx):
        _, v_start, v_end = self.patients[patient_idx]
        v_idx = np.random.randint(v_start, v_end)
        return self.get_full_volume_by_fsid(self.volumes[v_idx][0])


class MultiTaskPatientBalancedDatasetFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        tasks: List[str],
        loader: Callable[[str], Any],
        args,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform)
        assert args is not None
        self.tasks = tasks
        self.args = args
        self.loader = loader

        self.navigator = DatabankNavigator(args.filelist)

        print('🚨 Number of patients in filelist:', len(self.navigator.patients))
        print('🚨 Number of B-scans in filelist:', len(self.navigator.bscan_ids))

    def read_item(self, subpath: Union[Path, str]) -> dict:
        sample_dict = {}
        path = Path(self.root, subpath)
        sample_all = np.array(Image.open(path))
        # 0: B-scan, 1: layers, 2: lesions
        for task in self.tasks:
            if task == 'bscan':
                sample = sample_all[:, :, 0].astype(np.float32) / 255.0
            elif task == 'bscanlayermap':
                sample = sample_all[:, :, 1].astype(int)
                # IMPORTANT: Treat the sclera as background
                sample[sample == 12] = 0
                # print('sample', sample.shape, np.unique(sample))
                sample_les = sample_all[:, :, 2].astype(int) - 1
                # print('sample_les', sample_les.shape, np.unique(sample_les))
                sample[sample_les > 0] = sample_les[sample_les > 0]
            sample_dict[task] = sample
        return sample_dict

    def get_volume_info(self, index: int):
        return self.navigator.get_random_bscan_from_patient_idx(index)

    def __getitem__(self, index: int) -> Tuple:
        volume = self.get_volume_info(index)
        thicknesses = torch.tensor(volume['thicknesses'])
        thicknesses[1::2] = torch.clamp(thicknesses[1::2], max=2.0) * 3

        fsid = volume['fsid']
        bscan_id = volume['bscan_id']
        info = {
            'fsid': fsid,
            # Convert laterality and device to int
            'laterality': volume['laterality'],
            'device': volume['device'],
            'width_mm': volume['width_mm'],
            'position': volume['position'],
            'thicknesses': thicknesses,
        }

        part = fsid.split('-')[2]
        subdir_1 = part[:2]
        subdir_2 = part[2:3]
        subpath = Path(self.root) / subdir_1 / subdir_2 / fsid / f'{bscan_id}.png'
        sample_dict = self.read_item(subpath)

        if self.transform is not None:
            # Add laterality to sample_dict
            sample_dict['laterality'] = info['laterality']
            sample_dict = self.transform(sample_dict)
            # Remove from sample_dict and add to info
            info['laterality'] = sample_dict.pop('laterality')
            if 'lesion_presence' in sample_dict:
                 info['lesion_presence'] = sample_dict.pop('lesion_presence')

        return sample_dict, info

    def __len__(self) -> int:
        return len(self.navigator.patients)


class MultiTaskDatasetFolderRandom(MultiTaskPatientBalancedDatasetFolder):
    """Same as MultiTaskPatientBalancedDatasetFolder but with no
      patient-level balancing (i.e. just random sampling across all
      volumes).
    """

    def get_volume_info(self, index: int):
        # NOTE: Ignore index, just sample randomly from the navigator
        rand_patient = np.random.randint(0, len(self.navigator.patients))
        return self.navigator.get_random_bscan_from_patient_idx(rand_patient)


class MultiTaskDatasetFolder(MultiTaskPatientBalancedDatasetFolder):
    """Same as MultiTaskPatientBalancedDatasetFolder but with no
    patient-level balancing. Normal epochs over all B-scans.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Precompute volume start positions for O(log N) lookup
        self._vol_starts = np.array([v[1] for v in self.navigator.volumes])

    def get_volume_info(self, index: int):
        # Find which volume this global B-scan index belongs to
        v_idx = int(np.searchsorted(self._vol_starts, index, side='right') - 1)
        fsid, b_start, b_end = self.navigator.volumes[v_idx]
        meta = self.navigator.vol_meta[v_idx]
        bscan_id = int(self.navigator.bscan_ids[index])
        return {
            'thicknesses': self.navigator.thicknesses[index],
            'position': self.navigator.positions[index],
            'bscan_id': f"{bscan_id:03d}",
            'laterality': int(meta[0]),
            'device': int(meta[1]),
            'width_mm': meta[2],
            'fsid': fsid,
        }

    def __len__(self) -> int:
        return len(self.navigator.bscan_ids)


class MultiTaskConstrastiveDatasetFolder(MultiTaskPatientBalancedDatasetFolder):
    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        volume = self.navigator.get_random_volume_from_patient_idx(index)
        fsid = volume['fsid']
        part = fsid.split('-')[2]
        subdir_1 = part[:2]
        subdir_2 = part[2:3]
        subpath_fsid = Path(self.root) / subdir_1 / subdir_2 / fsid
        # NOTE: Get neighboring slices if the image as the two views if
        #   it has at least 19 slices, otherwise just get the same.
        num_bscans = len(volume['bscan_ids'])
        if num_bscans > 15:
            margin = num_bscans // 10
            rand_idx = np.random.randint(margin, num_bscans - margin)
            possible_offsets = chain(range(-margin, 0), range(1, margin + 1))
            offset = random.choice(list(possible_offsets))
            bscan_id_1 = volume['bscan_ids'][rand_idx]
            bscan_id_2 = volume['bscan_ids'][rand_idx + offset]
            subpath_1 = subpath_fsid / f'{bscan_id_1}.png'
            subpath_2 = subpath_fsid / f'{bscan_id_2}.png'
            # print(f"Index {index}: num_images={num_images}, rand_idx={rand_idx}, offset={offset}, rand_idx+offset={rand_idx + offset}")
            # Infos should have laterality, position, and thicknesses.
            info_1 = {
                'laterality': volume['laterality'],
                'position': volume['positions'][rand_idx],
                'thicknesses': volume['thicknesses'][rand_idx],
            }
            info_2 = {
                'laterality': volume['laterality'],
                'position': volume['positions'][rand_idx + offset],
                'thicknesses': volume['thicknesses'][rand_idx + offset],
            }
            sample_dict_1  = self.read_item(subpath_1)
            sample_dict_2 = self.read_item(subpath_2)
        else:
            rand_idx = np.random.randint(0, num_bscans)
            bscan_id = volume['bscan_ids'][rand_idx]
            subpath = subpath_fsid / f'{bscan_id}.png'
            sample_dict = self.read_item(subpath)
            sample_dict_1 = copy.deepcopy(sample_dict)
            sample_dict_2 = copy.deepcopy(sample_dict)
            info_1 = {
                'laterality': volume['laterality'],
                'position': volume['positions'][rand_idx],
                'thicknesses': volume['thicknesses'][rand_idx],
            }
            info_2 = copy.deepcopy(info_1)
        sample_dict_1['laterality'] = info_1['laterality']
        sample_dict_2['laterality'] = info_2['laterality']
        assert self.transform is not None
        # Add laterality to sample_dict
        sample_dict_1 = self.transform(sample_dict_1)
        sample_dict_2 = self.transform(sample_dict_2)
        # Remove from sample_dict and add to info
        info_1['laterality'] = sample_dict_1.pop('laterality')
        info_2['laterality'] = sample_dict_2.pop('laterality')
        info_1['lesion_presence'] = sample_dict_1.pop('lesion_presence')
        info_2['lesion_presence'] = sample_dict_2.pop('lesion_presence')

        # 1. Merge image-based tasks (bscan, bscanlayermap, etc.)
        merged_sample = {}
        for task in sample_dict_1.keys():
            # Creates a tensor of shape [2, C, H, W]
            merged_sample[task] = torch.stack([
                torch.as_tensor(sample_dict_1[task]),
                torch.as_tensor(sample_dict_2[task])
            ], dim=0)

        # 2. Merge metadata in info
        # We keep fsid as a single string (collator handles list of strings)
        # We stack the rest so they match the doubled batch size
        merged_info = {
            'fsid': fsid,
            'laterality': torch.tensor([info_1['laterality'], info_2['laterality']]),
            'device': torch.tensor([volume['device'], volume['device']]),
            'position': torch.stack([
                torch.as_tensor(info_1['position']),
                torch.as_tensor(info_2['position'])
            ], dim=0),
            'lesion_presence': torch.stack([
                torch.as_tensor(info_1['lesion_presence']),
                torch.as_tensor(info_2['lesion_presence'])
            ], dim=0),
            'width_mm': torch.tensor([volume['width_mm'], volume['width_mm']]),
            'thicknesses': torch.stack([
                torch.as_tensor(info_1['thicknesses']),
                torch.as_tensor(info_2['thicknesses'])
            ], dim=0),
        }

        return merged_sample, merged_info


class MultiTaskConstrastiveDatasetFolderOld(MultiTaskDatasetFolderOld):
    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        imgs = sorted(self.samples[index])
        # NOTE: Get neighboring slices if the image as the two views if
        #   it has at least 19 slices, otherwise just get the same.
        num_images = len(imgs)
        if num_images > 15:
            margin = num_images // 10
            rand_idx = random.randint(margin, num_images - margin - 1)
            possible_offsets = chain(range(-margin, 0), range(1, margin + 1))
            offset = random.choice(list(possible_offsets))
            subpath_1 = imgs[rand_idx]
            subpath_2 = imgs[rand_idx + offset]
            # print(f"Index {index}: num_images={num_images}, rand_idx={rand_idx}, offset={offset}, rand_idx+offset={rand_idx + offset}")
            sample_dict_1, info_1 = self.read_item(subpath_1)
            sample_dict_2, info_2 = self.read_item(subpath_2)
        else:
            subpath = random.choice(imgs)
            sample_dict, info = self.read_item(subpath)
            sample_dict_1 = copy.deepcopy(sample_dict)
            sample_dict_2 = copy.deepcopy(sample_dict)
            info_1 = copy.deepcopy(info)
            info_2 = copy.deepcopy(info)
        sample_dict_1['laterality'] = info_1['laterality']
        sample_dict_2['laterality'] = info_2['laterality']
        assert self.transform is not None
        # Add laterality to sample_dict
        sample_dict_1 = self.transform(sample_dict_1)
        sample_dict_2 = self.transform(sample_dict_2)
        # Remove from sample_dict and add to info
        info_1['laterality'] = sample_dict_1.pop('laterality')
        info_2['laterality'] = sample_dict_2.pop('laterality')
        info_1['lesion_presence'] = sample_dict_1.pop('lesion_presence')
        info_2['lesion_presence'] = sample_dict_2.pop('lesion_presence')

        # 1. Merge image-based tasks (bscan, bscanlayermap, etc.)
        merged_sample = {}
        for task in sample_dict_1.keys():
            # Creates a tensor of shape [2, C, H, W]
            merged_sample[task] = torch.stack([
                torch.as_tensor(sample_dict_1[task]),
                torch.as_tensor(sample_dict_2[task])
            ], dim=0)

        # 2. Merge metadata in info
        # We keep fsid as a single string (collator handles list of strings)
        # We stack the rest so they match the doubled batch size
        merged_info = {
            'fsid': info_1['fsid'],
            'laterality': torch.tensor([info_1['laterality'], info_2['laterality']]),
            'device': torch.tensor([info_1['device'], info_2['device']]),
            'position': torch.stack([
                torch.as_tensor(info_1['position']),
                torch.as_tensor(info_2['position'])
            ], dim=0),
            'lesion_presence': torch.stack([
                torch.as_tensor(info_1['lesion_presence']),
                torch.as_tensor(info_2['lesion_presence'])
            ], dim=0),
            'width_mm': torch.tensor([info_1['width_mm'], info_2['width_mm']]),
        }

        return merged_sample, merged_info




# IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.jpx', '.npy', '.npz')




# # TODO: specify the return type
# def accimage_loader(path: str) -> Any:
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


# def default_loader(path: str) -> Any:
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)


# class ImageFolder(DatasetFolder):
#     """A generic data loader where the images are arranged in this way: ::

#         root/dog/xxx.png
#         root/dog/xxy.png
#         root/dog/xxz.png

#         root/cat/123.png
#         root/cat/nsdf3.png
#         root/cat/asd932_.png

#     Args:
#         root (string): Root directory path.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         loader (callable, optional): A function to load an image given its path.
#         is_valid_file (callable, optional): A function that takes path of an Image file
#             and check if the file is a valid file (used to check of corrupt logs)

#     Attributes:
#         classes (list): List of the class names sorted alphabetically.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         imgs (list): List of (image path, class_index) tuples
#     """

#     def __init__(
#             self,
#             root: str,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             loader: Callable[[str], Any] = default_loader,
#             is_valid_file: Optional[Callable[[str], bool]] = None,
#     ):
#         super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
#                                           transform=transform,
#                                           target_transform=target_transform,
#                                           is_valid_file=is_valid_file)
#         self.imgs = self.samples

# class MultiTaskImageFolder(MultiTaskDatasetFolder):
#     """A generic multi-task dataset loader where the images are arranged in this way: ::

#         root/task_a/class_x/xxx.ext
#         root/task_a/class_y/xxy.ext
#         root/task_a/class_z/xxz.ext

#         root/task_b/class_x/xxx.ext
#         root/task_b/class_y/xxy.ext
#         root/task_b/class_z/xxz.ext

#     Args:
#         root (string): Root directory path.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         loader (callable, optional): A function to load an image given its path.
#         is_valid_file (callable, optional): A function that takes path of an Image file
#             and check if the file is a valid file (used to check of corrupt logs)

#     Attributes:
#         classes (list): List of the class names sorted alphabetically.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         imgs (list): List of (image path, class_index) tuples
#     """

#     def __init__(
#             self,
#             root: str,
#             tasks: List[str],
#             args,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             loader: Callable[[str], Any] = pil_loader,
#             is_valid_file: Optional[Callable[[str], bool]] = None,
#             prefixes: Optional[Dict[str,str]] = None,
#             max_images: Optional[int] = None,
#     ):
#         super(MultiTaskImageFolder, self).__init__(
#             root, tasks, loader,
#             args=args,
#             extensions=IMG_EXTENSIONS if is_valid_file is None else None,
#             transform=transform,
#             target_transform=target_transform,
#             is_valid_file=is_valid_file,
#             prefixes=prefixes,
#             max_images=max_images,
#         )
#         self.imgs = self.samples


# class MultiTaskContrastiveImageFolder(MultiTaskConstrastiveDatasetFolder):
#     """A generic multi-task dataset loader where the images are arranged in this way: ::

#         root/task_a/class_x/xxx.ext
#         root/task_a/class_y/xxy.ext
#         root/task_a/class_z/xxz.ext

#         root/task_b/class_x/xxx.ext
#         root/task_b/class_y/xxy.ext
#         root/task_b/class_z/xxz.ext

#     Args:
#         root (string): Root directory path.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         loader (callable, optional): A function to load an image given its path.
#         is_valid_file (callable, optional): A function that takes path of an Image file
#             and check if the file is a valid file (used to check of corrupt logs)

#     Attributes:
#         classes (list): List of the class names sorted alphabetically.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         imgs (list): List of (image path, class_index) tuples
#     """

#     def __init__(
#             self,
#             root: str,
#             tasks: List[str],
#             args,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             loader: Callable[[str], Any] = pil_loader,
#             is_valid_file: Optional[Callable[[str], bool]] = None,
#             prefixes: Optional[Dict[str,str]] = None,
#             max_images: Optional[int] = None,
#     ):
#         super(MultiTaskContrastiveImageFolder, self).__init__(
#             root, tasks, loader,
#             args=args,
#             extensions=IMG_EXTENSIONS if is_valid_file is None else None,
#             transform=transform,
#             target_transform=target_transform,
#             is_valid_file=is_valid_file,
#             prefixes=prefixes,
#             max_images=max_images,
#         )
#         self.imgs = self.samples
