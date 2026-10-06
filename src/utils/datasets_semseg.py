from typing import Dict, Tuple

import numpy as np
import torch
from typing import Callable, List, Optional, cast
from functools import partial
from copy import deepcopy
import os
from pathlib import Path

from skimage import io
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset

from utils import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
    VISIONFM_DEFAULT_MEAN, VISIONFM_DEFAULT_STD,
    PAD_MASK_VALUE, SEG_IGNORE_INDEX
)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.jpx', '.npy', '.npz')

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_valid_file(x: str, extensions: Tuple[str, ...]) -> bool:
    return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))


def make_seg_dataset(
    directory: str,
    extensions: Optional[Tuple[str, ...]] = None,
) -> List[Tuple[str, int]]:
    print(f"Making non-class dataset from {directory}")
    instances = []
    directory = os.path.expanduser(directory)
    if extensions is not None:
        c_is_valid_file = partial(is_valid_file, extensions=extensions)
    else:
        c_is_valid_file = partial(is_valid_file, extensions=IMG_EXTENSIONS)
    target_dir = directory
    assert os.path.isdir(target_dir), target_dir
    for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if c_is_valid_file(path):
                item = path, 0
                instances.append(item)
    return instances


def normalize_to_0_1(sample: np.ndarray) -> np.ndarray:
    """Normalize to 0-1 range data with any range (positive values)."""
    return (sample - np.min(sample)) / (np.max(sample) - np.min(sample))


class MultiTaskDatasetFolder(VisionDataset):
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
        args,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        prefixes: Optional[Dict[str,str]] = None,
        max_images: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.tasks = tasks
        self.args = args
        assert args is not None

        prefixes = {} if prefixes is None else prefixes
        prefixes.update({task: '' for task in tasks if task not in prefixes})

        samples = {
            task: make_seg_dataset(
                os.path.join(self.root, f'{prefixes[task]}{task}'),
                extensions,
            )
            for task in self.tasks
        }

        for task, task_samples in samples.items():
            if len(task_samples) == 0:
                msg = "Found 0 logs in subfolders of: {}\n".format(os.path.join(self.root, task))
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)

        self.extensions = extensions

        self.samples = samples

        # Select random subset of dataset if so specified
        if isinstance(max_images, int):
            total_samples = len(list(self.samples.values())[0])
            np.random.seed(0)
            permutation = np.random.permutation(total_samples)
            for task in samples:
                self.samples[task] = [self.samples[task][i] for i in permutation][:max_images]

        self.cache = {}
        self.ids = {}

    def __getitem__(self, index: int) -> Tuple:
        """
        Returns:
            tuple: (sample, target, id)
        """
        target = None
        if index in self.cache:
            sample_dict, target = deepcopy(self.cache[index])
        else:
            sample_dict = {}
            for task in self.tasks:
                path, target = self.samples[task][index]
                sample = io.imread(path)
                if 'semseg' in task:
                    # Convert to 0-indexed labels
                    sample = np.vectorize(self.args.mapping.get)(sample)
                else:
                    sample = normalize_to_0_1(sample)
                sample_dict[task] = sample
                if index not in self.ids:
                    self.ids[index] = Path(path).stem
            # self.cache[index] = deepcopy((sample_dict, target))

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # for k, v in sample_dict.items():
        #     print(f"Task: {k}, shape: {v.shape}")

        return sample_dict, target, self.ids[index]

    def __len__(self) -> int:
        return len(list(self.samples.values())[0])


class MultiTaskImageFolder(MultiTaskDatasetFolder):
    def __init__(
            self,
            root: str,
            tasks: List[str],
            args,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None,
    ):
        super().__init__(
            root,
            tasks,
            args=args,
            extensions=IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
            prefixes=prefixes,
            max_images=max_images,
        )
        self.imgs = self.samples


# Import ToRGB from albumentations
class ToRGB(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(ToRGB, self).__init__(always_apply, p)

    def apply(self, img, **params):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        return img


class ToGray(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(ToGray, self).__init__(always_apply, p)

    def apply(self, img, **params):
        if img.ndim == 3 and img.shape[2] == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        return img


class ToRange(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0, range=(0, 255)):
        super(ToRange, self).__init__(always_apply, p)
        self.range = range

    def apply(self, img, **params):
        min_val, max_val = np.min(img), np.max(img)
        img = img * (self.range[1] - self.range[0]) / (max_val - min_val) + self.range[0]
        return img


# def to_tensor(data):
#     for key in data:
#         if key == 'image':
#             if len(data[key].shape) == 2:
#                 data[key] = np.expand_dims(data[key], axis=0)
#         data[key] = torch.tensor(data[key], dtype=torch.float32)
#         if key == 'semseg':
#             data[key] = data[key].float()
#         # print(key, data[key].shape, data[key].min(), data[key].max())
#     return data


def simple_transform(
    train: bool,
    additional_targets: Dict[str, str],
    input_size: int = 512,
    pad_value: Tuple[int, int, int] = (128, 128, 128),
    pad_mask_value: int = PAD_MASK_VALUE,
    norm: str = 'minmax',
):
    """Default transform for semantic segmentation, applied on all modalities

    During training:
        1. Random horizontal Flip
        2. Rescaling so that longest side matches input size
        3. Color jitter (for RGB-modality only)
        4. Large scale jitter (LSJ)
        5. Padding
        6. Random crop to given size
        7. Normalization with ImageNet mean and std dev

    During validation / test:
        1. Rescaling so that longest side matches given size
        2. Padding
        3. Normalization with ImageNet mean and std dev
    """

    norm_list = []
    if norm == 'imagenet':
        norm_list += [
            ToRGB(p=1),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            # ToGray(p=1),
        ]
    elif norm == 'visionfm':
        norm_list += [
            ToRGB(p=1),
            A.Normalize(mean=VISIONFM_DEFAULT_MEAN, std=VISIONFM_DEFAULT_STD),
        ]
    elif norm == 'sam':
        # Rescale everything to [0, 255]
        norm_list += [
            ToRGB(p=1),
            ToRange(range=(0, 255), p=1),
        ]
    elif norm == 'z-score':
        norm_list += [
            ToRGB(p=1),
            A.Normalize(mean=0, std=1),
        ]

    if train:
        init_size = input_size+(int(input_size*0.1))
        transform_list = [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=init_size, width=init_size, p=1),
            # A.LongestMaxSize(max_size=input_size+(int(input_size*0.1)), p=1),
            # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.5),  # Color jittering from MoCo-v3 / DINO
            # A.RandomScale(scale_limit=(0.1 - 1, 2.0 - 1), p=1),  # This is LSJ (0.1, 2.0)
            # A.PadIfNeeded(min_height=input_size, min_width=input_size,
            #               position=A.augmentations.PadIfNeeded.PositionType.TOP_LEFT,
            #               border_mode=cv2.BORDER_CONSTANT,
            #               value=pad_value, mask_value=pad_mask_value),
            A.RandomCrop(height=input_size, width=input_size, p=1),
        ]
        transform_list += norm_list

        transform_list += [
            # A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            # A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ]
        transform = A.Compose(transform_list, additional_targets=additional_targets)

    else:
        transform_list = [
            # A.LongestMaxSize(max_size=input_size, p=1),
            A.Resize(height=input_size, width=input_size, p=1),
            # A.PadIfNeeded(min_height=input_size, min_width=input_size,
            #               position=A.augmentations.PadIfNeeded.PositionType.TOP_LEFT,
            #               border_mode=cv2.BORDER_CONSTANT,
            #               value=pad_value, mask_value=pad_mask_value),
        ]
        transform_list += norm_list
        transform_list += [
            # A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            # A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ]
        transform = A.Compose(transform_list, additional_targets=additional_targets)

    return transform


class DataAugmentationForSemSeg(object):
    """Data transform / augmentation for semantic segmentation downstream tasks.
    """

    def __init__(self, transform, seg_num_classes, seg_ignore_index=SEG_IGNORE_INDEX,
                 seg_reduce_zero_label=False, key_to_replace='bscan'):

        self.transform = transform
        self.seg_num_classes = seg_num_classes
        self.seg_ignore_index = seg_ignore_index
        self.seg_reduce_zero_label = seg_reduce_zero_label
        self.key_to_replace = key_to_replace

    def seg_adapt_labels(self, img):
        pad_replace = self.seg_ignore_index
        img[img == PAD_MASK_VALUE] = pad_replace

        if self.seg_reduce_zero_label:
            img[img == 0] = self.seg_ignore_index
            img = img - 1
            img[img == self.seg_ignore_index - 1] = self.seg_ignore_index

        return img

    def __call__(self, task_dict):
        key_to_replace = self.key_to_replace
        task_dict['image'] = task_dict.pop(key_to_replace)
        # Convert to np.array
        task_dict = {k: np.array(v) for k, v in task_dict.items()}

        task_dict = self.transform(**task_dict)

        # And then replace it back to rgb
        task_dict[key_to_replace] = task_dict.pop('image')

        for task in task_dict:
            if task in [key_to_replace]:
                task_dict[task] = task_dict[task].to(torch.float)
            elif task in ['semseg']:
                img = task_dict[task].to(torch.long)
                # img = self.seg_adapt_labels(img)
                task_dict[task] = img
            elif task in ['pseudo_semseg']:
                # If it's pseudo-semseg, then it's an input modality and should therefore be resized
                img = task_dict[task]
                img = F.interpolate(img[None,None,:,:], scale_factor=0.25, mode='nearest').long()[0,0]
                task_dict[task] = img

        return task_dict


def build_semseg_dataset(args, data_path, transform, max_images=None):
    transform = DataAugmentationForSemSeg(
        transform=transform,
        seg_num_classes=args.num_classes,
        seg_reduce_zero_label=args.seg_reduce_zero_label,
        key_to_replace=args.in_domains[0],
    )
    prefixes = None
    return MultiTaskImageFolder(
        data_path,
        args.all_domains,
        args=args,
        transform=transform,
        prefixes=prefixes,
        max_images=max_images,
    )

