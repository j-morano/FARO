from typing import Dict
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from mutils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mutils.dataset_folder import MultiTaskImageFolder


def simple_transform(
    train: bool,
    input_size: int = 512,
    norm: str = 'minmax',
):
    """Default transform for semantic segmentation using torchvision v2."""

    norm_list = []
    if norm == 'imagenet':
        print("Using imagenet normalization")
        norm_list += [
            v2.Grayscale(num_output_channels=3),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    elif norm == 'sam':
        print("Using SAM normalization")
        norm_list += [
            v2.Grayscale(num_output_channels=3),
            # Scale to [0, 255] — handled by ToDtype below
        ]
    elif norm == 'z-score':
        print("Using z-score normalization")
        norm_list += [
            v2.Grayscale(num_output_channels=3),
            v2.Normalize(mean=0, std=1),
        ]

    if train:
        init_size = input_size + int(input_size * 0.1)
        transform_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Resize(
                size=(init_size, init_size),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomCrop(size=(input_size, input_size)),
        ] + norm_list
    else:
        transform_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(
                size=(input_size, input_size),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ] + norm_list

    transform = v2.Compose(transform_list)
    print(
        f'[Train: {train}]'
        f' [Input size: {input_size}]'
        f' [Normalization: {norm}]'
        f' [Transform list: {transform_list}]'
    )
    return transform


class DataAugmentationForSemSeg:
    """Data transform for semantic segmentation using torchvision v2."""

    def __init__(
        self,
        transform,
        key_to_replace='bscan',
    ):
        self.transform = transform
        self.key_to_replace = key_to_replace

    def __call__(self, task_dict):
        key = self.key_to_replace

        # Convert image to tensor wrapped as tv_tensors.Image
        image = np.array(task_dict[key])
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        # HWC -> CHW for tv_tensors
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = tv_tensors.Image(image)

        # Wrap segmentation mask as tv_tensors.Mask so spatial transforms
        # are applied consistently
        if 'semseg' in task_dict:
            mask = np.array(task_dict['semseg'])
            mask = tv_tensors.Mask(torch.from_numpy(mask).long())
            image, mask = self.transform(image, mask)
            task_dict['semseg'] = mask.squeeze(0).long()
        else:
            image = self.transform(image)

        task_dict[key] = image.float()
        return task_dict


def build_semseg_dataset(args, data_path, transform, max_images=None):
    transform = DataAugmentationForSemSeg(
        transform=transform,
        key_to_replace=args.in_domains[0],
    )
    return MultiTaskImageFolder(
        data_path,
        args.all_domains,
        args=args,
        transform=transform,
        prefixes=None,
        max_images=max_images,
    )
