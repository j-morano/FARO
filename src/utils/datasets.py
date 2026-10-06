import random

import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms

from utils import create_transform
from utils.data_utils import pil_loader

from .data_constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    MEDICAL_TASKS,
    IMAGE_TASKS
)
from .dataset_folder import (
    MultiTaskDatasetFolder,
    MultiTaskConstrastiveDatasetFolder,
    MultiTaskPatientBalancedDatasetFolder
)
# ImageFolder,

from skimage.transform import resize


def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(),
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )


class DataAugmentationForMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        trans = [transforms.RandomResizedCrop(args.input_size)]
        if args.hflip > 0.0:
            trans.append(transforms.RandomHorizontalFlip(args.hflip))
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))])

        self.transform = transforms.Compose(trans)

    def __call__(self, image):
        return self.transform(image)

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


class DataAugmentationForMultiMAEOld(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        self.rgb_mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        self.rgb_std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.input_size = args.input_size
        self.hflip = args.hflip
        self.intensity_shift = args.intensity_shift
        self.random_crop = args.random_crop
        self.affine = transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        )
        self.input_size = args.input_size
        self.args = args

    def __call__(self, task_dict):
        flip = random.random() < self.hflip # Stores whether to flip all images or not
        ijhw = None # Stores crop coordinates used for all tasks
        affine = self.affine
        affine_params = affine.get_params(
            affine.degrees, affine.translate, affine.scale, affine.shear,
            [512, 512]
        )

        cropped_size = None
        if self.random_crop < 1:
            cropped_size = int(self.input_size * self.random_crop)
        # Crop and flip all tasks randomly, but consistently for all tasks
        # Random intensity augmentation for images
        start_i = None
        start_j = None
        for task in task_dict:
            if task in MEDICAL_TASKS:
                if flip:
                    task_dict[task] = np.flip(task_dict[task], axis=-1)
                if self.intensity_shift > 0:
                    if task not in ['layermaps', 'bscanlayermap']:
                        # IMPORTANT: It assumes that the image is in [0, 1] range
                        shift = np.random.normal(0, self.intensity_shift, 1).astype(np.float32)
                        if random.random() < 0.5:
                            shift = -shift
                        task_dict[task] = np.clip(task_dict[task] + shift, 0, 1)
                        # print(f"Shift = {shift}")
                # Random crop in numpy
                if self.random_crop < 1:
                    intial_shape = task_dict[task].shape
                    if start_i is None and start_j is None:
                        start_i = random.randint(0, self.input_size - cropped_size)
                        start_j = random.randint(0, self.input_size - cropped_size)
                    task_dict[task] = task_dict[task][
                        start_i:start_i+cropped_size,  # type: ignore
                        start_j:start_j+cropped_size  # type: ignore
                    ]
                    if task in ['layermaps', 'bscanlayermap']:
                        anti_aliasing = False
                        order = 0
                    else:
                        anti_aliasing = True
                        order = 1
                    task_dict[task] = resize(
                        task_dict[task],
                        intial_shape,
                        anti_aliasing=anti_aliasing,
                        preserve_range=True,
                        order=order
                    )

                img = torch.from_numpy(task_dict[task].copy()).contiguous()
                # print(">>>", task, img.shape, img.min(), img.max())
                img = img.unsqueeze(0)

                if task in ['bscan', 'bscanlayermap']:
                    # All transformations
                    c_params = affine_params
                else:
                    # Only translation in x direction
                    c_params = 0, (affine_params[1][0],0), affine_params[2], 0
                fill = 0
                # if task in ['bscan', 'slo']: # and 'bscanlayermap' in task_dict:
                #     fill = 1
                if not self.args.no_affine:
                    img = TF.affine(  # type: ignore
                        img,
                        *c_params,
                        interpolation=affine.interpolation,
                        fill=fill,  # type: ignore
                        center=affine.center,
                    )
                interpolation = TF.InterpolationMode.BILINEAR
                if task in ['layermaps', 'bscanlayermap']:
                    interpolation = TF.InterpolationMode.NEAREST

                if img.shape[1:] != tuple(self.input_size[task]):
                    # print(f"Resizing {task} from {img.shape} to {self.input_size[task]}")
                    # if 'layermap' in task:
                    #     order = 0
                    # else:
                    # if sample.shape[0] < config.args.input_size[task][0]:
                    #     order = 1
                    # else:
                    #     order = 0
                    # print(f"Using order {order} for {task}")
                    # print(f"Resizing {task} from {sample.shape} to {target_size}")
                    img = TF.resize(
                        img,
                        self.input_size[task],
                        interpolation=interpolation,
                    )
                if task in ['layermaps', 'bscanlayermap']:
                    img = img.squeeze(0)
                # img = TF.to_tensor(task_dict[task])[:1]
                # img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
                # print(f"Task = {task}, shape = {task_dict[task].shape}")
                task_dict[task] = img
                continue
            # if ijhw is None:
            #     # Official MAE code uses (0.2, 1.0) for scale and (0.75, 1.3333) for ratio
            #     ijhw = transforms.RandomResizedCrop.get_params(
            #         task_dict[task], scale=(0.2, 1.0), ratio=(0.75, 1.3333)
            #     )
            # i, j, h, w = ijhw
            # task_dict[task] = TF.crop(task_dict[task], i, j, h, w)
            # task_dict[task] = task_dict[task].resize((self.input_size, self.input_size))
            # if flip:
            #     task_dict[task] = TF.hflip(task_dict[task])

        # Convert to Tensor
        # for task in task_dict:
        #     # TODO: change this if adding new modalities
        #     if task in MEDICAL_TASKS:
        #         img = torch.from_numpy(task_dict[task].copy()).contiguous()
        #         # print(">>>", task, img.shape, img.min(), img.max())
        #         img = img.unsqueeze(0)

        #         if task in ['bscan', 'bscanlayermap']:
        #             # All transformations
        #             c_params = affine_params
        #         else:
        #             # Only translation in x direction
        #             c_params = 0, (affine_params[1][0],0), affine_params[2], 0
        #         fill = 0
        #         # if task in ['bscan', 'slo']: # and 'bscanlayermap' in task_dict:
        #         #     fill = 1
        #         if not self.args.no_affine:
        #             img = TF.affine(  # type: ignore
        #                 img,
        #                 *c_params,
        #                 interpolation=affine.interpolation,
        #                 fill=fill,  # type: ignore
        #                 center=affine.center,
        #             )
        #         interpolation = TF.InterpolationMode.BILINEAR
        #         if task in ['layermaps', 'bscanlayermap']:
        #             interpolation = TF.InterpolationMode.NEAREST

        #         if img.shape[1:] != tuple(self.input_size[task]):
        #             # print(f"Resizing {task} from {img.shape} to {self.input_size[task]}")
        #             # if 'layermap' in task:
        #             #     order = 0
        #             # else:
        #             # if sample.shape[0] < config.args.input_size[task][0]:
        #             #     order = 1
        #             # else:
        #             #     order = 0
        #             # print(f"Using order {order} for {task}")
        #             # print(f"Resizing {task} from {sample.shape} to {target_size}")
        #             img = TF.resize(
        #                 img,
        #                 self.input_size[task],
        #                 interpolation=interpolation,
        #             )
        #         if task in ['layermaps', 'bscanlayermap']:
        #             img = img.squeeze(0)
        #         # img = TF.to_tensor(task_dict[task])[:1]
        #         # img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
        #     if task in ['depth']:
        #         img = torch.Tensor(np.array(task_dict[task]) / 2 ** 16)
        #         img = img.unsqueeze(0)  # 1 x H x W
        #     elif task in ['rgb']:
        #         img = TF.to_tensor(task_dict[task])
        #         img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
        #     elif task in ['semseg', 'semseg_coco']:
        #         # TODO: add this to a config instead
        #         # Rescale to 0.25x size (stride 4)
        #         scale_factor = 0.25
        #         img = task_dict[task].resize((int(self.input_size * scale_factor), int(self.input_size * scale_factor)))
        #         # Using pil_to_tensor keeps it in uint8, to_tensor converts it to float (rescaled to [0, 1])
        #         img = TF.pil_to_tensor(img).to(torch.long).squeeze(0)

        #     task_dict[task] = img

        return task_dict

    def __repr__(self):
        repr = "(DataAugmentationForMultiMAE,\n"
        #repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


class DataAugmentationForMultiMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        self.rgb_mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        self.rgb_std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.input_size = args.input_size
        self.hflip = args.hflip
        self.intensity_shift = args.intensity_shift
        if len(args.random_crop) > 0:
            parts = args.random_crop.split("--")
            self.random_crop = [float(p) for p in parts]
        else:
            self.random_crop = []
        self.affine = transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.8, 1.1),
            shear=0,
            # interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        )
        self.input_size = args.input_size
        self.args = args

    def random_add_speckle_noise(self, image: Tensor, p=0.8) -> Tensor:
        """Add speckle noise to an image."""
        if random.random() > p:
            return image
        else:
            noise = torch.randn_like(image) * 0.1  # Adjust the noise level as needed
            noisy_image = image + image * noise
            return torch.clamp(noisy_image, 0.0, 1.0)

    def random_gamma_correction(
        self,
        image: Tensor,
        gamma_range=(0.8, 1.2),
        eps=1e-6,
        p=0.8,
    ) -> Tensor:
        """Apply random gamma correction to an image."""
        if random.random() > p:
            return image
        else:
            gamma = random.uniform(*gamma_range)
            corrected_image = (image + eps) ** gamma
            return torch.clamp(corrected_image, 0.0, 1.0)

    def random_gaussian_blur(
        self,
        image: Tensor,
        kernel_size_range=(3, 7),
        sigma_range=(0.1, 2.0),
        p=0.5,
    ) -> Tensor:
        """Apply random Gaussian blur to an image."""
        if random.random() > p:
            return image
        else:
            kernel_size = random.choice(
                range(kernel_size_range[0], kernel_size_range[1] + 1, 2),
            )  # Ensure kernel size is odd
            sigma = [random.uniform(*sigma_range)]
            blurred_image = TF.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
            return blurred_image

    def crop(self, image: Tensor, crop_params: tuple) -> Tensor:
        """Crop the image to the specified size starting from (start_h, start_w)."""
        start_h, start_w, size = crop_params
        return image[..., start_h:start_h + size, start_w:start_w + size]

    def get_crop_params(self, image: Tensor, rel_size_low=0.8, rel_size_high=1.0):
        """Get random crop parameters for the image."""
        h, w = image.shape
        crop_size = random.randint(int(h * rel_size_low), int(h * rel_size_high))
        start_h = random.randint(0, h - crop_size)
        start_w = random.randint(0, w - crop_size)
        return start_h, start_w, crop_size

    def __call__(self, task_dict):
        flip = random.random() < self.hflip  # Stores whether to flip all images or not
        affine = self.affine
        affine_params = affine.get_params(
            affine.degrees, affine.translate, affine.scale, affine.shear,
            [512, 512]
        )
        crop_params = None
        if len(self.random_crop) > 0:
            crop_params = self.get_crop_params(
                next(iter(task_dict.values())),
                rel_size_low=self.random_crop[0],
                rel_size_high=self.random_crop[1]
            )
        if flip and 'laterality' in task_dict:
            task_dict['laterality'] = 1 - task_dict['laterality']
        for task in set(task_dict.keys()) & set(IMAGE_TASKS):
            if flip:
                task_dict[task] = np.flip(task_dict[task], axis=-1)
            if self.intensity_shift > 0 and task not in ['layermaps', 'bscanlayermap']:
                # IMPORTANT: It assumes that the image is in [0, 1] range
                shift = np.random.normal(0, self.intensity_shift, 1).astype(np.float32)
                if random.random() < 0.5:
                    shift = -shift
                task_dict[task] = np.clip(task_dict[task] + shift, 0, 1)

            img = torch.from_numpy(task_dict[task].copy()).contiguous().unsqueeze(0)

            if task in ['bscan', 'bscanlayermap']:
                # All transformations
                c_params = affine_params
            else:
                # Only translation in x direction
                c_params = 0, (affine_params[1][0],0), affine_params[2], 0
            if task in ['layermaps', 'bscanlayermap']:
                interpolation = TF.InterpolationMode.NEAREST
            else:
                interpolation = TF.InterpolationMode.BILINEAR
            if not self.args.no_affine:
                img = TF.affine(  # type: ignore
                    img,
                    *c_params,
                    interpolation=interpolation,
                    fill=0,  # type: ignore
                    center=affine.center,
                )
            if crop_params is not None:
                img = self.crop(img, crop_params)

            if img.shape[1:] != tuple(self.input_size[task]):
                img = TF.resize(
                    img,
                    self.input_size[task],
                    interpolation=interpolation,
                )

            if self.args.contrastive and task == 'bscan':
                img = self.random_add_speckle_noise(img)
                img = self.random_gamma_correction(img)
                img = self.random_gaussian_blur(img)

            if task in ['layermaps', 'bscanlayermap']:
                img = img.squeeze(0)
            task_dict[task] = img

        if 'bscanlayermap' in task_dict.keys():
            mask = task_dict['bscanlayermap']
            # Define your range of lesion classes
            lesion_range = torch.arange(12, 19, device=mask.device)
            # Use broadcasting to check all classes at once
            # mask[..., None] adds a new dimension to compare against the range
            # .any() checks if that class exists anywhere in the mask
            presence = (mask[..., None] == lesion_range).any(dim=0).any(dim=0)
            task_dict['lesion_presence'] = presence.float()

        return task_dict

    def __repr__(self):
        repr = "(DataAugmentationForMultiMAE,\n"
        #repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


# def build_pretraining_dataset(args):
#     transform = DataAugmentationForMAE(args)
#     print("Data Aug = %s" % str(transform))
#     return ImageFolder(args.data_path, transform=transform)

def build_patient_balanced_multimae_pretraining_dataset(args):
    transform = DataAugmentationForMultiMAE(args)
    print(f'Data from {args.data_path}')
    return MultiTaskPatientBalancedDatasetFolder(
        args.data_path,
        args.all_domains,
        args=args,
        transform=transform,
        loader=pil_loader,
    )


def build_multimae_pretraining_dataset(args):
    transform = DataAugmentationForMultiMAE(args)
    print(f'Data from {args.data_path}')
    return MultiTaskDatasetFolder(
        args.data_path,
        args.all_domains,
        args=args,
        transform=transform,
        loader=pil_loader,
    )


def build_multimae_contrastive_pretraining_dataset(args):
    transform = DataAugmentationForMultiMAE(args)
    print(f'Data from {args.data_path}')
    return MultiTaskConstrastiveDatasetFolder(
        args.data_path,
        args.all_domains,
        args=args,
        transform=transform,
        loader=pil_loader,
    )


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        # root = os.path.join(args.data_path, 'train' if is_train else 'val')
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    return transforms.Compose([
        transforms.ToTensor(),
    ])
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            # transform.transforms[0] = transforms.RandomCrop(
            #     args.input_size, padding=4)
            ...
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        # t.append(
        #     transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        # )
        # t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
