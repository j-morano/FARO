import argparse
from typing import List, Callable, Optional
import unittest

import torch
from torch import nn
import torchvision.transforms as tvtr
from torchvision.transforms import v2
from torch.nn import functional as F
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from einops import rearrange

from mirage_wrapper import miragecls_factory
# from mutils.transforms import (
#     RandomIntensityChannel,
#     RandomAffineChannel,
#     Identity,
#     MinMaxNormChannel,
#     MinMaxNorm,
#     ToRGB,
#     # NaiveNormChannel
# )
from mutils.factory import get_factory_adder
from mutils.vit import vit_large_patch16, vit_base_patch16
from models.octcube.inference_utils import create_models
from models.ibot.models.vision_transformer import vit_base as ibot_vit_base
from models.urfound import model_urfound
from run_posttraining import Vision3DModel, ModelArgs



add_config, fm_config_factory = get_factory_adder()


########################################################################
# Constants
########################################################################

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


########################################################################
# Common
########################################################################


# class MinMaxNorm:
#     """Transforms each channel to the range [0, 1] safely."""
#     def __init__(self, eps=1e-8):
#         self.eps = eps

#     def __call__(self, tensor):
#         t_min = tensor.min()
#         t_max = tensor.max()
#         # Adding eps prevents division by zero
#         return (tensor - t_min) / (t_max - t_min + self.eps)


class ToFloat:
    """Converts tensor to float and scales to [0, 1] if it's an integer type."""
    def __call__(self, tensor):
        if tensor.dtype == torch.uint8:
            return tensor.float() / 255.0
        elif tensor.dtype == torch.uint16:
            return tensor.float() / 65535.0
        else:
            return tensor.float()


class FoundModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.weight_decay = 1e-2
        if args.fill is not None:
            if args.fill < 0:
                args.fill = None
        args.lr = 1e-3
        self.args = args
        self.embed_dim = 768
        self.model: nn.Module

    def get_optimizer(self, model):
        return torch.optim.AdamW(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

    def build_transform(self, subset, augment):
        # Order: Spatial -> Color -> Tensor -> Normalize
        print(f'>>> Building transform "{subset}"')
        msg = []
        if self.args.fill is None:
            fill = 0
        else:
            fill = self.args.fill
        msg.append(f'Using fill={fill}')
        msg.append('Convert to grayscale')
        transforms_list = self.get_color_norm()
        msg.append(f'Resize to {self.args.input_size}x{self.args.input_size}')
        transforms_list.append(
            v2.Resize(
                size=(self.args.input_size, self.args.input_size),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
        if augment:
            msg.append('Random horizontal flip (0.5)')
            transforms_list.append(
                v2.RandomHorizontalFlip(p=0.5),
            )
            if self.args.strong_aug:
                msg.append(f'Random affine (fill={fill})')
                transforms_list.append(
                    v2.RandomAffine(
                        degrees=(-10, 10),
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                        shear=5,
                        interpolation=tvtr.InterpolationMode.BICUBIC,
                        fill=fill,
                    )
                )
                msg.append('Random intensity shift (per channel)')
                transforms_list.append(
                    v2.ColorJitter(
                        brightness=0.2,
                        contrast=0.1,
                        saturation=0.0,
                        hue=0.0,
                    )
                )
            else:
                msg.append('No random affine')

        msg += [
            'To float tensor with [0, 1] range',
            'Normalize'
        ]
        transforms_list += [
            # To tensor
            v2.ToImage(),
            # Scaling here divides by the max of the input data type
            v2.ToDtype(torch.float32, scale=True),
            # Normalize
        ]
        transforms_list += self.get_model_norm()
        print('\n'.join(msg))

        return v2.Compose(transforms_list)
        # norm_list = self.get_model_norm()
        # min_max = self.get_min_max()
        # norm_list +=

        # transforms_list = [
        #     grayscale,
        #     tvtr.ToTensor(),
        #     tvtr.ConvertImageDtype(torch.float32),
        #     min_max,
        # ]
        # if augment:
        #     print('Random horizontal flip (0.5)')
        #     print(intensity_msg)
        #     print(affine_msg)
        #     transforms_list += [
        #         intensity,
        #         affine,
        #     ]
        # print('Norm list:', norm_list)
        # transforms_list += norm_list
        # transforms = tvtr.Compose(transforms_list)

        # return transforms

    def get_color_norm(self) -> List[Callable]:
        return [
            v2.Grayscale(num_output_channels=1),
        ]

    def get_model_norm(self) -> List[Callable]:
        # By default, min-max normalization
        return [ ToFloat() ]
        # return [
        #     ToRGB(),
        #     tvtr.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        # ]

    # def get_min_max(self) -> Callable:
    #     return Identity()

    def set_requires_grad(self):
        # print(model.state_dict().keys())
        print('Freezing encoder layers for linear probing')
        # freeze encoder layers for linear probing
        print('Tuned parameters:')
        for name, param in self.model.named_parameters():
            if 'head.' not in name:  # and 'norm.' not in name:
                param.requires_grad = False
            else:
                print('\t', name)
                param.requires_grad = True


class FoundModel3D:
    def __init__(self, args):
        args.weight_decay = 1e-2
        if args.fill is not None:
            if args.fill < 0:
                args.fill = None
        args.lr = 1e-3
        self.args = args
        self.model: nn.Module
        self.slices = None
        # self.head: nn.Module

    def get_optimizer(self, model):
        return torch.optim.AdamW(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

    def resize_3d(self, x):
        # x shape: (C, H, W) -> add batch and channel dim for F.interpolate
        if self.slices is None:
            slices = x.shape[1]
        else:
            slices = self.slices
        x = x.unsqueeze(0)
        x = F.interpolate(
            x,
            size=(slices, self.args.input_size, self.args.input_size),
            mode='trilinear',
            align_corners=False
        )
        return x.squeeze(0)

    def rand_flip(self, x):
        if torch.rand(1) > 0.5:
            return torch.flip(x, dims=[-1])
        # if torch.rand(1) > 0.5:
        #     return torch.flip(x, dims=[-3])
        return x

    def intensity_shift(self, x):
        # Simulating ColorJitter/Brightness for 3D
        shift = (torch.rand(1) * 0.4) - 0.2 # range [-0.2, 0.2]
        return torch.clamp(x + shift, 0, 1)

    def normalize_3d(self, x):
        return x

    def to_tensor(self, x):
        x = torch.from_numpy(x).float()
        if x.max() > 255:
            return x / 65535.0
        elif x.max() > 1.0:
            return x / 255.0
        else:
            return x

    def build_transform(self, subset, augment):
        print(f'>>> Building Native 3D transform "{subset}"')
        msg = []

        # We define a sequence of functions instead of v2.Compose
        # because many v2 classes don't support 5D tensors.
        transforms = []

        # 0. Conversion to tensor and scaling to [0, 1]
        msg.append('Convert to float tensor with [0, 1] range')
        transforms.append(self.to_tensor)

        # 1. Resize (3D Trilinear)
        msg.append(f'3D Trilinear Resize to {self.args.input_size}^3')
        transforms.append(self.resize_3d)

        if augment:
            # 2. Horizontal Flip (Dim -1 is Width)
            msg.append('Random horizontal flip (0.5)')
            transforms.append(self.rand_flip)

            if self.args.strong_aug:
                msg.append('Random 3D Intensity Shift')
                transforms.append(self.intensity_shift)

        # 3. Normalization
        msg.append('Normalize (Native)')
        transforms.append(self.normalize_3d)

        print('\n'.join(msg))

        # Return a simple lambda that applies the list of functions
        return lambda img: self.apply_list(img, transforms)

    def apply_list(self, img, funcs):
        for f in funcs:
            img = f(img)
        return img

    def set_requires_grad(self):
        # print(model.state_dict().keys())
        print('Freezing encoder layers for linear probing')
        # freeze encoder layers for linear probing
        print('Tuned parameters:')
        for name, param in self.model.named_parameters():
            # NOTE: CHANGED WITH RESPECT TO THE 2D VERSION
            if not name.startswith('head.'):  # and 'norm.' not in name:
                param.requires_grad = False
            else:
                print('\t', name)
                param.requires_grad = True



# class DummyModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool3d(1),
#             nn.Flatten(),
#         )
#         self.head = nn.Linear(16, num_classes)
#
#     def forward(self, x):
#         return self.head(self.encoder(x))
#
#
# @add_config('miragefm3d-dummy')
# class MIRAGEFM3DDummy(FoundModel3D):
#     def __init__(self, args):
#         super().__init__(args)
#         if args.input_size is None:
#             args.input_size = 512
#         if args.pool is None:
#             args.pool = 'ultra_mix'
#         self.model = DummyModel(num_classes=args.num_classes)
#         self.args = args




class MIRAGEFM3DModel(nn.Module):
    def __init__(self, args, model_args, bridge_weights):
        super().__init__()
        self.args = args
        self.model = Vision3DModel(model_args, args.device)
        if model_args.use_proj:
            dim = 1024
        else:
            dim = 768
        self.model.load_bridge(bridge_weights)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        if args.pool.lower() == 'none':
            self.head = nn.Linear(18*dim, args.num_classes)
        elif args.pool in ['avg']:
            self.head = nn.Linear(dim, args.num_classes)
        elif args.pool in ['2avg']:
            self.head = nn.Linear(2*dim, args.num_classes)

    def forward(self, x, return_features=False):
        with torch.no_grad():
            features = self.model(x)
        features = self.norm(features)
        if self.args.pool == 'none' and not return_features:
            features = rearrange(features, 'B F D -> B (F D)')
        elif self.args.pool == 'avg':
            features = features.mean(dim=1)
        elif self.args.pool == '2avg':
            num_q = self.model.args.num_queries
            r_features = features[:, :num_q, :].mean(dim=1)
            l_features = features[:, num_q:, :].mean(dim=1)
            features = torch.cat([r_features, l_features], dim=1)
        if return_features:
            return features
        out = self.head(features)
        return out


@add_config('miragefm3d')
class MIRAGEFM3D(FoundModel3D):
    def __init__(self, args):
        super().__init__(args)
        if args.input_size is None:
            args.input_size = 512
        if args.pool is None:
            args.pool = 'none'
        model_args = ModelArgs(
            num_queries=16,
            resampler='axbidmlstm',
            lesion_pool='bidmlstm',
        )
        self.model = MIRAGEFM3DModel(
            args,
            model_args,
            '_weights/r-axbidmlstm_lp-bidmlstm_nq-16_as-32_sk-True_fw-True__010.pt'
        )
        self.args = args


@add_config('miragefm3d_naive')
class MIRAGEFM3DNaive(FoundModel3D):
    def __init__(self, args):
        super().__init__(args)
        if args.input_size is None:
            args.input_size = 512
        if args.pool is None:
            args.pool = 'none'
        model_args = ModelArgs(
            num_queries=16,
            resampler='meanpool',
            lesion_pool='max_avg',
        )
        self.model = MIRAGEFM3DModel(
            args,
            model_args,
            '_weights/r-meanpool_lp-max_avg_nq-16_as-32_sk-True_fw-True__010.pt'
        )
        self.args = args


@add_config('miragefm3d_naive_proj')
class MIRAGEFM3DNaiveProj(FoundModel3D):
    def __init__(self, args):
        super().__init__(args)
        if args.input_size is None:
            args.input_size = 512
        if args.pool is None:
            args.pool = 'none'
        model_args = ModelArgs(
            num_queries=16,
            resampler='meanpool',
            lesion_pool='max_avg',
            use_proj=True,
        )
        self.model = MIRAGEFM3DModel(
            args,
            model_args,
            '_weights/r-meanpool_lp-max_avg_nq-16_as-32_sk-True_fw-True__010.pt'
        )
        self.args = args


@add_config('miragefm3d_xlstm_xattn')
class MIRAGEFM3DxLSTMXAttn(FoundModel3D):
    def __init__(self, args):
        super().__init__(args)
        if args.input_size is None:
            args.input_size = 512
        if args.pool is None:
            args.pool = 'none'
        model_args = ModelArgs(
            num_queries=16,
            resampler='axbidmlstm',
            lesion_pool='xattn_v2',
        )
        self.model = MIRAGEFM3DModel(
            args,
            model_args,
            '_weights/r-axbidmlstm_lp-xattn_v2_nq-16_as-32_sk-True_fw-True__010.pt'
        )
        self.args = args



@add_config('miragefm3d_xattn_xlstm')
class MIRAGEFM3DXAttnxLSTM(FoundModel3D):
    def __init__(self, args):
        super().__init__(args)
        if args.input_size is None:
            args.input_size = 512
        if args.pool is None:
            args.pool = 'none'
        model_args = ModelArgs(
            num_queries=16,
            resampler='xattn_v2',
            lesion_pool='bidmlstm',
        )
        self.model = MIRAGEFM3DModel(
            args,
            model_args,
            '_weights/r-xattn_v2_lp-bidmlstm_nq-16_as-32_sk-True_fw-True__010.pt'
        )
        self.args = args



@add_config('miragefm3d_xattn_xattn')
class MIRAGEFM3DXAttnXAttn(FoundModel3D):
    def __init__(self, args):
        super().__init__(args)
        if args.input_size is None:
            args.input_size = 512
        if args.pool is None:
            args.pool = 'none'
        model_args = ModelArgs(
            num_queries=16,
            resampler='xattn_v2',
            lesion_pool='xattn_v2',
        )
        self.model = MIRAGEFM3DModel(
            args,
            model_args,
            '_weights/r-xattn_v2_lp-xattn_v2_nq-16_as-32_sk-True_fw-True__010.pt'
        )
        self.args = args



@add_config('miragefm3d_xattn_xattn_60')
class MIRAGEFM3DXAttnXAttn60(MIRAGEFM3DXAttnXAttn):
    def __init__(self, args):
        super().__init__(args)
        self.slices = 60



########################################################################
# MIRAGE
########################################################################


class MIRAGEFM(FoundModel):
    weights_fn: Optional[str] = None
    pool: Optional[str] = None
    num_global_tokens: int = 10
    model_size: str = "base"
    chd: bool = False

    def __init__(self, args):
        super().__init__(args)
        self.patch_size = 32
        if args.input_size is None:
            args.input_size = 512
        if self.model_size == 'base':
            self.embed_dim = 768
        elif self.model_size == 'large':
            self.embed_dim = 1024
        if args.pool is not None:
            self.pool = args.pool
        if self.pool is not None:
            print("Using pool from class variable:", self.pool)
            args.pool = self.pool
        else:
            if args.pool is None:
                args.pool = 'ultra_mix'

        self.model = miragecls_factory[args.pool](
            input_size=args.input_size,
            patch_size=self.patch_size,
            num_classes=args.num_classes,
            modalities='bscan',
            # NOTE: weights are loaded in the model
            weights=self.weights_fn,
            device=args.device,
            model_size=self.model_size,
            num_global_tokens=self.num_global_tokens,
            chd=self.chd
        )
        self.args = args

    def get_model_norm(self) -> List[Callable]:
        return [ ToFloat() ]

    # def get_min_max(self) -> Callable:
    #     return MinMaxNormChannel()

class MIRAGEFMChD(MIRAGEFM):
    chd = True
    def get_color_norm(self) -> List[Callable]:
        return [ ]

#-----------------------------------------------------------------------
# ABLATION


@add_config('nonPB-MAE-only')
class MIRAGEnonPBMAEOnlyFM(MIRAGEFM):
    weights_fn = './_weights/ablation/v5_multimae-b-nonPB-MAE-only_64_bscan-to-bscan_512--32_shuffle_014.pth'
    pool = 'patch'

@add_config('nonPB-MIRAGE')
class MIRAGEnonPBMIRAGEFM(MIRAGEFM):
    weights_fn = './_weights/ablation/v5_multimae-b-nonPB-MIRAGE_64_bscan-bscanlayermap-to-bscan-bscanlayermap_512-128--32-8_shuffle_014.pth'
    pool = 'patch'

@add_config('nonPB-MIRAGE-tgt-layers')
class MIRAGEnonPBMIRAGETgtLayersFM(MIRAGEFM):
    weights_fn = './_weights/ablation/v5_multimae-b-nonPB-MIRAGE-tgt-layers_64_bscan-to-bscan-bscanlayermap_512-128--32-8_shuffle_014.pth'
    pool = 'patch'

@add_config('PB-MIRAGE-tgt-layers')
class MIRAGEPBMIRAGETgtLayersFM(MIRAGEFM):
    weights_fn = './_weights/ablation/v5_multimae-b-PB-MIRAGE-tgt-layers_64_bscan-to-bscan-bscanlayermap_512-128--32-8_shuffle_014.pth'
    pool = 'patch'

@add_config('PB-Mv2-Le')
class MIRAGEPBMIRAGEv2LeFM(MIRAGEFM):
    weights_fn = './_weights/ablation/v5_multimae-b-pbnofftLeOnly_64_bscan-to-bscan-bscanlayermap_512-128--32-8_shuffle_014.pth'
    pool = 'conw_PaLe'


@add_config('Mv2-Le-v6')
class MIRAGEv2Lev6FM(MIRAGEFM):
    weights_fn = './_weights/ablation/v6_multimae-b-nonpbnofftLeOnly_64_bscan-bscanlayermap_512-128--32-8_shuffle_014.pth'
    pool = 'conw_PaLe'


@add_config('Mv2-Le-v6-Dice')
class MIRAGEv2Lev6DiceFM(MIRAGEFM):
    weights_fn = './_weights/ablation/v6_multimae-b-dice-pbnofftLeOnly_64_bscan-to-bscan-bscanlayermap_512-128--32-8_shuffle_014.pth'
    pool = 'conw_PaLe'


@add_config('PB-Mv2-Le_ChD')
class MIRAGEPBMIRAGEv2LeFMChD(MIRAGEFMChD):
    weights_fn = './_weights/ablation/v5_multimae-b-pbnofftLeOnly_64_bscan-to-bscan-bscanlayermap_512-128--32-8_shuffle_014.pth'
    pool = 'conw_PaLe'


@add_config('Mv2-Large-014')
class MIRAGEMv2LargeFM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-L_014.pth'
    pool = 'conw_PaLe'
    model_size = 'large'


@add_config('Mv2-Large-099')
class MIRAGEMv2Large099FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT_L_099.pth'
    pool = 'conw_PaLe'
    model_size = 'large'


#-----------------------------------------------------------------------


@add_config('miragev1-base')
class MIRAGEv1BaseFM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEv1-Base.pth'
    model_size = 'base'
    num_global_tokens = 1


@add_config('miragev1-bscan-only')
class MIRAGEv1BscanOnlyFM(MIRAGEFM):
    weights_fn = './_weights/v3_multimae-b_pret-multimae_49_1600e_bscan_512-32_checkpoint-1599.pth'
    model_size = 'base'
    num_global_tokens = 1


@add_config('miragev1-large')
class MIRAGEv1LargeFM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEv1-Large.pth'
    model_size = 'large'
    num_global_tokens = 1


@add_config('miragev1-large-chd')
class MIRAGEv1LargeFMChD(MIRAGEFMChD):
    weights_fn = './_weights/MIRAGEv1-Large.pth'
    model_size = 'large'
    num_global_tokens = 1


@add_config('mirage-large')
class MIRAGELargeFM(MIRAGEFM):
    weights_fn = './_weights/MIRAGE-Large.pth'


@add_config('mirageoct-large')
class MIRAGEOCTLargeFM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-Large.pth'


@add_config('mirageoct-ongoing-large')
class MIRAGEOCTLargeOngoingFM(MIRAGEFM):
    weights_fn = './_weights/checkpoint-last_v1.pth'


@add_config('mirageoct-v2-large')
class MIRAGEOCTLargev2FM(MIRAGEFM):
    weights_fn = './_weights/checkpoint-last_v2.pth'


@add_config('mirageoct-con-large')
class MIRAGEOCTLargeConstrastiveFM(MIRAGEFM):
    weights_fn = './_weights/checkpoint-last_con_v1.pth'


@add_config('mirageoct-con-large-v2')
class MIRAGEOCTLargeConstrastivev2FM(MIRAGEFM):
    weights_fn = './_weights/checkpoint-last_con_v2.pth'


@add_config('mirageoct-con-multislice')
class MIRAGEOCTLargeConstrastiveMultiSliceFM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-CON-checkpoint-059.pth'


@add_config('mirageoct-con-multislice-79')
class MIRAGEOCTLargeConstrastiveMultiSlice79FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-CON-checkpoint-079.pth'


@add_config('mirageoct-con-multislice-84')
class MIRAGEOCTLargeConstrastiveMultiSlice84FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-CON-checkpoint-084.pth'


@add_config('mirageoct-con-multislice-94')
class MIRAGEOCTLargeConstrastiveMultiSlice94FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-CON-checkpoint-094.pth'


@add_config('mirageoct-conw-multislice-104')
class MIRAGEOCTLargeConstrastiveWidthMultiSlice104FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-CONW-checkpoint-104.pth'


@add_config('mirageoct-conwms-014')
class MIRAGEOCTBaseCONWMS014FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-CONWMS-014.pth'


@add_config('mirageoct-conwmsnew-014')
class MIRAGEOCTBaseCONWMSNew014FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-CONWMS_NEW-014.pth'


@add_config('mirageoct-conwl-multislice-105')
class MIRAGEOCTLargeConstrastiveWidthLinMultiSlice105FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-CONWL-checkpoint-105.pth'

@add_config('mirageoct-mm-checkpoint-002')
class MIRAGEOCTLargeMM002FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-MM-checkpoint-002.pth'

@add_config('mirageoct-mm-checkpoint-003')
class MIRAGEOCTLargeMM003FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-MM-checkpoint-003.pth'


@add_config('mirageoct-mm-006')
class MIRAGEOCTLargeMM006FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-MM-006.pth'


@add_config('mirageoct-pbwt-014')
class MIRAGEOCTBasePatientBalancedWidthThickness014FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-PBWT-014.pth'


@add_config('mirageoct-pbwtmmi-014')
class MIRAGEOCTBasePatientBalancedWidthThicknessMultimodalInputs014FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-PBWTMMI-014.pth'


@add_config('mirageoct-pbwt-029')
class MIRAGEOCTBasePatientBalancedWidthThickness029FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-PBWT-029.pth'


@add_config('mirageoct-pbwnofft-014')
class MIRAGEOCTBasePatientBalancedWidthNoFFTFM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-PBWNoFFT-014.pth'


@add_config('mirageoct-pbw-014')
class MIRAGEOCTBasePatientBalancedWidthFM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-PBW-014.pth'


@add_config('mirageoct-pbwnofftleo-014')
class MIRAGEOCTBasePatientBalancedNoFFTLeOnlyFM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-PBnofftLeOnly-014.pth'


@add_config('mirageoct-pbnofftlepos-014')
class MIRAGEOCTBasePatientBalancedNoFFTLePos(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-PBnofftLePos-014.pth'


@add_config('mirageoct-pbnofflethhc-014')
class MIRAGEOCTBasePatientBalancedNoFFTLeThHC(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-PBnofftLeThHC-014.pth'


@add_config('mirageoct-pbwnofftleth-014')
class MIRAGEOCTBasePatientBalancedNoFFTLeTh(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-PBnofftLeTh-014.pth'


@add_config('mirageoct-pbwnofftleo-049')
class MIRAGEOCTBasePatientBalancedNoFFTLeOnlyFM49(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-PBnofftLeOnly-049.pth'


@add_config('mirageoct-conwmst-005')
class MIRAGEOCTBaseConWMSThickness005FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-CONWMST-005.pth'


@add_config('mirageoct-conwmst-014')
class MIRAGEOCTBaseConWMSThickness014FM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-CONWMST-014.pth'




@add_config('mirage-base')
class MIRAGEBaseFM(MIRAGEFM):
    weights_fn = './_weights/MIRAGE-Base.pth'


@add_config('mirageoct-base')
class MIRAGEOCTBaseFM(MIRAGEFM):
    weights_fn = './_weights/MIRAGEOCT-Base.pth'



########################################################################
# SOTA
########################################################################


#=======================================================================
# RETFound

@add_config('retfound')
class RETFoundFM(FoundModel):
    chd: bool = False

    def __init__(self, args):
        super().__init__(args)
        self.patch_size = 16
        self.embed_dim = 1024
        if args.input_size is None:
            args.input_size = 224
        if args.pool is None:
            args.pool = 'patch'
        self.model = vit_large_patch16(
            num_classes=args.num_classes,
            img_size=args.input_size,
            pool=args.pool,
            chd=self.chd
        )
        # Load weights from local args.weights path
        state_dict = torch.load(
            './_weights/RETFound-Large.pth',
            map_location=args.device,
            weights_only=False,
        )['model']
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if 'decoder_' not in key:
                new_state_dict[key] = state_dict.pop(key)
        interpolate_pos_embed_vit(self.model, new_state_dict)
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(msg)
        self.args = args

    def get_color_norm(self) -> List[Callable]:
        return [
            v2.Grayscale(num_output_channels=3),
        ]

    def get_model_norm(self) -> List[Callable]:
        # ImageNet
        normalize = v2.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        return [normalize]


class ChDColorNormRGB:
    def __init__(self):
        self.to_tensor = v2.ToImage()
        self.norm_fun = v2.Grayscale(num_output_channels=3)

    def __call__(self, x):
        x = self.to_tensor(x)          # PIL -> tensor (C, H, W)
        x_1 = x[0:1, :, :]            # no batch dim yet
        x_2 = x[1:2, :, :]
        x_1_norm = self.norm_fun(x_1)  # (3, H, W)
        x_2_norm = self.norm_fun(x_2)  # (3, H, W)
        return torch.cat([x_1_norm, x_2_norm], dim=0)  # (6, H, W)


@add_config('retfound-chd')
class RETFoundFMChD(RETFoundFM):
    """Adapted this to fit the format of the change detection dataset,
    where the first channel is the t0 image, and second channel is t1.
    """
    chd = True

    def get_color_norm(self) -> List[Callable]:
        return [ ChDColorNormRGB() ]

    def get_model_norm(self) -> List[Callable]:
        normalize = v2.Normalize(
            mean=IMAGENET_MEAN * 2,
            std=IMAGENET_STD * 2,
        )
        return [normalize]




#=======================================================================
# VisionFM

def interpolate_pos_embed_vit(model, checkpoint_model):
    if 'pos_embed' not in checkpoint_model:
        raise ValueError("Checkpoint model does not contain 'pos_embed' key for position embeddings.")

    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    print(model.pos_embed.shape, pos_embed_checkpoint.shape)

    # Grid size in the checkpoint
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # Grid size for the new resolution
    new_size = int(num_patches ** 0.5)

    if orig_size == new_size:
        return

    print(f"Position embedding interpolate: {orig_size}x{orig_size} -> {new_size}x{new_size}")

    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

    checkpoint_model['pos_embed'] = torch.cat((extra_tokens, pos_tokens), dim=1)


@add_config('visionfm-base')
class VisionFM(FoundModel):
    # https://github.com/ABILab-CUHK/VisionFM/blob/648ab0895f8e80e2c937f658bd2f1629ea6d27e7/inference_visionfm_for_multiclass_classification.py#L103
    # https://github.com/ABILab-CUHK/VisionFM/blob/648ab0895f8e80e2c937f658bd2f1629ea6d27e7/utils.py#L55
    # NOTE: They use their own mean and std from their dataset
    VISIONFM_MEAN = (0.21091926, 0.21091926, 0.21091919)
    VISIONFM_STD  = (0.17598894, 0.17598891, 0.17598893)
    chd: bool = False

    def __init__(self, args):
        super().__init__(args)
        self.patch_size = 16
        if args.input_size is None:
            args.input_size = 224
        if args.pool is None:
            args.pool = 'cls'
        self.model = vit_base_patch16(
            num_classes=args.num_classes,
            img_size=args.input_size,
            pool=args.pool,
            chd=self.chd
        )
        # Load weights from local args.weights path
        state_dict = torch.load(
            './_weights/VisionFM-Base.pth',
            map_location=args.device,
            weights_only=False,
        )['teacher']
        for key in list(state_dict.keys()):
            state_dict[key.replace('backbone.', '')] = state_dict.pop(key)
        interpolate_pos_embed_vit(self.model, state_dict)
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(msg)
        self.args = args

    def get_color_norm(self) -> List[Callable]:
        return [
            v2.Grayscale(num_output_channels=3),
        ]

    def get_model_norm(self) -> List[Callable]:
        normalize = v2.Normalize(
            mean = self.VISIONFM_MEAN,
            std = self.VISIONFM_STD
        )
        return [normalize]


@add_config('visionfm-base-chd')
class VisionFMChD(VisionFM):
    """Adapted this to fit the format of the change detection dataset,
    where the first channel is the t0 image, and second channel is t1.
    """
    chd = True

    def get_color_norm(self) -> List[Callable]:
        return [ ChDColorNormRGB() ]

    def get_model_norm(self) -> List[Callable]:
        normalize = v2.Normalize(
            mean=self.VISIONFM_MEAN * 2,
            std=self.VISIONFM_STD * 2,
        )
        return [normalize]


#=======================================================================
# iBOT baseline

class IBOTModel(nn.Module):
    def __init__(self, state_dict, args):
        super().__init__()
        self.model = ibot_vit_base(
            img_size=[args.input_size],
        )
        if args.pool == 'cls_patch':
            input_features = 768 * 2
        else:
            input_features = 768
        self.head = nn.Linear(
            in_features=input_features,
            out_features=args.num_classes,
        )
        interpolate_pos_embed_vit(self.model, state_dict)
        msg = self.model.load_state_dict(state_dict, strict=True)
        print(msg)
        self.pool = args.pool

    def forward(self, x, return_features=False):
        features = self.model(x, return_all_tokens=True)
        if self.pool == 'cls_patch':
            # Assuming features shape is [Batch, Num_Tokens, 768]
            cls_token = features[:, 0]  # [Batch, 768]
            patch_tokens = features[:, 1:, :]  # [Batch, Num_Tokens-1, 768]
            avg_patch = patch_tokens.mean(dim=1)  # [Batch, 768]
            features = torch.cat([cls_token, avg_patch], dim=1)  # [Batch, 1536]
        elif self.pool == 'patch':
            features = features[:, 1:, :].mean(dim=1)  # Global avg pool of patch tokens
        elif self.pool == 'seg':
            features = features[:, 1:, :]
        else:
            features = features[:, 0]  # CLS token
        if return_features:
            return features
        return self.head(features)


@add_config('ibot-base')
class IBOTBaseFM(FoundModel):
    def __init__(self, args):
        super().__init__(args)
        self.patch_size = 16
        if args.input_size is None:
            args.input_size = 224
        if args.pool is None:
            args.pool = 'cls'
        state_dict = torch.load(
            './_weights/ibot-checkpoint-014.pth',
            map_location=args.device,
            weights_only=False,
        )['student']
        # Remove 'module.backbone.' from state_dict keys
        # Also remove all 'module.head.' and 'masked_embed' keys
        for key in list(state_dict.keys()):
            if key.startswith('module.backbone.'):
                if 'masked_embed' in key:
                    state_dict.pop(key)
                else:
                    new_key = key.replace('module.backbone.', '')
                    state_dict[new_key] = state_dict.pop(key)
            elif key.startswith('module.head.'):
                state_dict.pop(key)
        self.model = IBOTModel(
            state_dict=state_dict,
            args=args,
        )
        self.args = args

    def get_color_norm(self) -> List[Callable]:
        return [
            v2.Grayscale(num_output_channels=3),
        ]



#=======================================================================
# OCTCube


class OCTCubeModel(nn.Module):
    def __init__(self, model, pool='patch', num_classes=16, device='cuda', chd=False):
        super().__init__()
        self.model = model
        if num_classes != 16:
            if pool == 'cls_patch':
                input_features = 1024 * 2
            else:
                input_features = 1024
            # Remove original head and add new one
            self.model.head = nn.Identity()
            self.head = nn.Linear(
                in_features=input_features,
                out_features=num_classes,
                bias=True,
            )
        self.pool = pool
        self.device = device
        self.chd = chd

    def apply_pooling(self, out):
        if self.pool == 'cls_patch':
            cls_token = out[:, 0]
            avg_patch_tokens = out[:, 1:].mean(dim=1)
            out = torch.cat([cls_token, avg_patch_tokens], dim=1)
        elif self.pool == 'patch':
            out = out[:, 1:].mean(dim=1)
        elif self.pool == 'none':
            pass
        elif self.pool == 'seg':
            out = out[:, 1:]
        else:
            out = out[:, 0]
        return out

    def forward_chd(self, x, return_features=False):
        t0 = x[:, 0:3, :, :]
        t1 = x[:, 3:6, :, :]
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            t0 = t0.unsqueeze(1)
            t1 = t1.unsqueeze(1)
            out_t0 = self.model(t0, return_all_features=True)
            out_t1 = self.model(t1, return_all_features=True)
            out_t0 = self.apply_pooling(out_t0)
            out_t1 = self.apply_pooling(out_t1)
            out = torch.cat([out_t0, out_t1], dim=1)
            if return_features:
                return out
            out = self.head(out)
        return out

    def forward(self, x, return_features=False):
        # In float 16
        if self.chd:
            return self.forward_chd(x, return_features=return_features)
        if len(x.shape) == 5:
            x = x.squeeze(1)
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            # Add temporal dimension
            x = x.unsqueeze(1)
            out = self.model(x, return_all_features=True)
            # CLS: 0, Patch: rest
            out = self.apply_pooling(out)
            if return_features:
                return out
            out = self.head(out)
        return out


def get_octcube(args, chd=False, num_frames=3):
    octcube_args = argparse.Namespace(
        model='flash_attn_vit_large_patch16',
        model_type='3D_st_flash_attn',
        ckpt='./_weights/OCTCube.pth',
        t_patch_size=3,
        num_frames=num_frames,
        input_size=args.input_size,
        patch_size=16,
        nb_classes=16,
        drop_path=0.2,
        global_pool=True,
        sep_pos_embed=True,
        cls_embed=True,
        device=args.device,
    )
    model = create_models(octcube_args)
    return OCTCubeModel(
        model,
        pool=args.pool,
        num_classes=args.num_classes,
        device=args.device,
        chd=chd,
    )


@add_config('octcube')
class OCTCubeFM(FoundModel):
    chd: bool = False
    def __init__(self, args):
        super().__init__(args)
        self.patch_size = 16
        self.embed_dim = 1024
        if args.input_size is None:
            args.input_size = 256
        if args.pool is None:
            args.pool = 'patch'
        # https://github.com/ZucksLiu/OCTCubeM/blob/main/inference_OCTCube.ipynb
        self.model = get_octcube(args, chd=self.chd)
        self.args = args

    def get_color_norm(self) -> List[Callable]:
        return [
            v2.Grayscale(num_output_channels=3),
        ]


@add_config('octcube-chd')
class OCTCubeFMChD(OCTCubeFM):
    """Adapted this to fit the format of the change detection dataset,
    where the first channel is the t0 image, and second channel is t1.
    """
    chd = True
    def get_color_norm(self) -> List[Callable]:
        return [ ChDColorNormRGB() ]



@add_config('octcube3d')
class OCTCubeFM3D(FoundModel3D):
    def __init__(self, args):
        super().__init__(args)
        self.patch_size = 16
        self.embed_dim = 1024
        if args.input_size is None:
            args.input_size = 256
        if args.pool is None:
            args.pool = 'patch'
        self.slices = 48
        self.model = get_octcube(args, num_frames=48)
        self.args = args


@add_config('octcube3d_60')
class OCTCubeFM3D60(FoundModel3D):
    def __init__(self, args):
        super().__init__(args)
        self.patch_size = 16
        self.embed_dim = 1024
        if args.input_size is None:
            args.input_size = 256
        if args.pool is None:
            args.pool = 'patch'
        self.slices = 60
        self.model = get_octcube(args, num_frames=60)
        self.args = args


#=======================================================================
# UrFound

class UrFoundModel(nn.Module):
    def __init__(self, args, chd=False):
        super().__init__()
        self.chd = chd
        self.model = model_urfound.__dict__['urmodel'](norm_pix_loss=True)
        checkpoint = torch.load(
            './_weights/urfound_mm.pth',
            map_location=args.device,
            weights_only=False,
        )['model']
        # Remove all bert_encoder weights from the checkpoint
        for key in list(checkpoint.keys()):
            if 'bert_encoder' in key:
                checkpoint.pop(key)
        msg = self.model.load_state_dict(checkpoint, strict=False)
        print(msg)
        self.pool = args.pool
        if self.pool == 'cls_patch':
            input_features = 768 * 2
        else:
            input_features = 768
        self.head = nn.Linear(
            in_features=input_features,
            out_features=args.num_classes,
        )

    def pool_features(self, features):
        if self.pool == 'cls_patch':
            cls_token = features[:, 0]
            avg_patch_tokens = features[:, 1:].mean(dim=1)
            features = torch.cat([cls_token, avg_patch_tokens], dim=1)
        elif self.pool == 'patch':
            features = features[:, 1:].mean(dim=1)
        elif self.pool == 'seg':
            features = features[:, 1:]
        else:
            features = features[:, 0]
        return features

    def forward_chd(self, x, return_features=False):
        t0 = x[:, 0:3, :, :]
        t1 = x[:, 3:6, :, :]
        out_t0 = self.model(t0)
        out_t1 = self.model(t1)
        out_t0 = self.pool_features(out_t0)
        out_t1 = self.pool_features(out_t1)
        out = torch.cat([out_t0, out_t1], dim=1)
        if return_features:
            return out
        out = self.head(out)
        return out

    def forward(self, x, return_features=False):
        if self.chd:
            return self.forward_chd(x, return_features=return_features)
        features = self.model(x)
        features = self.pool_features(features)
        if return_features:
            return features
        return self.head(features)


@add_config('urfound')
class UrFoundFM(FoundModel):
    chd: bool = False

    def __init__(self, args):
        super().__init__(args)
        self.patch_size = 16
        self.embed_dim = 768
        if args.input_size is None:
            args.input_size = 224
        if args.pool is None:
            args.pool = 'cls_patch'
        self.model = UrFoundModel(args, chd=self.chd)
        self.args = args

    def get_color_norm(self) -> List[Callable]:
        return [
            v2.Grayscale(num_output_channels=3),
        ]

    def get_model_norm(self) -> List[Callable]:
        normalize = v2.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        return [normalize]


@add_config('urfound-chd')
class UrFoundFMChD(UrFoundFM):
    """Adapted this to fit the format of the change detection dataset,
    where the first channel is the t0 image, and second channel is t1.
    """
    chd = True

    def get_color_norm(self) -> List[Callable]:
        return [ ChDColorNormRGB() ]

    def get_model_norm(self) -> List[Callable]:
        normalize = v2.Normalize(
            mean=IMAGENET_MEAN * 2,
            std=IMAGENET_STD * 2,
        )
        return [normalize]




########################################################################
# Unit tests
########################################################################


class TestFMs(unittest.TestCase):
    default_args = argparse.Namespace(
        input_size=224,
        num_classes=10,
        weight_decay=1e-2,
        lr=1e-5,
        fill=None,
        affine=True,
        pool='cls',
        device='cpu',
    )

    def assert_input_output(self, model_wrapper, num_channels=3, device='cpu'):
        model = model_wrapper.model
        model.to(device)
        self.assertIsInstance(model, nn.Module)
        dummy_input = torch.randn(
            1,
            num_channels,
            model_wrapper.args.input_size,
            model_wrapper.args.input_size,
        ).to(device)
        output = model(dummy_input)
        self.assertEqual(output.shape[1], model_wrapper.args.num_classes)

    def test_visionfm_base(self):
        args = TestFMs.default_args
        args.weights = './_weights/VisionFM-Base.pth'
        self.assert_input_output(VisionFM(args))

    def test_retfound(self):
        args = TestFMs.default_args
        args.pool = 'patch'
        self.assert_input_output(RETFoundFM(args))

    def test_octcube(self):
        args = TestFMs.default_args
        args.input_size = 256
        args.device = 'cuda'
        self.assert_input_output(OCTCubeFM(args), device='cuda')
