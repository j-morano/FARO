from typing import Iterable, Optional, Tuple, Union
from functools import partial
import os

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat


from mutils.factory import get_factory_adder


add_adapter, adapter_factory = get_factory_adder()


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path: Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0 (disabled for isotropic ConvNeXt).

    Code from: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Adapter(nn.Module):
    def __init__(self, main_tasks: Iterable[str] = ('bscan',)):
        super().__init__()
        self.main_tasks = main_tasks


@add_adapter('convnext')
class ConvNeXtAdapter(Adapter):
    """Output adapter with ConvNext blocks for semantic segmentation

    :param num_classes: Number of classes
    :param num_heads: Number of attention heads
    :param embed_dim: Token dimension after projection, and before reshaping operation.
    :param preds_per_patch: Increases size of feature map by reshaping each patch  Each patch gets reshaped
        from embed_dim x 1 x 1 to (embed_dim / preds_per_patch) x (preds_per_patch ** 0.5) x (preds_per_patch ** 0.5)
    :param main_tasks: Tasks to use for the adapter. Only tokens coming from these tasks are kept.
    :param patch_size: Size of patches
    :param depth: Number of ConvNeXt blocks
    :interpolate_mode: Interpolation mode for final upsampling
    """

    def __init__(
        self,
        num_classes,
        embed_dim: int = 6144,
        preds_per_patch: int = 16,
        main_tasks: Iterable[str] = ('bscan',),
        patch_size: Union[int, list] = 16,
        depth: int = 4,
        interpolate_mode: str = 'bilinear',
        image_size: Optional[Tuple[int, int]] = None,
        dim_tokens_enc: int = 768,
    ):
        super().__init__(main_tasks)
        self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = [patch_size, patch_size]
        self.embed_dim = embed_dim
        self.preds_per_patch = preds_per_patch
        self.class_dim = embed_dim // preds_per_patch
        self.num_classes = num_classes
        self.interpolate_mode = interpolate_mode
        self.image_size = image_size
        if isinstance(image_size, int):
            self.image_size = [image_size, image_size]
        self.proj_dec = nn.Linear(dim_tokens_enc, self.embed_dim)

        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim=self.class_dim)
            for _ in range(depth)
        ])
        self.final_layer = nn.Conv2d(self.class_dim, self.num_classes, 1)

    def forward(self, x: torch.Tensor):
        H, W = self.image_size
        N_H, N_W = H // self.patch_size[0], W // self.patch_size[1]

        x = self.proj_dec(x)
        x = rearrange(x, "b n (p c) -> b (n p) c", n=N_H * N_W, p=self.preds_per_patch, c=self.class_dim)
        x = rearrange(x, "b (nh nw ph pw) c -> b c (nh ph) (nw pw)",
                      nh=N_H, nw=N_W,
                      ph=int(self.preds_per_patch ** 0.5),
                      pw=int(self.preds_per_patch ** 0.5))
        x = self.blocks(x)
        x = self.final_layer(x)

        # Interpolate to semseg res
        x = F.interpolate(x, size=(H, W), mode=self.interpolate_mode)

        return x


@add_adapter('linear')
class LinearSegAdapter(Adapter):
    """Output adapter with a single 1x1 conv layer for semantic segmentation

    :param num_classes: Number of classes
    :param main_tasks: Tasks to use for the adapter. Only tokens coming from these tasks are kept.
    :param patch_size: Size of patches
    :interpolate_mode: Interpolation mode for final upsampling
    """

    def __init__(
        self,
        num_classes,
        main_tasks: Iterable[str] = ('bscan',),
        patch_size: Union[int, list] = 16,
        interpolate_mode: str = 'bilinear',
        task: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        dim_tokens_enc: int = 768,
    ):
        super().__init__(main_tasks)
        self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = [patch_size, patch_size]
        self.num_classes = num_classes
        self.interpolate_mode = interpolate_mode
        self.task = task
        self.image_size = image_size
        if isinstance(image_size, int):
            self.image_size = [image_size, image_size]

        self.final_layer = nn.Conv2d(dim_tokens_enc, self.num_classes, 1)

    def forward(self, x: torch.Tensor):
        H, W = self.image_size
        N_H, N_W = H // self.patch_size[0], W // self.patch_size[1]

        x = rearrange(x, 'b (nh nw) d -> b d nh nw', nh=N_H, nw=N_W)

        x = self.final_layer(x)

        # Interpolate to semseg res
        x = F.interpolate(x, size=(H, W), mode=self.interpolate_mode)

        return x


@add_adapter('cnn')
class CNNRefSegAdapter(Adapter):
    """Output adapter with a lightweight CNN + refiner for semantic segmentation.
    :param num_classes: Number of classes
    :param main_tasks: Tasks to use for the adapter. Only tokens coming from these tasks are kept.
    :param patch_size: Size of patches
    :interpolate_mode: Interpolation mode for final upsampling
    """
    def __init__(
        self,
        num_classes,
        main_tasks: Iterable[str] = ('bscan',),
        patch_size: Union[int, list] = 16,
        interpolate_mode: str = 'bilinear',
        task: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        dim_tokens_enc: int = 768,
    ):
        super().__init__(main_tasks)
        self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = [patch_size, patch_size]
        self.num_classes = num_classes
        self.interpolate_mode = interpolate_mode
        self.task = task
        self.image_size = image_size
        if isinstance(image_size, int):
            self.image_size = [image_size, image_size]

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_tokens_enc, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(),
        )
        self.final_layer = nn.Conv2d(32, self.num_classes, kernel_size=1)
        self.refiner = nn.Sequential(
            nn.Conv2d(self.num_classes, self.num_classes, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups=1, num_channels=self.num_classes),
            nn.ReLU(),
            nn.Conv2d(self.num_classes, self.num_classes, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor):
        H, W = self.image_size
        N_H, N_W = H // self.patch_size[0], W // self.patch_size[1]
        x = rearrange(x, 'b (nh nw) d -> b d nh nw', nh=N_H, nw=N_W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_layer(x)
        x = F.interpolate(x, size=(H, W), mode=self.interpolate_mode)
        x = self.refiner(x)
        return x

