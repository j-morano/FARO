from argparse import Namespace
from pathlib import Path
import sys
import argparse

import numpy as np
import torch
from torch import nn
from skimage import io
from skimage.transform import resize
from torchvision.utils import save_image
from matplotlib import pyplot as plt

from mutils.factory import get_factory_adder
from mutils.pos_embed import interpolate_pos_embed


# Import the model architecture from the parent directory

parent_dir = str(Path(__file__).resolve().parent.parent)
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.architecture import get_model_architecture  # type: ignore



add_miragecls, miragecls_factory = get_factory_adder()


@add_miragecls('patch')
class MIRAGEClsGlobal(nn.Module):
    def __init__(
        self,
        input_size=512,
        patch_size=32,
        num_classes=0,
        modalities='bscan',
        weights=None,
        device='cuda',
        model_size='base',
        num_global_tokens=10,
        chd=False,
    ):
        super().__init__()
        assert num_classes > 0

        grid_size = input_size // patch_size

        self.chd = chd
        self.args = Namespace(
            in_domains=["bscan"],
            out_domains=[],  # NOTE: No output, just features
            standardize_depth=True,
            extra_norm_pix_loss=False,
            model="pretrain_multimae_base",  # only encoder
            decoder_dim=256,
            input_size={modalities: [input_size, input_size]},
            patch_size={modalities: [patch_size, patch_size]},
            grid_sizes={modalities: [grid_size, grid_size]},
            sample_only=modalities,
            alphas=1.0,
            num_encoded_tokens=64,
            num_global_tokens=num_global_tokens,
            decoder_use_task_queries=True,
            decoder_depth=2,
            weights=weights,
            decoder_num_heads=8,
            decoder_use_xattn=True,
            drop_path=0.0,
            custom_sampling=False,
        )
        if model_size == 'large':
             self.args.model = "pretrain_multimae_large"

        vm = get_model_architecture(self.args)
        vm.force_no_shuffle=True

        checkpoint = torch.load(self.args.weights, map_location=device, weights_only=False)["model"]
        try:
            msg = vm.load_state_dict(checkpoint, strict=False)
        except RuntimeError:
            print("Interpolating position embedding...")
            interpolate_pos_embed(vm, checkpoint)
            msg = vm.load_state_dict(checkpoint, strict=False)
        print('Missing keys:', msg.missing_keys)
        assert len(msg.missing_keys) == 0, "THERE ARE MISSING KEYS IN WEIGHTS!"
        vm.to(device)

        self.model = vm

        self.num_classes = num_classes
        # Get the embedding dimension from the first layer norm of the
        #   type: (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.embed_dim = self.model.encoder[0].norm1.normalized_shape[0]
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-06, elementwise_affine=True)
        print('  Embedding dimension:', self.embed_dim)
        self.build_head()

    def build_head(self, factor=1):
        self.head = nn.Linear(self.embed_dim * factor, self.num_classes)

    def forward_chd(self, x, return_features=False):
        # x in this case is an RGB image where 0 is the first image, 1
        #   is the second image.
        x_1 = x[:, 0:1, :, :]
        x_2 = x[:, 1:2, :, :]
        x_1_d = {self.args.in_domains[0]: x_1}
        x_2_d = {self.args.in_domains[0]: x_2}
        out_1, _masks_1 = self.model(x_1_d, mask_inputs=False)
        out_2, _masks_2 = self.model(x_2_d, mask_inputs=False)
        out_1 = self.norm(out_1)
        out_2 = self.norm(out_2)
        out_1 = self.pool(out_1)
        out_2 = self.pool(out_2)
        out = torch.cat([out_1, out_2], dim=1)
        if return_features:
            return out
        return self.head(out)

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, C, H, W) tensor. H and W are determined by the
            input_size parameter in the constructor. It expects a tensor
            in the range [0, 1].
        Returns:
            (B, C, H, W) tensor
        """
        if self.chd:
            return self.forward_chd(x, return_features=return_features)
        if not isinstance(x, dict):
            x_d = {self.args.in_domains[0]: x}
        else:
            x_d = x
        # for k, v in x_d.items():
        #     print(f"Input {k} shape: {v.shape}, min: {v.min()}, max: {v.max()}")
        out, _masks = self.model(x_d, mask_inputs=False, force_no_shuffle=True)
        out = self.norm(out)
        out = self.pool(out)
        if return_features:
            return out
        return self.head(out)

    def pool(self, x):
        return x[:, :-self.args.num_global_tokens, :].mean(dim=1)

    def get_output_adapters(self):
        return None


@add_miragecls('seg_lesion')
class MIRAGEClsCLSSeg(MIRAGEClsGlobal):
    def build_head(self, factor=1):
        self.head = nn.Identity()
    def pool(self, x):
        # Return tuple of patch tokens and lesion token
        patch = x[:, :-self.args.num_global_tokens, :]
        global_ = x[:, -self.args.num_global_tokens:, :]
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        return patch, lesion_presence


@add_miragecls('seg_all')
class MIRAGEClsCLSSegAll(MIRAGEClsGlobal):
    def build_head(self, factor=1):
        self.head = nn.Identity()
    def pool(self, x):
        # Return tuple of patch tokens and lesion token
        patch = x[:, :-self.args.num_global_tokens, :]
        global_ = x[:, -self.args.num_global_tokens:, :]
        return patch, global_


@add_miragecls('seg')
class MIRAGEClsCLSSeg(MIRAGEClsGlobal):
    def build_head(self, factor=1):
        self.head = nn.Identity()
    def pool(self, x):
        return x[:, :-self.args.num_global_tokens, :]


@add_miragecls('cls')
class MIRAGEClsCLS(MIRAGEClsGlobal):
    def pool(self, x):
        return x[:, -self.args.num_global_tokens, :]


@add_miragecls('token_mix')
class MIRAGEClsTokenMix(MIRAGEClsGlobal):
    def build_head(self, factor=2):
        super().build_head(factor)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :].mean(dim=1)
        return torch.cat([patch, global_], dim=1)


@add_miragecls('ultra_mix')
class MIRAGEClsUltraMix(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=9)

    def pool(self, x):
        # Concatenate mean patch token and all global tokens
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :].flatten(start_dim=1)
        return torch.cat([patch, global_], dim=1)
        # 'laterality': PredictionHead(dim_tokens, num_classes=2, index=1),
        # 'device': PredictionHead(dim_tokens, num_classes=2, index=2),
        # 'position': PredictionHead(dim_tokens, num_classes=1, index=3),
        # 'lesion_presence': PredictionHead(dim_tokens, num_classes=7, index=4),


# @add_miragecls('mini_mix')
# class MIRAGEClsMiniMix(MIRAGEClsGlobal):
#     def build_head(self, factor=None):
#         super().build_head(factor=3)

#     def pool(self, x):
#         # Concatenate mean patch token and all global tokens
#         patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
#         global_ = x[:, -self.args.num_global_tokens:, :]
#         cls_ = global_[:, 0, :].flatten(start_dim=1)
#         position = global_[:, 3, :].flatten(start_dim=1)
#         lesion_presence = global_[:, 4, :].flatten(start_dim=1)
#         return torch.cat([patch, position, lesion_presence], dim=1)
#         # 'laterality': PredictionHead(dim_tokens, num_classes=2, index=1),
#         # 'device': PredictionHead(dim_tokens, num_classes=2, index=2),
#         # 'position': PredictionHead(dim_tokens, num_classes=1, index=3),
#         # 'lesion_presence': PredictionHead(dim_tokens, num_classes=7, index=4),


# @add_miragecls('super_mix')
# class MIRAGEClsSuperMix(MIRAGEClsGlobal):
#     def build_head(self, factor=None):
#         super().build_head(factor=3)

#     def pool(self, x):
#         patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
#         global_ = x[:, -self.args.num_global_tokens:, :]
#         # cls_ = global_[:, 0, :].flatten(start_dim=1)
#         # laterality = global_[:, -1, :].flatten(start_dim=1)
#         # device = global_[:, -2, :].flatten(start_dim=1)
#         position = global_[:, -3, :].flatten(start_dim=1)
#         lesion_presence = global_[:, -4, :].flatten(start_dim=1)
#         return torch.cat([patch, position, lesion_presence], dim=1)

#         # 'cls': PredictionHead(dim_tokens, num_classes=256, index=0),
#         # 'laterality': PredictionHead(dim_tokens, num_classes=2, index=-1),
#         # 'device': PredictionHead(dim_tokens, num_classes=2, index=-2),
#         # 'position': PredictionHead(dim_tokens, num_classes=1, index=-3),
#         # 'lesion_presence': PredictionHead(dim_tokens, num_classes=7, index=-4),



@add_miragecls('patch_cls')
class MIRAGEPatchCls(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, -1, :].flatten(start_dim=1)
        # device = global_[:, -2, :].flatten(start_dim=1)
        # position = global_[:, -3, :].flatten(start_dim=1)
        # lesion_presence = global_[:, -4, :].flatten(start_dim=1)
        return torch.cat([patch, cls_], dim=1)


@add_miragecls('con_mix')
class MIRAGEClsConMix(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=4)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, -1, :].flatten(start_dim=1)
        # device = global_[:, -2, :].flatten(start_dim=1)
        position = global_[:, -3, :].flatten(start_dim=1)
        lesion_presence = global_[:, -4, :].flatten(start_dim=1)
        return torch.cat([patch, cls_, position, lesion_presence], dim=1)

        # 'cls': PredictionHead(dim_tokens, num_classes=256, index=0),
        # 'laterality': PredictionHead(dim_tokens, num_classes=2, index=-1),
        # 'device': PredictionHead(dim_tokens, num_classes=2, index=-2),
        # 'position': PredictionHead(dim_tokens, num_classes=1, index=-3),
        # 'lesion_presence': PredictionHead(dim_tokens, num_classes=7, index=-4),


@add_miragecls('PaLe_mix')
# DEPRECATED
class MIRAGEPaLeMix(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, -1, :].flatten(start_dim=1)
        # device = global_[:, -2, :].flatten(start_dim=1)
        # position = global_[:, -3, :].flatten(start_dim=1)
        lesion_presence = global_[:, -4, :].flatten(start_dim=1)
        return torch.cat([patch, lesion_presence], dim=1)


@add_miragecls('width_mix')
class MIRAGEClsWidthMix(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=5)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, cls_, position, width, lesion_presence], dim=1)


########################################################################
# CONW


@add_miragecls('conw_PaCLaDPoLeW')
class MIRAGEConWPaCLaDPoLeW(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=7)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        laterality = global_[:, 1, :].flatten(start_dim=1)
        device = global_[:, 2, :].flatten(start_dim=1)
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, cls_, laterality, device, position, lesion_presence, width], dim=1)


@add_miragecls('conw_PaCLaPoLeW')
class MIRAGEConWPaCLaPoLeW(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=6)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, cls_, laterality, position, lesion_presence, width], dim=1)


@add_miragecls('conw_PaCPoLeW')
class MIRAGEConWPaCPoLeW(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=5)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, cls_, position, lesion_presence, width], dim=1)


@add_miragecls('conw_PaCPoLe')
class MIRAGEConWPaCPoLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=4)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, cls_, position, lesion_presence], dim=1)


@add_miragecls('conw_PaPoLe')
class MIRAGEConWPaPoLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, position, lesion_presence], dim=1)


@add_miragecls('conw_PaLe')
class MIRAGEConWPaLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, lesion_presence], dim=1)



# @add_miragecls('conw_PaLe_ChD')
class MIRAGEConWPaLeChD(MIRAGEClsGlobal):
    # NOTE: This model is for change detection
    def build_head(self, factor=None):
        super().build_head(factor=4)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, lesion_presence], dim=1)

    def forward(self, x, return_features=False):
        # x in this case is an RGB image where 0 is the first image, 1
        #   is the second image.
        x_1 = x[:, 0:1, :, :]
        x_2 = x[:, 1:2, :, :]
        x_1_d = {self.args.in_domains[0]: x_1}
        x_2_d = {self.args.in_domains[0]: x_2}
        out_1, _masks_1 = self.model(x_1_d, mask_inputs=False)
        out_2, _masks_2 = self.model(x_2_d, mask_inputs=False)
        out_1 = self.norm(out_1)
        out_2 = self.norm(out_2)
        out_1 = self.pool(out_1)
        out_2 = self.pool(out_2)
        out = torch.cat([out_1, out_2], dim=1)
        if return_features:
            return out
        return self.head(out)


@add_miragecls('conw_Pa')
class MIRAGEConWPa(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=1)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        # global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        # lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch], dim=1)


@add_miragecls('conw_Le')
class MIRAGEConWLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=1)

    def pool(self, x):
        # patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([lesion_presence], dim=1)


@add_miragecls('conw_C')
class MIRAGEConWC(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=1)

    def pool(self, x):
        # patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        # lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([cls_], dim=1)


########################################################################
# PBWT

@add_miragecls('pbwt_PaThLe')
class MIRAGEPBWTPaThLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        thicknesses = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, thicknesses, lesion_presence], dim=1)



@add_miragecls('pbwt_Th')
class MIRAGEPBWTTh(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=1)

    def pool(self, x):
        # patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        thicknesses = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        # lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([thicknesses], dim=1)


@add_miragecls('pbwt_PaLe')
class MIRAGEPBWTPaLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # thicknesses = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, lesion_presence], dim=1)


@add_miragecls('pbwt_CPaLe')
class MIRAGEPBWTCPaLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([cls_, patch, lesion_presence], dim=1)


@add_miragecls('pbwt_PaPosLe')
class MIRAGEPBWTPaPosLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        return torch.cat([patch, position, lesion_presence], dim=1)


@add_miragecls('pbwt_C')
class MIRAGEPBWTC(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=1)

    def pool(self, x):
        # patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        # lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([cls_], dim=1)


@add_miragecls('pbwt_Pa')
class MIRAGEPBWTPa(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=1)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        # global_ = x[:, -self.args.num_global_tokens:, :]
        # thicknesses = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        # lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch], dim=1)


@add_miragecls('pbwt_Le')
class MIRAGEPBWTLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=1)

    def pool(self, x):
        # patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # thicknesses = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([lesion_presence], dim=1)


########################################################################
# IMPORTANT: PBWT PROPER ABLATION

# 'thicknesses': PredictionHead(dim_tokens, num_classes=22, index=0),
# 'laterality': PredictionHead(dim_tokens, num_classes=2, index=1),
# 'device': PredictionHead(dim_tokens, num_classes=2, index=2),
# 'position': PredictionHead(dim_tokens, num_classes=1, index=3),
# 'lesion_presence': PredictionHead(dim_tokens, num_classes=7, index=4),
# 'width_mm': PredictionHead(dim_tokens, num_classes=1, index=5),


@add_miragecls('MPBWT_AllAvg')
class MPBWTAllAvg(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :].mean(dim=1)
        return torch.cat([patch, global_], dim=1)


@add_miragecls('MPBWT_PaLe')
class MPBWTPaLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        return torch.cat([patch, lesion_presence], dim=1)

@add_miragecls('MPBWT_PaNonLe')
class MPBWTNonPaLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        global_minus_lesions = torch.cat([global_[:, :4, :], global_[:, 5:, :]], dim=1)
        global_minus_lesions = global_minus_lesions.mean(dim=1)
        return torch.cat([patch, global_minus_lesions], dim=1)


@add_miragecls('MPBWT_PaPos')
class MPBWTPaPos(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        position = global_[:, 3, :].flatten(start_dim=1)
        return torch.cat([patch, position], dim=1)


@add_miragecls('MPBWT_PaTh')
class MPBWTPaTh(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        thicknesses = global_[:, 0, :].flatten(start_dim=1)
        return torch.cat([patch, thicknesses], dim=1)


@add_miragecls('MPBWT_PaWi')
class MPBWTPaWi(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, width], dim=1)


@add_miragecls('MPBWT_PaPa')
class MPBWTPaPa(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        return torch.cat([patch, patch], dim=1)


@add_miragecls('MPBWT_PaPaPa')
class MPBWTPaPaPa(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        return torch.cat([patch, patch, patch], dim=1)


@add_miragecls('MPBWT_PaLeTh')
class MPBWTPaLeTh(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        thicknesses = global_[:, 0, :].flatten(start_dim=1)
        return torch.cat([patch, lesion_presence, thicknesses], dim=1)


@add_miragecls('MPBWT_PaLeLe')
class MPBWTPaLeLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        lesion_presence_dup = lesion_presence
        return torch.cat([patch, lesion_presence, lesion_presence_dup], dim=1)



@add_miragecls('MPBWT_PaThTh')
class MPBWTPaThTh(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        thicknesses = global_[:, 0, :].flatten(start_dim=1)
        thicknesses_dup = thicknesses
        return torch.cat([patch, thicknesses, thicknesses_dup], dim=1)


@add_miragecls('MPBWT_LeLeLe')
class MPBWTLeLeLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        # patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        lesion_presence_dup1 = lesion_presence
        lesion_presence_dup2 = lesion_presence
        return torch.cat([lesion_presence, lesion_presence_dup1, lesion_presence_dup2], dim=1)


@add_miragecls('MPBWT_ThThTh')
class MPBWTThThTh(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        # patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        thicknesses = global_[:, 0, :].flatten(start_dim=1)
        thicknesses_dup1 = thicknesses
        thicknesses_dup2 = thicknesses
        return torch.cat([thicknesses, thicknesses_dup1, thicknesses_dup2], dim=1)



########################################################################
# CONWMST

@add_miragecls('conwmst_PaLe')
class MIRAGEConWMSTPaLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :]
        # laterality = global_[:, 1, :]
        # device = global_[:, 2, :]
        # position = global_[:, 3, :]
        lesion_presence = global_[:, 4, :]
        # width = global_[:, 5, :]
        # thicknesses = global_[:, 6, :]
        return torch.cat([patch, lesion_presence], dim=1)

@add_miragecls('conwmst_PaCThLe')
class MIRAGEConWMSTPaCThLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=4)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :]
        # laterality = global_[:, 1, :]
        # device = global_[:, 2, :]
        # position = global_[:, 3, :]
        lesion_presence = global_[:, 4, :]
        # width = global_[:, 5, :]
        thicknesses = global_[:, 6, :]
        return torch.cat([patch, cls_, thicknesses, lesion_presence], dim=1)


@add_miragecls('conwmst_PaThLe')
class MIRAGEConWMSTPaThLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :]
        # laterality = global_[:, 1, :]
        # device = global_[:, 2, :]
        # position = global_[:, 3, :]
        lesion_presence = global_[:, 4, :]
        # width = global_[:, 5, :]
        thicknesses = global_[:, 6, :]
        return torch.cat([patch, thicknesses, lesion_presence], dim=1)


@add_miragecls('conwmst_Le')
class MIRAGEConWMSTLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=1)

    def pool(self, x):
        # patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :]
        # laterality = global_[:, 1, :]
        # device = global_[:, 2, :]
        # position = global_[:, 3, :]
        lesion_presence = global_[:, 4, :]
        # width = global_[:, 5, :]
        # thicknesses = global_[:, 6, :]
        return torch.cat([lesion_presence], dim=1)


########################################################################

@add_miragecls('abl_width_PaCLaDePoLeW')
class MIRAGEClsAblWidthPaCLaDePoLeW(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=7)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        laterality = global_[:, 1, :].flatten(start_dim=1)
        device = global_[:, 2, :].flatten(start_dim=1)
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, cls_, laterality, device, position, lesion_presence, width], dim=1)


@add_miragecls('abl_width_PaCLaPoLeW')
class MIRAGEClsAblWidthPaCLaPoLeW(MIRAGEClsAblWidthPaCLaDePoLeW):
    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        width = global_[:, 5, :].flatten(start_dim=1)
        zeros = torch.zeros_like(patch)
        return torch.cat([patch, cls_, laterality, zeros, position, lesion_presence, width], dim=1)


@add_miragecls('abl_width_PaCPoLeW')
class MIRAGEClsAblWidthPaCPoLeW(MIRAGEClsAblWidthPaCLaDePoLeW):
    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        width = global_[:, 5, :].flatten(start_dim=1)
        zeros = torch.zeros_like(patch)
        return torch.cat([patch, cls_, zeros, zeros, position, lesion_presence, width], dim=1)


@add_miragecls('abl_width_PaCPoLe')
class MIRAGEClsAblWidthPaCPoLe(MIRAGEClsAblWidthPaCLaDePoLeW):
    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        zeros = torch.zeros_like(patch)
        return torch.cat([patch, cls_, zeros, zeros, position, lesion_presence, zeros], dim=1)


@add_miragecls('abl_width_PaPoLe')
class MIRAGEClsAblWidthPaPoLe(MIRAGEClsAblWidthPaCLaDePoLeW):
    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        zeros = torch.zeros_like(patch)
        return torch.cat([patch, zeros, zeros, zeros, position, lesion_presence, zeros], dim=1)


@add_miragecls('abl_width_PaLe')
class MIRAGEClsAblWidthPaLe(MIRAGEClsAblWidthPaCLaDePoLeW):
    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        zeros = torch.zeros_like(patch)
        return torch.cat([patch, zeros, zeros, zeros, zeros, lesion_presence, zeros], dim=1)


@add_miragecls('w_PaLe')
class MIRAGEClswPaLe(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        return torch.cat([patch, lesion_presence], dim=1)



@add_miragecls('abl_width_Le')
class MIRAGEClsAblWidthLe(MIRAGEClsAblWidthPaCLaDePoLeW):
    def pool(self, x):
        # patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        zeros = torch.zeros_like(lesion_presence)
        return torch.cat([zeros, zeros, zeros, zeros, zeros, lesion_presence, zeros], dim=1)


@add_miragecls('abl_width_PaCLe')
class MIRAGEClsAblWidthPaCLe(MIRAGEClsAblWidthPaCLaDePoLeW):
    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        zeros = torch.zeros_like(patch)
        return torch.cat([patch, cls_, zeros, zeros, zeros, lesion_presence, zeros], dim=1)


@add_miragecls('abl_width_PaC')
class MIRAGEClsAblWidthPaC(MIRAGEClsAblWidthPaCLaDePoLeW):
    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        # lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        zeros = torch.zeros_like(patch)
        return torch.cat([patch, cls_, zeros, zeros, zeros, zeros, zeros], dim=1)


@add_miragecls('abl_width_Pa')
class MIRAGEClsAblWidthPa(MIRAGEClsAblWidthPaCLaDePoLeW):
    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        # global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, 1, :].flatten(start_dim=1)
        # device = global_[:, 2, :].flatten(start_dim=1)
        # position = global_[:, 3, :].flatten(start_dim=1)
        # lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        # width = global_[:, 5, :].flatten(start_dim=1)
        zeros = torch.zeros_like(patch)
        return torch.cat([patch, zeros, zeros, zeros, zeros, zeros, zeros], dim=1)


@add_miragecls('patch_lesions')
class MIRAGEPatchLesionsMix(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        # cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, -1, :].flatten(start_dim=1)
        # device = global_[:, -2, :].flatten(start_dim=1)
        # position = global_[:, -3, :].flatten(start_dim=1)
        lesion_presence = global_[:, -4, :].flatten(start_dim=1)
        return torch.cat([patch, lesion_presence], dim=1)


@add_miragecls('patch_cls_lesions')
class MIRAGEPatchClsLesionsMix(MIRAGEClsGlobal):
    def build_head(self, factor=None):
        super().build_head(factor=3)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        cls_ = global_[:, 0, :].flatten(start_dim=1)
        # laterality = global_[:, -1, :].flatten(start_dim=1)
        # device = global_[:, -2, :].flatten(start_dim=1)
        # position = global_[:, -3, :].flatten(start_dim=1)
        lesion_presence = global_[:, -4, :].flatten(start_dim=1)
        return torch.cat([patch, cls_, lesion_presence], dim=1)



def to_tensor(fn):
    fn = str(fn)
    if fn.endswith('.jpeg') or fn.endswith('.jpg') or fn.endswith('.png'):
        img = io.imread(fn)
        if img.ndim == 3:
            img = img[..., 0]
    elif fn.endswith('.npy'):
        img = np.load(fn)
    else:
        raise ValueError('Unsupported file format:', fn.split('.')[-1])
    if 'layermap' in fn:
        img = resize(img, (128, 128), order=0, preserve_range=True, anti_aliasing=False)
        img = torch.tensor(img).unsqueeze(0).long()
    else:
        img = resize(img, (512, 512), order=1, preserve_range=True, anti_aliasing=True)
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
        # Normalize to [0, 1]
        img = img / 255.0
    print('Input:', Path(fn).stem, img.dtype, img.shape, img.min(), img.max())
    return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', action='store_true', help='Extract features only')
    parser.add_argument('--model_size', type=str, default='base', help='Model size', choices=['base', 'large'])
    parser.add_argument('--image_path', type=str, default='./_example_images', help='Path to input images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # NOTE: ViT-Base and ViT-Large versions of MIRAGE are available
    if args.model_size == 'base':
        weights = './_weights/MIRAGE-Base.pth'
    else:
        weights = './_weights/MIRAGE-Large.pth'

    model = MIRAGEWrapper(weights=weights, device=args.device)
    model.eval()
    if args.features:
        model.model.output_adapters = None

    print(f'Using device: {model.device}')

    for fsid in Path(args.image_path).iterdir():
        bscan = to_tensor(fsid / 'bscan.npy')
        slo = to_tensor(fsid / 'slo.npy')
        bscanlayermap = to_tensor(fsid / 'bscanlayermap.npy')

        # NOTE: uncomment to test with different input modalities
        input_data = {
            'bscan': bscan,
            # 'slo': slo,
            # 'bscanlayermap': bscanlayermap,
        }

        with torch.no_grad():
            out = model(input_data)
            if args.features:
                print(out.shape)
                np.save(fsid / f'__out_features.npy', out.cpu().numpy())
            else:
                print('Outputs:')
                for k, v in out.items():
                    print('\t', k, v.shape, v.min(), v.max())
                    if 'layermap' in k:
                        v = v.argmax(1) / 12
                    save_image(v, fsid / f'__out_{k}.png')
