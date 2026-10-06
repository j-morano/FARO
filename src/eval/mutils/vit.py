from functools import partial

import torch
import torch.nn as nn
import timm.models.vision_transformer as vit



class PatchPooler:
    """Global avg. pooling without CLS token."""
    def __call__(self, x):
        return x[:, 1:, :].mean(dim=1)

class TokenMixPooler:
    """Token mixing: CLS token + global avg. pooling of patch tokens."""
    def __call__(self, x):
        cls_token = x[:, 0]
        patch_token = x[:, 1:].mean(dim=1)
        return torch.cat((cls_token, patch_token), dim=1)

class CLSPooler:
    """CLS token."""
    def __call__(self, x):
        return x[:, 0]

class SegPooler:
    """Segmentation pooler, returning all patch features."""
    def __call__(self, x):
        return x[:, 1:, :]


class VisionTransformer(vit.VisionTransformer):
    """Vision Transformer with support for different pooling methods."""

    def __init__(self, pool='global', chd=False, **kwargs):
        super().__init__(**kwargs)

        self.pool = pool
        self.chd = chd
        # NOTE: Make sure these are None, otherwise the forward method
        #   will try to use them.
        assert self.head_dist is None
        assert self.dist_token is None
        if self.pool == 'patch':
            print("ViT: Using global patch pool")
            self.pooler = PatchPooler()
        elif self.pool == 'cls_patch':
            print("ViT: Using CLS+Patch")
            self.head = nn.Linear(
                in_features=kwargs["embed_dim"] * 2,
                out_features=kwargs["num_classes"],
                bias=True
            )
            self.pooler = TokenMixPooler()
        elif self.pool == 'seg':
            print("ViT: Using segmentation pooler")
            self.pooler = SegPooler()
        else:
            print("ViT: Using CLS pooler")
            self.pooler = CLSPooler()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.pooler(x)

    def forward_chd(self, x):
        x_1 = x[:, 0:3, :, :]
        x_2 = x[:, 3:6, :, :]
        x_1 = self.forward_features(x_1)
        x_2 = self.forward_features(x_2)
        return torch.cat([x_1, x_2], dim=1)

    def forward(self, x, return_features=False):
        if self.chd:
            x = self.forward_chd(x)
        else:
            x = self.forward_features(x)
        if return_features:
            return x
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

