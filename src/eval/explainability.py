from pathlib import Path
from functools import partial
from typing import Optional, Union, Tuple, Dict
from dataclasses import dataclass
from argparse import Namespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
import matplotlib.pyplot as plt


MAX_SLICES = 256



@dataclass
class ModelArgs:
    """Configuration for the model components.
        - dim: The hidden dimension for the resampler and lesion
            aggregator.
        - num_queries: The number of query tokens used in the resampler.
        - resampler: The type of resampler to use.
        - lesion_pool: The method for aggregating lesion presence across
            slices.
    """
    num_queries: int = 8
    resampler: str = "xattn"
    lesion_pool: str = "attnpool"


def pair(t):
    if t is None:
        return None
    if isinstance(t, tuple):
        return t
    elif isinstance(t, list):
        return tuple(t)
    else:
        return (t, t)


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3

    Source: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = rearrange(pos_emb, 'b (h w) d -> b d h w', h=h, w=w, d=embed_dim)
    return pos_emb


class PatchedInputAdapter(nn.Module):
    def __init__(
        self,
        num_channels: int,
        stride_level: int,
        patch_size_full: Union[int, Tuple[int,int]],
        dim_tokens: Optional[int] = None,
        sincos_pos_emb: bool = True,
        learnable_pos_emb: bool = False,
        image_size: Union[int, Tuple[int]] = 224,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        print(f'Image size: {self.image_size}, Patch size: {self.patch_size_full}')
        self.num_patches = (self.image_size[0] // patch_size_full[0]) * (self.image_size[1] // patch_size_full[1])

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens: Dimension of tokens
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if self.sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, h_posemb, w_posemb))

        # Image -> tokens projection
        self.proj = nn.Conv2d(
            in_channels=self.num_channels, out_channels=self.dim_tokens,
            kernel_size=(self.P_H, self.P_W), stride=(self.P_H, self.P_W)
        )

    def forward(self, x):
        """
        Forward pass through input adapter, transforming image to sequence of tokens.
        Adds task and positional encodings.

        :param x: Input image tensor
        """
        B, C, H, W = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        assert (H % self.P_H == 0) and (W % self.P_W == 0), f'Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}'
        N_H, N_W = H // self.P_H, W // self.P_W # Number of patches in height and width

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = rearrange(self.proj(x), 'b d nh nw -> b (nh nw) d')

        # Create positional embedding
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bicubic', align_corners=False)
        x_pos_emb = rearrange(x_pos_emb, 'b d nh nw -> b (nh nw) d')

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        return x


DOMAIN_CONF = {
    'bscan': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
    }
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PredictionHead(nn.Module):
    def __init__(self, dim_tokens, num_classes, index, hidden_dim=None):
        super().__init__()
        # Use a 2-layer MLP for better feature extraction
        hidden_dim = hidden_dim or dim_tokens
        self.head = nn.Sequential(
            nn.Linear(dim_tokens, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.index = index

    def forward(self, x):
        return self.head(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.transpose = True
        self.return_attn = False  # toggle to capture attention weights

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.return_attn:
            # Manual attention computation to expose the weights
            # (F.scaled_dot_product_attention does not return them)
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            attn = attn.softmax(dim=-1)
            self.last_attn = attn.detach()  # stash for retrieval after forward
            out = attn @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.attn_drop)

        if self.transpose:
            out = out.transpose(1, 2)
        out = out.reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MIRAGEPredLesionPresenceOnly(nn.Module):
    def __init__(
        self,
        args,
        input_adapters: Dict[str, nn.Module],
        num_global_tokens: int = 1,
        dim_tokens: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6) # type: ignore
    ):
        super().__init__()

        self.args = args

        # Initialize input and output adapters
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens)
        self.input_adapters = nn.ModuleDict(input_adapters)

        self.dim_tokens = dim_tokens

        # Additional learnable tokens that can be used by encoder to process/store global information
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens))

        # Transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.encoder = nn.Sequential(*[
            Block(dim=dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])

    def get_num_layers(self):
        return len(self.encoder)

    def forward(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]):
        # If input x is a Tensor, assume it's bscan
        x = {'bscan': x} if isinstance(x, torch.Tensor) else x

        B, C, H, W = list(x.values())[0].shape

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        # Add this snippet
        task_masks = {
            domain: torch.zeros((B, tensor.shape[1]), device=tensor.device)
            for domain, tensor in input_task_tokens.items()
        }

        ## Generating masks
        mask_all = torch.cat([task_masks[task] for task in input_task_tokens.keys()], dim=1)
        ids_shuffle = torch.argsort(mask_all, dim=1)
        _ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :(mask_all == 0).sum()]

        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)

        # Apply mask
        input_tokens = torch.gather(input_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2]))

        # Add global tokens to input tokens
        global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B)
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        ## Transformer forward pass
        encoder_tokens = self.encoder(input_tokens)
        # Forward extra modules using the global tokens
        return encoder_tokens


def pretrain_miragev2leonly_base(
        input_adapters: Dict[str, nn.Module],
        args,
        **kwargs):
    model = MIRAGEPredLesionPresenceOnly(
        args,
        input_adapters=input_adapters,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def get_model_architecture(args):
    """Creates and returns model from arguments
    """
    print(f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}")
    if isinstance(args.patch_size, int):
        args.patch_size = {domain: (args.patch_size, args.patch_size) for domain in args.all_domains}

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=tuple(args.patch_size[domain]),
            image_size=args.input_size[domain],
        )
        for domain in args.in_domains
    }

    model = pretrain_miragev2leonly_base(
        input_adapters=input_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
        args=args,
    )

    return model


def load_vm(
    device,
    model="pretrain_miragev2leonly_base",
    weights="_weights/MIRAGEOCT-PBnofftLeOnly-014.pth",
) -> nn.Module:
    vm_args = Namespace(
        in_domains=["bscan"],
        out_domains=[],  # NOTE: No output, just features
        standardize_depth=True,
        extra_norm_pix_loss=False,
        # model="pretrain_miragev2wt_base",
        model=model,
        decoder_dim=256,
        input_size={"bscan": [512, 512]},
        patch_size={"bscan": [32, 32]},
        grid_sizes={"bscan": [16, 16]},
        sample_only='bscan',
        alphas=1.0,
        num_encoded_tokens=64,
        num_global_tokens=10,
        decoder_use_task_queries=True,
        decoder_depth=2,
        # weights="_weights/__MIRAGEOCT-PBWT-014.pth",  # old version
        weights=weights,
        decoder_num_heads=8,
        decoder_use_xattn=True,
        drop_path=0.0,
        custom_sampling=False,
    )

    vm = get_model_architecture(vm_args)

    checkpoint = torch.load(vm_args.weights, map_location=device, weights_only=False)
    msg = vm.load_state_dict(checkpoint["model"], strict=False)
    print('Missing keys:', msg.missing_keys)
    vm.to(device)

    return vm


class LesionAggregatorXAttnV2(nn.Module):
    def __init__(self, dim=768, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, 2, dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.ln_post = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.pos_z_embed = nn.Embedding(MAX_SLICES, dim)
        print("Using cross-attention for lesion aggregation!")

    def forward(self, x):
        """
        x: [B, S, D]
        """
        B, S, D = x.shape
        pos_ids = torch.linspace(
            0,
            MAX_SLICES-1,
            steps=S,
            device=x.device,
        ).round().long()
        x = x + self.pos_z_embed(pos_ids)[None]

        # Cross-Attention: Q hunts in the slice tokens (K, V)
        attn_out, _ = self.cross_attn(
            query=self.ln_q(self.queries),
            key=self.ln_kv(x),
            value=self.ln_kv(x)
        )
        x_q = self.queries + attn_out
        x_q = x_q + self.mlp(self.ln_post(x_q))

        return x_q  # [1, 2, d]

class VisualResamplerV2(nn.Module):
    def __init__(
        self,
        dim=1024,
        num_queries=16,
        num_heads=8,
        mlp_ratio=4.0,
        grid_size=16,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))

        self.pos_z_embed = nn.Embedding(MAX_SLICES, dim)
        self.pos_xy_embed = nn.Parameter(torch.randn(1, 1, grid_size ** 2, dim) * 0.02)

        # Block 1: Cross-Attention (Existing)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_q1 = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.ln_post1 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

        # Block 2: Self-Attention (New - for matching depth/params)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_q2 = nn.LayerNorm(dim)
        self.ln_post2 = nn.LayerNorm(dim)
        print("Using cross-attention (v2 with self-attention) for resampler.")

    def forward(self, x):
        """Batched forward.
        Args:
            x: patch tokens [B, S, P(HxW), D]
        """
        B, S, P, D = x.shape

        # NOTE: using positional embeddings for each slice and patch.
        pos_ids = torch.linspace(
            0,
            MAX_SLICES-1,
            steps=S,
            device=x.device,
        ).round().long()
        x = x + self.pos_z_embed(pos_ids)[None, :, None, :]
        x = x + self.pos_xy_embed

        x = rearrange(x, 'b s p d -> b (s p) d')
        q = self.queries.expand(B, -1, -1)

        # Block 1
        attn1, _ = self.cross_attn(
            query=self.ln_q1(q),
            key=self.ln_kv(x),
            value=self.ln_kv(x),
        )
        q = q + attn1
        q = q + self.mlp(self.ln_post1(q))

        # Block 2 (Self-Attention allows queries to interact)
        attn2, _ = self.self_attn(
            query=self.ln_q2(q),
            key=self.ln_q2(q),
            value=self.ln_q2(q),
        )
        q = q + attn2
        q = q + self.mlp(self.ln_post2(q))

        return q


class VisionRunner:
    def __init__(self, args: ModelArgs, device="cuda"):
        super().__init__()
        print(f"Using {MAX_SLICES} slices as max!")
        self.args = args
        self.device = device
        self.vm = load_vm(device)
        self.resampler = VisualResamplerV2(
            dim=self.vm.dim_tokens, num_queries=args.num_queries,
        ).to(device).to(torch.bfloat16)
        self.lesion_aggregator = LesionAggregatorXAttnV2(
            dim=self.vm.dim_tokens,
        ).to(device).to(torch.bfloat16)
        print("Post-training components initialized successfully.")

    def get_vm_features(self, bscan_tensor):
        """Get features from the vision model, including patch tokens
        and aggregated lesion presence.
        The function handles volumes with different number of B-scans
        by treating them as separate samples in the batch dimension.
        Args:
            bscan_tensor: [B, S, H, W]
        Returns:
            patch_tokens: [B, S, P, D]
            lesion_presence_global: [B, 2, D]
        """
        batch_size = bscan_tensor.shape[0]
        num_slices = bscan_tensor.shape[2]
        bscan = rearrange(bscan_tensor, 'b c s h w -> (b s) c h w')
        with torch.no_grad():
            encoder_tokens = self.vm(bscan) # [B*S, T, D]
            patch_tokens = encoder_tokens[:, :-self.vm.num_global_tokens] # [B*S, P, D]
            global_tokens = encoder_tokens[:, -self.vm.num_global_tokens:] # [B*S, G, D]
            lesion_presence = global_tokens[:, 4:5, :]  # [B*S, 1, D]
        lesion_presence = rearrange(lesion_presence, '(b s) 1 d -> b s d', b=batch_size, s=num_slices)
        lesion_presence_global = self.lesion_aggregator(lesion_presence) # [B, 2, D]
        patch_tokens = rearrange(patch_tokens, '(b s) p d -> b s p d', b=batch_size, s=num_slices)
        resampled_patches = self.resampler(patch_tokens) # [B, Q, D]
        combined_vis = torch.cat([resampled_patches, lesion_presence_global], dim=1)
        return combined_vis


class Vision3DModel(nn.Module, VisionRunner):
    def __init__(self, args: ModelArgs, device="cuda"):
        nn.Module.__init__(self)
        VisionRunner.__init__(self, args, device=device)

    def forward(self, bscan):
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            bscan = bscan.to(self.device).to(torch.bfloat16) # [B, C, S, H, W]
            return self.get_vm_features(bscan)

    def load_bridge(self, weights):
        checkpoint = torch.load(weights, map_location=self.device, weights_only=False)
        self.resampler.load_state_dict(checkpoint['resampler'])
        self.lesion_aggregator.load_state_dict(checkpoint['lesion_aggregator'])
        print("Bridge weights loaded successfully!")


class MIRAGEFM3DModel(nn.Module):
    def __init__(self, args, model_args, bridge_weights, pool=True):
        super().__init__()
        self.args = args
        self.model = Vision3DModel(model_args, args.device)
        self.model.load_bridge(bridge_weights)
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.pool = pool

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x).to(torch.float32)
            features = self.norm(features)
            if self.pool:
                num_q = self.model.args.num_queries
                r_features = features[:, :num_q, :].mean(dim=1)
                l_features = features[:, num_q:, :].mean(dim=1)
                features = torch.cat([r_features, l_features], dim=1)
            return features



class MIRAGEFM3D:
    def __init__(self, args):
        if args.input_size is None:
            args.input_size = 512
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
        self.model.eval()
        self.args = args

    def to_tensor(self, x):
        """This expects a numpy array of the form [NUM_SLICES, H, W]."""
        x = torch.from_numpy(x).float()
        if x.max() > 255:
            x = x / 65535.0
        elif x.max() > 1.0:
            x = x / 255.0
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        slices = min(x.shape[2], MAX_SLICES)
        if x.shape[2:] != torch.Size([slices, self.args.input_size, self.args.input_size]):
            x = F.interpolate(
                x,
                size=(slices, self.args.input_size, self.args.input_size),
                mode='trilinear',
                align_corners=False,
            )
        return x


def get_lesion_attention_map(model_vm, bscan_single, lesion_token_idx=4, layer_idx=-1, grid_size=(16, 16)):
    """
    Runs the B-scan encoder on a single B-scan, capturing the [Lesions]
    token's attention to patch tokens from a specific encoder layer.

    Args:
        model_vm: the MIRAGEPredLesionPresenceOnly instance (self.vm)
        bscan_single: [1, 1, H, W] tensor, a single B-scan
        lesion_token_idx: index of the [Lesions] token WITHIN the global
            tokens block (e.g. 4, matching PredictionHead(..., index=4))
        layer_idx: which encoder block to extract attention from
        grid_size: (N_H, N_W) patch grid size, for reshaping the map
    Returns:
        attn_map: [N_H, N_W] numpy array, attention from [Lesions]
            token to each patch, averaged over heads
    """
    target_block = model_vm.encoder[layer_idx]
    target_block.attn.return_attn = True

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _ = model_vm(bscan_single)

    attn = target_block.attn.last_attn.float()  # cast back to float32 for downstream numpy ops
    target_block.attn.return_attn = False

    num_patches = grid_size[0] * grid_size[1]
    global_lesion_idx = num_patches + lesion_token_idx

    lesion_to_patch = attn[0, :, global_lesion_idx, :num_patches]
    lesion_to_patch = lesion_to_patch.mean(dim=0)

    attn_map = lesion_to_patch.reshape(grid_size).cpu().numpy()
    return attn_map


def plot_attention_overlay(bscan_img, attn_map, save_path=None):
    H, W = bscan_img.shape
    attn_upsampled = F.interpolate(
        torch.tensor(attn_map)[None, None], size=(H, W), mode='bilinear', align_corners=False
    )[0, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.02)  # tight gap between the two panels

    axes[0].imshow(bscan_img, cmap='gray')
    axes[0].set_title('B-scan')
    axes[0].axis('off')

    axes[1].imshow(bscan_img, cmap='gray')
    axes[1].imshow(attn_upsampled, cmap='jet', alpha=0.5)
    axes[1].set_title('[Lesions] token attention')
    axes[1].axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f'Saved to {save_path}')
    plt.close()


if __name__ == "__main__":
    args = Namespace(
        input_size=512,
        patch_size=32,
        num_classes=2,
        device='cuda',
    )
    model_class = MIRAGEFM3D(args)
    model = model_class.model.to(args.device)
    model_vm = model.model.vm

    dataset_root = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/eval/_datasets/Classification_3D/OCTAVE')
    out_dir = Path('visualization/__plots_feat/explainability')
    out_dir.mkdir(parents=True, exist_ok=True)

    for subset in ['train', 'val', 'test']:
        subset_path = dataset_root / subset
        if not subset_path.exists():
            continue
        for class_dir in sorted(subset_path.iterdir()):
            if not class_dir.is_dir():
                continue
            for file in sorted(class_dir.iterdir()):
                if file.suffix != '.npz':
                    continue

                volume = np.load(file)['vol']
                volume_tensor = model_class.to_tensor(volume)

                mid_slice_idx = volume_tensor.shape[2] // 2
                bscan_single = volume_tensor[:, :, mid_slice_idx, :, :].to(args.device)
                bscan_img = volume_tensor[0, 0, mid_slice_idx].cpu().numpy()

                attn_map = get_lesion_attention_map(
                    model_vm, bscan_single, lesion_token_idx=4, layer_idx=-1, grid_size=(16, 16),
                )

                fn = out_dir / f'{subset}_{class_dir.name}_{file.stem}.pdf'
                plot_attention_overlay(bscan_img, attn_map, save_path=fn)
