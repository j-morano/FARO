from pathlib import Path
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union, Tuple, Dict, Any
from argparse import Namespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat


MAX_SLICES = 256



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
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PredictionHead(nn.Module):
    def __init__(self, dim_tokens, num_classes, index, hidden_dim=None):
        super().__init__()
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.transpose = True

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        x = F.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            dropout_p=self.attn_drop
        )#.transpose(1, 2).reshape(B, N, C)
        if self.transpose:
            x = x.transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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


class FARO2D(nn.Module):
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
        norm_layer: Any = partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()

        self.args = args

        # Initialize input and output adapters
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens)
        self.input_adapters = nn.ModuleDict(input_adapters)

        self.dim_tokens = dim_tokens

        # Additional learnable tokens (registers + lesion tokens)
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

        # Generated masks (all visible)
        task_masks = {
            domain: torch.zeros((B, tensor.shape[1]), device=tensor.device)
            for domain, tensor in input_task_tokens.items()
        }

        ## Using generated masks
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
        return encoder_tokens


def faro_base(
        input_adapters: Dict[str, Any],
        args,
        **kwargs):
    model = FARO2D(
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
        'bscan': PatchedInputAdapter(
            stride_level=1,
            num_channels=1,
            patch_size_full=tuple(args.patch_size['bscan']),
            image_size=args.input_size['bscan'],
        )
    }

    model = faro_base(
        input_adapters=input_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
        args=args,
    )

    return model


def load_vm(
    device,
    model="faro_base",
    weights="_weights/FARO_base.pth",
) -> nn.Module:
    vm_args = Namespace(
        in_domains=["bscan"],
        out_domains=[],  # NOTE: No output, just features
        standardize_depth=True,
        extra_norm_pix_loss=False,
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


class LesionAggregator(nn.Module):
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
        print("Using lesion aggregator.")

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

        # Cross-Attention: Q searches in the slice tokens (K, V)
        attn_out, _ = self.cross_attn(
            query=self.ln_q(self.queries),
            key=self.ln_kv(x),
            value=self.ln_kv(x)
        )
        x_q = self.queries + attn_out
        x_q = x_q + self.mlp(self.ln_post(x_q))

        return x_q  # [1, 2, d]

class PatchAggregator(nn.Module):
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

        # Block 1: Cross-Attention
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_q1 = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.ln_post1 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

        # Block 2: Self-Attention
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_q2 = nn.LayerNorm(dim)
        self.ln_post2 = nn.LayerNorm(dim)
        print("Using patch aggregator.")

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

        # Block 2 (Self-Attention for queries to interact)
        attn2, _ = self.self_attn(
            query=self.ln_q2(q),
            key=self.ln_q2(q),
            value=self.ln_q2(q),
        )
        q = q + attn2
        q = q + self.mlp(self.ln_post2(q))

        return q


class VisionRunner:
    def __init__(self, num_queries:int=16, device="cuda"):
        super().__init__()
        print(f"Using {MAX_SLICES} slices as max!")
        self.device = device
        self.vm = load_vm(device)
        self.patch_aggregator = PatchAggregator(
            dim=self.vm.dim_tokens, num_queries=num_queries,
        ).to(device).to(torch.bfloat16)
        self.lesion_aggregator = LesionAggregator(
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
        resampled_patches = self.patch_aggregator(patch_tokens) # [B, Q, D]
        combined_vis = torch.cat([resampled_patches, lesion_presence_global], dim=1)
        return combined_vis


class Vision3DModel(nn.Module, VisionRunner):
    def __init__(self, num_queries:int=16, device="cuda"):
        nn.Module.__init__(self)
        VisionRunner.__init__(self, num_queries, device=device)
        self.num_queries = num_queries

    def forward(self, bscan):
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            bscan = bscan.to(self.device).to(torch.bfloat16) # [B, C, S, H, W]
            return self.get_vm_features(bscan)

    def load_bridge(self, weights):
        checkpoint = torch.load(weights, map_location=self.device, weights_only=False)
        self.patch_aggregator.load_state_dict(checkpoint['patch_aggregator'])
        self.lesion_aggregator.load_state_dict(checkpoint['lesion_aggregator'])
        print("Bridge weights loaded successfully!")


class FARO3DModel(nn.Module):
    def __init__(self, args, num_queries, bridge_weights, pool=True):
        super().__init__()
        self.args = args
        self.model = Vision3DModel(num_queries, args.device)
        self.model.load_bridge(bridge_weights)
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.pool = pool

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x).to(torch.float32)
            features = self.norm(features)
            if self.pool:
                num_q = self.model.num_queries
                r_features = features[:, :num_q, :].mean(dim=1)
                l_features = features[:, num_q:, :].mean(dim=1)
                features = torch.cat([r_features, l_features], dim=1)
            return features


@dataclass
class ModelArgs:
    input_size: int = 512
    patch_size: int = 32
    num_classes: int = 2
    device: str = 'cuda'


class FARO(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.input_size is None:
            args.input_size = 512
        self.model = FARO3DModel(
            args,
            16,
            '_weights/VFE_r.pth'
        )
        self.model.eval()
        self.model.to(args.device)
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

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.to_tensor(x)
        return self.model(x)



if __name__ == "__main__":
    from tqdm import tqdm
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    args = ModelArgs()
    model = FARO(args)
    path = Path('./_datasets/Classification_3D/OCTAVE/test')
    features = torch.Tensor()
    targets = []
    for idx, class_name in enumerate(sorted(path.iterdir())):
        print(f"Processing class: {class_name.name}")
        for file in tqdm(sorted(class_name.iterdir())):
            volume = np.load(file)['vol']
            out = model(volume)
            features = torch.cat((features, out.cpu()), dim=0)
            targets.append(idx)
    print("Features shape:", features.shape)
    targets = torch.tensor(targets)
    print("Targets shape:", targets.shape)

    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.numpy())

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=targets.numpy(), cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(set(targets.numpy()))))
    plt.title('t-SNE of FARO Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
