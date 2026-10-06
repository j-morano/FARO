from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from argparse import Namespace, ArgumentParser
import json
import time
import random
import socket
import re
from dataclasses import dataclass

import h5py
import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.utils import save_image
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# from rouge_score import rouge_scorer

from utils.architecture import get_model_architecture




########################################################################
# Constants for approach config


MAX_SLICES = 256



########################################################################
# Helper functions and classes


def get_factory_adder() -> Tuple[Any, Dict[str, Any]]:
    """Get a function that adds a class to a list and the corresponding
    list. Useful for creating a factory with a list of classes. The
    intended use is as a decorator.
    You can also can specify a different name for the class in the list,
    to use it at creation time instead of the class name.
    Example:
        >>> add_class, classes_dict = get_factory_adder()
        >>> @add_class
        ... class A:
        ...     pass
        >>> @add_class('Cc')
        ... class C:
        ...     pass
    """
    classes_dict = {}
    def _add_class(class_: Any, name: Optional[str]=None) -> Any:
        if name is None:
            name = class_.__name__
        classes_dict[name] = class_
        return class_

    def add_class(class_: Any, name: Optional[str]=None) -> Any:
        if not callable(class_):
            name = class_
            def wrapper(class_: Any) -> Any:
                return _add_class(class_, name)
            return wrapper
        else:
            return _add_class(class_)

    return add_class, classes_dict



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
    use_proj: bool = False



########################################################################
# Architecture Components


#-----------------------------------------------------------------------
# Projection

class MedicalProjection(nn.Module):
    def __init__(self, vm_dim, llm_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vm_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    def forward(self, x):
        return self.proj(x)


#-----------------------------------------------------------------------
# Resampler


add_resampler, resampler_factory = get_factory_adder()

# @add_resampler("xattn")
# class VisualResampler(nn.Module):
#     def __init__(
#         self,
#         dim=1024,
#         num_queries=3,
#         num_heads=8,
#         mlp_ratio=4.0
#     ):
#         super().__init__()
#         # 1. The Bottleneck: These 'queries' represent the output tokens
#         # 8 queries means the LLM only ever sees 8 tokens per image
#         self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
#         print(num_queries, "learnable query tokens initialized for resampling.")
#
#         # 2. The Bridge: Cross-Attention
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=dim,
#             num_heads=num_heads,
#             batch_first=True
#         )
#
#         # 3. The Refiner: Post-Attention MLP
#         self.ln_q = nn.LayerNorm(dim)
#         self.ln_kv = nn.LayerNorm(dim)
#
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(dim * mlp_ratio), dim)
#         )
#         self.ln_post = nn.LayerNorm(dim)
#
#     def forward(self, x, return_attn=False):
#         """
#         Args:
#             x: Visual features from MIRAGEv2 [s, hw, dim]
#         Returns:
#             Compressed summary tokens [s, num_queries, dim]
#         """
#         x = rearrange(x, 'b hw d -> 1 (b hw) d')
#
#         b = x.shape[0]
#         q = self.queries.expand(b, -1, -1)
#
#         # Cross-Attention: Q hunts in the VM tokens (K, V)
#         # We apply LayerNorm before attention (Pre-Norm architecture)
#         attn_out, attn_weights = self.cross_attn(
#             query=self.ln_q(q),
#             key=self.ln_kv(x),
#             value=self.ln_kv(x)
#         )
#
#         # Residual + FeedForward
#         x_q = q + attn_out
#         x_q = x_q + self.mlp(self.ln_post(x_q))
#
#         if return_attn:
#             return x_q, attn_weights
#
#         return x_q
#
#
# @add_resampler("xlstm")
# class VisualResamplerXLSTM(nn.Module):
#     def __init__(
#         self,
#         dim=768,
#         num_queries=8,
#         num_heads=8,
#         num_blocks=2,
#         mlp_ratio=4.0,
#         max_context_length=13000,
#     ):
#         super().__init__()
#         self.num_queries = num_queries
#
#         # Learned query tokens — placed at the end of the sequence
#         self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
#
#         cfg_str = f"""
#             mlstm_block:
#                 mlstm:
#                     conv1d_kernel_size: 4
#                     qkv_proj_blocksize: 4
#                     num_heads: {num_heads}
#             context_length: {max_context_length}
#             num_blocks: {num_blocks}
#             embedding_dim: {dim}
#             slstm_at: []
#         """
#         cfg = OmegaConf.create(cfg_str)
#         cfg = from_dict(
#             data_class=xLSTMBlockStackConfig,
#             data=OmegaConf.to_container(cfg),  # type: ignore
#             config=DaciteConfig(strict=True)
#         )
#         self.xlstm = xLSTMBlockStack(cfg)
#
#         print(f"VisualResamplerXLSTM initialized with {num_queries} query tokens.")
#
#     def forward(self, x):
#         """
#         Args:
#             x: patch tokens [S, HW, dim]
#         Returns:
#             query outputs [1, num_queries, dim]
#         """
#         x = rearrange(x, 's hw d -> 1 (s hw) d')
#         b = x.shape[0]
#         q = self.queries.expand(b, -1, -1)
#
#         # Queries go at the end — they see the full patch sequence
#         # via the recurrent state before being computed
#         seq = torch.cat([x, q], dim=1)  # [1, S*HW + num_queries, dim]
#
#         out = self.xlstm(seq)  # [1, S*HW + num_queries, dim]
#
#         # Return only the query outputs
#         return out[:, -self.num_queries:, :]  # [1, num_queries, dim]
#
#
# @add_resampler("axxlstm")
# class VisualResamplerAxialXLSTM(nn.Module):
#     def __init__(
#         self,
#         dim=768,
#         num_queries=8,
#         num_heads=8,
#         num_blocks=2,
#         mlp_ratio=4.0,
#         max_context_length=128,
#         grid_size=16,
#         architecture='mlstm',
#     ):
#         super().__init__()
#         self.num_queries = num_queries
#         num_patches = grid_size * grid_size
#
#         self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
#
#         self.xattn = VisualResampler(
#             dim=dim,
#             num_queries=num_queries,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#         )
#         if architecture == 'hybrid':
#             # Assuming num_blocks = 2
#             # We put an sLSTM at index 1 (the second block)
#             cfg_str = f"""
#                 mlstm_block:
#                     mlstm:
#                         conv1d_kernel_size: 3
#                         qkv_proj_blocksize: 4
#                         num_heads: {num_heads}
#                 slstm_block:
#                     slstm:
#                         backend: "vanilla"
#                         num_heads: {num_heads}
#                 context_length: {max_context_length}
#                 num_blocks: {num_blocks}
#                 embedding_dim: {dim}
#                 slstm_at: [1]
#             """
#         elif architecture == 'mlstm':
#             cfg_str = f"""
#                 mlstm_block:
#                     mlstm:
#                         conv1d_kernel_size: 4
#                         qkv_proj_blocksize: 4
#                         num_heads: {num_heads}
#                 context_length: {max_context_length}
#                 num_blocks: {num_blocks}
#                 embedding_dim: {dim}
#                 slstm_at: []
#             """
#         cfg = OmegaConf.create(cfg_str)
#         cfg = from_dict(
#             data_class=xLSTMBlockStackConfig,
#             data=OmegaConf.to_container(cfg),  # type: ignore
#             config=DaciteConfig(strict=True)
#         )
#         self.xlstm = xLSTMBlockStack(cfg)
#
#         print(f"VisualResamplerXLSTM initialized with {num_queries} query tokens.")
#
#     def forward(self, x):
#         """
#         Args:
#             x: patch tokens [s, hw, dim]
#         Returns:
#             query outputs [1, num_queries, dim]
#         """
#         x = rearrange(x, 'b hw d -> hw b d')
#
#         out = self.xlstm(x)
#
#         out = out[:, -1:, :]
#
#         out = rearrange(out, 'hw b d -> b hw d')
#
#         out = out + self.pos_embed
#
#         out = self.xattn(out)
#
#         # Return only the query outputs
#         return out


@add_resampler("axbidmlstm")
class VisualResamplerAxialBidMLSTM(nn.Module):
    def __init__(
        self,
        dim=768,
        num_queries=8,
        num_heads=8,
        num_blocks=2,
        mlp_ratio=4.0,
        max_context_length=MAX_SLICES,
        grid_size=16,
    ):
        super().__init__()
        self.num_queries = num_queries
        num_patches = grid_size * grid_size

        self.pos_embed = nn.Parameter(torch.randn(1, 1, num_patches, dim) * 0.02)
        self.dir_embed = nn.Parameter(torch.randn(1, 2, 1, dim) * 0.02)

        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        print(num_queries, "learnable query tokens initialized for resampling.")

        cfg_str = f"""
            mlstm_block:
                mlstm:
                    conv1d_kernel_size: 3
                    qkv_proj_blocksize: 4
                    num_heads: {num_heads}
            context_length: {max_context_length}
            num_blocks: 1
            embedding_dim: {dim}
            slstm_at: []
        """
        cfg = OmegaConf.create(cfg_str)
        cfg = from_dict(
            data_class=xLSTMBlockStackConfig,
            data=OmegaConf.to_container(cfg),  # type: ignore
            config=DaciteConfig(strict=True)
        )
        self.xlstm_fwd = xLSTMBlockStack(cfg)
        self.xlstm_bwd = xLSTMBlockStack(cfg)

        cfg_spt_str = f"""
            mlstm_block:
                mlstm:
                    conv1d_kernel_size: 3
                    qkv_proj_blocksize: 4
                    num_heads: {num_heads}
            context_length: 1024
            num_blocks: 1
            embedding_dim: {dim}
            slstm_at: []
        """
        cfg_spt = OmegaConf.create(cfg_spt_str)
        cfg_spt = from_dict(
            data_class=xLSTMBlockStackConfig,
            data=OmegaConf.to_container(cfg_spt),  # type: ignore
            config=DaciteConfig(strict=True)
        )
        self.xlstm_spt = xLSTMBlockStack(cfg_spt)

        print(f"VisualResamplerAxialBidMLSTM initialized with {num_queries} query tokens and bidirectional axial processing with {dim} hidden dimension.")

    def forward(self, x):
        """Batched forward.
        Args:
            x: patch tokens [B, S, P(HxW), D]
        """
        B, S, P, D = x.shape

        # Axial processing (Treating each spatial patch as its own sequence of slices)
        x = rearrange(x, 'b s p d -> (b p) s d')

        # Sequential Bidirectional Residuals (ViL Style)
        out = x + self.xlstm_fwd(x)
        flipped = torch.flip(out, dims=[1])
        out = out + torch.flip(self.xlstm_bwd(flipped), dims=[1])

        # Extract Bilateral context for each patch
        out = torch.cat([out[:, 0:1], out[:, -1:]], dim=1) # [B*P, 2, D]

        # Prepare for Spatial xLSTM (The "Pure xLSTM Resampler" logic)
        out = rearrange(out, '(b p) i d -> b i p d', b=B, p=P)
        out = out + self.pos_embed
        out = out + self.dir_embed
        out = rearrange(out, 'b i p d -> b (p i) d')

        # Append Latent Queries (Acting as information sinks)
        q = self.queries.expand(B, -1, -1)
        out = torch.cat([out, q], dim=1) # [B, P*2 + Q, D]

        # The Spatial xLSTM processes the sequence.
        # Note: No residual here because queries are new 'sink' tokens.
        out = self.xlstm_spt(out)
        out = out[:, -self.num_queries:, :]

        return out


@add_resampler("xattn_v2")
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


@add_resampler("meanpool")
class VisualResamplerMeanPool(nn.Module):
    """A non-learnable resampler that simply averages the patch tokens
    across the slice dimension.
    """
    def __init__(self, dim, num_queries=16, **kwargs):
        super().__init__()
        self.num_queries = num_queries
        assert int(num_queries ** 0.5) ** 2 == num_queries,\
            f"num_queries={num_queries} is not a perfect square"
        self.tgt_grid = int(num_queries ** 0.5)

    def forward(self, x):
        """
        x: [B, S, P(HxW), D]
        """
        # Mean over slice dimension -> [B, P, D]
        x = x.mean(dim=1)
        # Now pool P (256) down to num_queries (16)
        # Reshape to grid and do spatial pooling
        B, P, D = x.shape
        grid = int(P ** 0.5)  # 16
        x = rearrange(x, 'b (h w) d -> b d h w', h=grid, w=grid)
        # Pool 16x16 -> 4x4 to get exactly 16 tokens
        x = F.adaptive_avg_pool2d(x, (self.tgt_grid, self.tgt_grid))
        x = rearrange(x, 'b d h w -> b (h w) d')  # [B, 16, D]
        return x


#-----------------------------------------------------------------------
# Lesion Aggregator


add_aggregator, aggregator_factory = get_factory_adder()


# @add_aggregator("attnpool")
# class LesionAggregatorAttnPool(nn.Module):
#     def __init__(self, dim=768):
#         super().__init__()
#         # 1. The Gate: A small MLP that looks at a token and decides its "importance"
#         self.gate = nn.Sequential(
#             nn.Linear(dim, dim // 2),
#             nn.GELU(),
#             nn.Linear(dim // 2, 1)  # Produces a single scalar score per slice
#         )
#
#         # 2. Optional: A projection to transform the features before weighting
#         self.feat_proj = nn.Linear(dim, dim)
#         print("Using Attention Pooling for lesion aggregation!")
#
#     def forward(self, lesion_presence):
#         """
#         lesion_presence: [1, S, d] from VM global tokens
#         Returns: [1, 1, d] aggregated token
#         """
#         # Squeeze out the middle dimension to get [S, d]
#         x = lesion_presence.squeeze(0)  # Shape: [49, 768]
#
#         # 1. Compute attention scores for each slice
#         scores = self.gate(x)  # Shape: [49, 1]
#
#         # 2. Normalize scores across the slice dimension (S) using Softmax
#         # This ensures the weights sum to 1.0
#         weights = F.softmax(scores, dim=0)  # Shape: [49, 1]
#
#         # 3. Weighted Sum: Multiply each token by its learned weight
#         # We transform the features slightly while we are at it
#         x_projections = self.feat_proj(x)  # Shape: [49, 768]
#
#         # Summing across the S dimension
#         aggregated = torch.sum(x_projections * weights, dim=0, keepdim=True) # [1, 768]
#
#         # Add a batch dimension for the LLM [1, 1, 768]
#         return aggregated.unsqueeze(0)


# @add_aggregator("xlstm")
# class LesionAggregator(nn.Module):
#     def __init__(self, dim=768, num_heads=4, max_slices=MAX_BSCANS):
#         super().__init__()
#         cfg_str = f"""
#             mlstm_block:
#                 mlstm:
#                     conv1d_kernel_size: 4
#                     qkv_proj_blocksize: 4
#                     num_heads: {num_heads}
#             context_length: {max_slices}
#             num_blocks: 1
#             embedding_dim: {dim}
#             slstm_at: []
#         """
#         cfg = OmegaConf.create(cfg_str)
#         cfg = from_dict(
#             data_class=xLSTMBlockStackConfig,
#             data=OmegaConf.to_container(cfg),  # type: ignore
#             config=DaciteConfig(strict=True)
#         )
#         self.xlstm = xLSTMBlockStack(cfg)
#         print("Using xLSTM for lesion aggregation!")
#
#     def forward(self, x):
#         """
#         lesion_presence: [1, S, d]
#         returns: [1, 1, d]
#         """
#         out = self.xlstm(x)  # [1, S, d]
#         return out[:, -1:, :]  # [1, 1, d] — last token sees full sequence


@add_aggregator("bidmlstm")
class LesionAggregatorBidMLSTM(nn.Module):
    def __init__(self, dim=768, num_heads=4, max_slices=MAX_SLICES):
        super().__init__()
        cfg_str = f"""
            mlstm_block:
                mlstm:
                    conv1d_kernel_size: 3
                    qkv_proj_blocksize: 4
                    num_heads: {num_heads}
            context_length: {max_slices}
            num_blocks: 1
            embedding_dim: {dim}
            slstm_at: []
        """
        cfg = OmegaConf.create(cfg_str)
        cfg = from_dict(
            data_class=xLSTMBlockStackConfig,
            data=OmegaConf.to_container(cfg),  # type: ignore
            config=DaciteConfig(strict=True)
        )
        self.xlstm_fwd = xLSTMBlockStack(cfg)
        self.xlstm_bwd = xLSTMBlockStack(cfg)
        print(f"Using bidirectional xLSTM for lesion aggregation with {dim}!")

    def forward(self, x):
        """
        x: [B, S, D]
        """
        # 1. Forward Pass with Residual (ViL Block 1)
        # x = x + mLSTM_fwd(LN(x))
        x = x + self.xlstm_fwd(x)

        # 2. Backward Pass with Residual (ViL Block 2)
        # We flip, process, then flip back to add the residual correctly
        flipped = torch.flip(x, dims=[1])
        x = x + torch.flip(self.xlstm_bwd(flipped), dims=[1])

        # 3. Bilateral Aggregation (Paper Section 4.1e)
        # Concatenate the 'anchor' tokens (first and last)
        summary = torch.cat([x[:, 0:1], x[:, -1:]], dim=1) # [1, 2, d]

        return summary


@add_aggregator("xattn_v2")
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


@add_aggregator("max_avg")
class LesionAggregatorMaxAvgPool(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        print("Using simple Max(c)Avg Pooling for lesion aggregation!")

    def forward(self, lesion_presence):
        # Simple Max Pooling across the slice dimension
        max_token = lesion_presence.max(dim=1, keepdim=True)[0]  # [1, 1, d]
        avg_token = lesion_presence.mean(dim=1, keepdim=True)  # [1, 1, d]
        return torch.cat([max_token, avg_token], dim=1)  # [1, 2, d]



########################################################################
# Model Loader Functions

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


def load_llm(device):
    llm_id = "Qwen/Qwen3.5-0.8B-Base"

    tokenizer = AutoTokenizer.from_pretrained(llm_id)

    # NOTE: Do this so that the pad token is not the same as eos token
    #   and is not ignored during loss calculation.
    pad_token = "<|im_end|>"
    vision_start_token = "<|vision_start|>"
    vision_end_token = "<|vision_end|>"
    assert tokenizer.convert_tokens_to_ids(vision_start_token) != -1
    assert tokenizer.convert_tokens_to_ids(vision_end_token) != -1
    assert tokenizer.convert_tokens_to_ids(pad_token) != -1
    assert tokenizer.convert_tokens_to_ids(pad_token) != tokenizer.eos_token_id
    tokenizer.pad_token = pad_token

    llm = AutoModelForCausalLM.from_pretrained(
        llm_id,
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map={"": device},
        tie_word_embeddings=True
    )

    return llm, tokenizer



########################################################################
# Training


class VisionRunner:
    def __init__(self, args: ModelArgs, device="cuda"):
        super().__init__()
        print(f"Using {MAX_SLICES} slices as max!")
        self.args = args
        self.device = device
        self.vm = load_vm(device)
        self.resampler = resampler_factory[args.resampler](
            dim=self.vm.dim_tokens, num_queries=args.num_queries,
        ).to(device).to(torch.bfloat16)
        self.lesion_aggregator = aggregator_factory[args.lesion_pool](
            dim=self.vm.dim_tokens,
        ).to(device).to(torch.bfloat16)
        if args.use_proj:
            self.projection = MedicalProjection(
                self.vm.dim_tokens,
                1024,
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
            encoder_tokens, _, _ = self.vm({'bscan': bscan}, mask_inputs=False, force_no_shuffle=True) # [B*S, T, D]
            patch_tokens = encoder_tokens[:, :-self.vm.num_global_tokens] # [B*S, P, D]
            global_tokens = encoder_tokens[:, -self.vm.num_global_tokens:] # [B*S, G, D]
            lesion_presence = global_tokens[:, 4:5, :]  # [B*S, 1, D]
        lesion_presence = rearrange(lesion_presence, '(b s) 1 d -> b s d', b=batch_size, s=num_slices)
        lesion_presence_global = self.lesion_aggregator(lesion_presence) # [B, 2, D]
        patch_tokens = rearrange(patch_tokens, '(b s) p d -> b s p d', b=batch_size, s=num_slices)
        resampled_patches = self.resampler(patch_tokens) # [B, Q, D]
        combined_vis = torch.cat([resampled_patches, lesion_presence_global], dim=1)
        if self.args.use_proj:
            combined_vis = self.projection(combined_vis)
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
        if self.args.use_proj:
            self.projection.load_state_dict(checkpoint['projection'])
        print("Bridge weights loaded successfully!")


class MedicalRunner(VisionRunner):
    def __init__(self, args: ModelArgs, device="cuda"):
        super().__init__(args, device=device)
        self.llm, self.tokenizer = load_llm(device)
        print("Dim tokens from VM:", self.vm.dim_tokens)
        self.projection = MedicalProjection(
            self.vm.dim_tokens,
            self.llm.config.hidden_size,
        ).to(device).to(torch.bfloat16)

        self.set_modes()

    def load_weights(self, weights, device):
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
        self.resampler.load_state_dict(checkpoint['resampler'])
        self.lesion_aggregator.load_state_dict(checkpoint['lesion_aggregator'])
        self.projection.load_state_dict(checkpoint['projection'])

    def precompute_embeddings(self):
        print("Precomputing some fixed embeddings for efficiency!")
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            # prompt = "Befundbericht Augenheilkunde: "
            prompt = "Augen-OCT Befundbericht: "
            prompt_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).input_ids.to(self.device)
            vision_start_token = "<|vision_start|>"
            vision_start_id = self.tokenizer.convert_tokens_to_ids(vision_start_token)
            vision_end_token = "<|vision_end|>"
            vision_end_id = self.tokenizer.convert_tokens_to_ids(vision_end_token)
            self.prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
            self.vision_start_embed = self.llm.get_input_embeddings()(torch.tensor([[vision_start_id]], device=self.device))
            self.vision_end_embed = self.llm.get_input_embeddings()(torch.tensor([[vision_end_id]], device=self.device))

    def set_modes(self):
        """Set modes (train/eval) of the different modules."""
        pass


def calculate_token_weights(reports, tokenizer, llm, device):
    # Get the actual size the LLM expects (e.g., 248320)
    vocab_size = llm.config.vocab_size
    print(f"Calculating weights for {vocab_size} tokens...")

    # Initialize weights for the FULL vocab size
    counts = torch.ones(vocab_size, device=device)

    for _, info in reports:
        report_text = info['oct_findings']
        ids = tokenizer.encode(report_text, add_special_tokens=False)
        for t in ids:
            # Add a safety check for out-of-bounds tokens
            if t < vocab_size:
                counts[t] += 1

    # Log-Inverse weighting
    weights = 1.0 / torch.log1p(counts)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=0.1, max=5.0)

    # Padding and EOS logic
    weights[tokenizer.pad_token_id] = 0.0
    if tokenizer.eos_token_id is not None:
        weights[tokenizer.eos_token_id] = 1.0

    # Check weights for common vs rare tokens
    trocken_id = tokenizer.encode("Trocken", add_special_tokens=False)[0]
    parafoveale_id = tokenizer.encode("parafoveale", add_special_tokens=False)[0]

    print(f"Weight for 'Trocken': {weights[trocken_id].item():.4f}")
    print(f"Weight for 'parafoveale': {weights[parafoveale_id].item():.4f}")

    return weights.to(torch.bfloat16)


class MedicalTrainer(MedicalRunner):
    def __init__(
        self,
        args: ModelArgs,
        dataloader: DataLoader,
        device="cuda",
        lr=2e-4,
        num_epochs=10,
        accumulation_steps=1,
        max_grad_norm=1.0,
        save_dir=Path(),
        reports=None,
        frequency_weighted=False,
    ):
        super().__init__(args, device=device)
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.dataloader = dataloader
        assert self.dataloader is not None

        self.trainable_params = list(self.resampler.parameters()) \
            + list(self.lesion_aggregator.parameters()) \
            + list(self.projection.parameters())

        self.initialize_weights()

        self.optimizer = optim.AdamW(
            self.trainable_params,
            lr=lr,
            weight_decay=0.01
        )
        # Print number of trainable parameters
        total_params = sum(p.numel() for p in self.trainable_params)
        print(f"Total trainable parameters: {total_params/1e6:.2f}M")
        for p in self.vm.parameters():
            assert not p.requires_grad, "VM parameters should be frozen!"
        for p in self.llm.parameters():
            assert not p.requires_grad, "LLM parameters should be frozen!"
        for p in self.resampler.parameters():
            assert p.requires_grad, "Resampler parameters should be trainable!"
        for p in self.lesion_aggregator.parameters():
            assert p.requires_grad, "Lesion aggregator parameters should be trainable!"
        for p in self.projection.parameters():
            assert p.requires_grad, "Projection parameters should be trainable!"
        print("VERIFIED parameter freezing and training status!")

        num_updates = int(np.ceil(len(self.dataloader) / self.accumulation_steps) * self.num_epochs)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_updates,
        )

        if frequency_weighted:
            self.token_weights = calculate_token_weights(
                reports,
                self.tokenizer,
                self.llm,
                self.device,
            )
            self.criterion = nn.CrossEntropyLoss(
                weight=self.token_weights,
                ignore_index=self.tokenizer.pad_token_id,
            )
        else:
            # Ensure padding is ignored
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        self.precompute_embeddings()

    def initialize_weights(self):
        # ADD THIS: Standard Transformer initialization
        for m in self.trainable_params:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def set_modes(self):
        print("Setting training modes!")
        self.vm.eval()  # Frozen
        for param in self.vm.parameters():
            param.requires_grad = False
        self.llm.eval()  # Frozen
        for param in self.llm.parameters():
            param.requires_grad = False
        # Trainable
        self.resampler.train()
        self.lesion_aggregator.train()
        self.projection.train()

    def train_step(self, bscan_tensor, info, accumulate=False):
        """
        bscan_tensor: [B, C, S, H, W]
        """
        start_time = time.time()
        # 1. Forward pass in mixed precision
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            # --- 1: Trainable Bridge ---
            combined_vis = self.get_vm_features(bscan_tensor) # [B, Q+2, D]

            # --- 2: Trainable Projection ---
            visual_embeds = self.projection(combined_vis) # [B, Q+2, D_llm]

            # --- 3: LLM Integration ---
            label_text = info['oct_findings'][0]
            label_ids = self.tokenizer(
                label_text + self.tokenizer.eos_token,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).input_ids.to(self.device)
            label_embeds = self.llm.get_input_embeddings()(label_ids)

            inputs_embeds = torch.cat([
                self.vision_start_embed,
                visual_embeds,
                self.vision_end_embed,
                self.prompt_embeds,
                label_embeds,
            ], dim=1)

            # --- 4: Forward & Loss ---
            outputs = self.llm(inputs_embeds=inputs_embeds)
            logits = outputs.logits

            # Start predicting labels exactly where they begin in the sequence
            # Index is: length of prompt + length of visual tokens - 1
            start_idx = (
                self.vision_start_embed.shape[1]
                + visual_embeds.shape[1]
                + self.vision_end_embed.shape[1]
                + self.prompt_embeds.shape[1]
                - 1
            )

            shift_logits = logits[:, start_idx:-1, :].contiguous()
            shift_labels = label_ids.contiguous()

            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
            )

        # 2. Backward pass outside autocast for numerical stability
        loss = loss / self.accumulation_steps
        loss.backward()
        grad_norm = None
        if not accumulate:
            # Optional: Gradient clipping is very helpful for VLMs
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_params,
                max_norm=self.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return loss.item() * self.accumulation_steps, time.time() - start_time, grad_norm

    def train(self):
        window = self.accumulation_steps

        for epoch in range(self.num_epochs):
            monitor = {
                "loss": [],
                "data_time": [],
                "compute_time": [],
            }
            # Assuming 'train_loader' gives (bscan_tensor, label_string)

            prev_time = time.time()
            for step, (bscan, info) in enumerate(self.dataloader):
                monitor["data_time"].append(time.time() - prev_time)
                # Transfer to device
                bscan = bscan.to(self.device).to(torch.bfloat16)
                # Bscan shape: [Batch, Channels, Slices, Height, Width]

                if epoch == 0 and step == 0:  # Debug
                    # Info example: {'patient_id': tensor([42]), 'oct_findings': ['Trocken, Fibrose.'], 'visual_acuities': tensor([0.5229], dtype=torch.float64), 'age': tensor([41]), 'sex': ['Female'], 'diagnoses': ['k.A.'], 'clinical_term': ['Rechtes Auge: nach wie vor subretinale. Flüssigkeit (gleichbleibend) - vorerst Obseravnz; linkes Auge: Stabiles Narbenstadium.']}
                    print("B-scan shape (original):", bscan.shape)
                    print("Info:", info)
                if step == 0:
                    debug_path = self.save_dir / "__debug"
                    debug_path.mkdir(exist_ok=True)
                    rand_bscan_idx = random.randint(0, bscan.shape[2]-1)
                    save_image(
                        bscan[:, 0, rand_bscan_idx:rand_bscan_idx+1],
                        debug_path / f"{epoch+1:03d}_bscan.png"
                    )

                # Run step
                accumulate = (
                    ((step + 1) % self.accumulation_steps != 0)
                    and (step + 1 != len(self.dataloader))
                )
                loss, compute_time, grad_norm = self.train_step(bscan, info, accumulate)
                monitor["compute_time"].append(compute_time)
                monitor["loss"].append(loss)

                # Periodic verbose print
                if not accumulate:
                    avg_loss = np.mean(monitor["loss"][-window:])
                    avg_data_time = np.mean(monitor["data_time"][-window:])
                    avg_compute_time = np.mean(monitor["compute_time"][-window:])
                    print(
                        f"[Epoch {epoch+1}/{self.num_epochs}, step {step+1}/{len(self.dataloader)}]"
                        f" Avg Loss: {avg_loss:.4f}"
                        f" | Grad Norm: {grad_norm:.4f}"
                        f" | Avg Data+Compute Time (s): {avg_data_time:.4f}+{avg_compute_time:.4f}",
                        flush=True
                    )

                prev_time = time.time()

            avg_loss = np.mean(monitor["loss"])
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'resampler': self.resampler.state_dict(),
                'lesion_aggregator': self.lesion_aggregator.state_dict(),
                'projection': self.projection.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'loss': avg_loss,
            }, self.save_dir / f'{epoch+1:03d}.pt')

            with open(self.save_dir / f'{epoch+1:03d}_monitor.json', 'w') as f:
                epoch_monitor = {
                    'loss': np.mean(monitor["loss"]),
                    'data_time': np.mean(monitor["data_time"]),
                    'compute_time': np.mean(monitor["compute_time"]),
                }
                json.dump(epoch_monitor, f)


class NoAugment:
    def __init__(self):
        print("Using NoAugment (no data augmentation).")

    def __call__(self, volume):
        # NOTE: volume: [C, H, W] (C is slices), [0, 1] range
        return volume


class LightOCTAugment:
    def __init__(self, p=0.5, drop_prob=0.2):
        print("Using LightOCTAugment with p =", p)
        self.p = p
        self.drop_prob = drop_prob

    def __call__(self, volume):
        # NOTE: volume: [C, H, W] (C is slices), [0, 1] range
        if random.random() < self.drop_prob and volume.shape[0] > 7:
            # Randomly drop some slices to simulate missing data,
            #   keeping at least 7.
            num_tgt_slices = random.randint(7, volume.shape[0])
            # Sample slices from current volume evenly to maintain
            #   coverage.
            indices = torch.linspace(0, volume.shape[0]-1, steps=num_tgt_slices).round().long()
            volume = volume[indices]

        if random.random() < self.p:
            # Subtle speckle noise
            noise = torch.randn_like(volume) * 0.02
            volume = volume + (volume * noise)

        if random.random() < self.p:
            # Rotation (+/- 5 degrees)
            # Keeps anatomy mostly vertical but breaks pixel-perfect memorization
            angle = random.uniform(-5, 5)
            volume = TF.rotate(
                volume,
                angle,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=[0],
            )

        if random.random() < self.p:
            # Resize Crop (95% - 100%)
            # We use the class method to get static params for the whole volume
            img_size = volume.shape[-1]

            # get_params returns (i, j, h, w) for the crop
            i, j, h, w = T.RandomResizedCrop.get_params(
                volume,
                scale=[0.95, 1.0],
                ratio=[1.0, 1.0], # Keep it 1:1 square
            )

            # Apply the same crop and resize to all slices in the volume
            volume = TF.resized_crop(volume, i, j, h, w, [img_size, img_size])

        if random.random() < self.p:
            # Shift brightness and contrast
            # brightness_factor < 1 makes it darker, > 1 brighter
            bf = random.uniform(0.9, 1.1)
            volume = volume * bf

        if random.random() < self.p:
            # Gamma adjustment to simulate different acquisition settings
            gamma = random.uniform(0.9, 1.1)
            volume = torch.pow(volume + 1e-6, gamma)

        volume = torch.clamp(volume, 0, 1)

        return volume


class DEPRECATEDVisionLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, reports, data_root, transform=None):
        # Load your dataset from 'data_dir'
        # Each item should be a tuple: (bscan_tensor, label_string)
        self.data_root = Path(data_root)
        self.samples = reports
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fsid, info = self.samples[index]
        part = fsid.split('-')[2]
        subdir_1 = part[:2]
        subdir_2 = part[2:3]
        c_src_dir = self.data_root / subdir_1 / subdir_2 / fsid
        bscan_fns = sorted(c_src_dir.glob('*.png'))
        volume = np.zeros((len(bscan_fns), 512, 512), dtype=np.uint8)
        for i, bscan_fn in enumerate(bscan_fns):
            # Load the image using PIL but keep only the first channel
            #   of the RGB image. Not convert, just discard other
            #   channels.
            bscan = np.array(Image.open(bscan_fn).getchannel(0))
            volume[i] = bscan
        assert volume.shape[0] > 0
        # volume = np.stack(volume, axis=0, dtype=np.uint8)
        volume = torch.from_numpy(volume).float()

        if volume.max() > 1.0:
            volume = volume / 255.0  # Normalize to [0, 1]

        if self.transform:
            volume = self.transform(volume)

        return volume.unsqueeze(0), info


class NewVisionLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, reports, data_root, transform=None):
        # Load your dataset from 'data_dir'
        # Each item should be a tuple: (bscan_tensor, label_string)
        self.data_root = Path(data_root)
        self.samples = reports
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def split_to_slices(self, data):
        num_slices = data.shape[0]
        tgt_num_slices = min(num_slices, MAX_SLICES)
        selected_indices = np.linspace(0, num_slices - 1, tgt_num_slices, dtype=int)
        return selected_indices

    def __getitem__(self, index):
        fsid, info = self.samples[index]
        part = fsid.split('-')[2]
        subdir_1 = part[:2]
        subdir_2 = part[2:3]
        fn = self.data_root / subdir_1 / subdir_2 / f"{fsid}.h5"
        with h5py.File(fn, 'r') as f:
            volume = f['vol'][()]
        volume = torch.from_numpy(volume).float()

        if volume.max() > 1.0:
            volume = volume / 255.0  # Normalize to [0, 1]

        if volume.shape[0] > MAX_SLICES:
            volume = volume[self.split_to_slices(volume)]

        if self.transform:
            volume = self.transform(volume)

        return volume.unsqueeze(0), info


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def test_loader(remaining_args):
    print("Testing dataset and dataloader...")
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str)
    args = parser.parse_args(remaining_args)

    reports_fn = "./cfgs/reports_clean_split.json"
    with open(reports_fn, "r") as f:
        reports = json.load(f)["train"]
    report_list = [(fsid, meta) for fsid, meta in reports.items()]

    transform = LightOCTAugment(p=0.5, drop_prob=1.0)
    dataset = NewVisionLanguageDataset(report_list, args.data_root, transform=transform)
    dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=0)

    for bscan, info in dataloader:
        print("B-scan batch shape:", bscan.shape)
        print("Info:", info)


def get_report_list(reports_fn, split, skip_ka=False):
    with open(reports_fn, "r") as f:
        reports = json.load(f)[split]

    if skip_ka:
        # Skip entries that are just "k.a." or empty
        report_list = []
        for fsid, meta in reports.items():
            finding = meta['oct_findings'].strip().lower()
            if finding not in ["k.a.", "ka", "k. a.", ""]:
                report_list.append((fsid, meta))
    else:
        report_list = [(fsid, meta) for fsid, meta in reports.items()]

    return report_list


def main_train(remaining_args):
    print("Starting TRAINING...")
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--version", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--accumulation_steps", type=int, default=32)
    parser.add_argument("--num_queries", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="__posttraining/")
    parser.add_argument("--device", type=str, default="cuda:0")
    # Make frequency weighting default, as well as skipping k.a. entries
    parser.add_argument("--skip_ka", action="store_true")
    parser.add_argument("--no_skip_ka", dest="skip_ka", action="store_false")
    parser.set_defaults(skip_ka=True)
    parser.add_argument("--frequency_weighted", action="store_true")
    parser.add_argument("--no_frequency_weighted", dest="frequency_weighted", action="store_false")
    parser.set_defaults(frequency_weighted=True)
    parser.add_argument("--resampler", type=str, default="xattn")
    parser.add_argument("--lesion_pool", type=str, default="max")
    parser.add_argument("--drop_prob", type=float, default=0.2)
    args = parser.parse_args(remaining_args)

    assert args.batch_size == 1, args.batch_size

    if socket.gethostname() == "hemingway":
        args.data_root = '/mnt/Data/SSHFS/msc_raid_n10/FullVIBES-v2/vlm_volumes/'

    print("-" * 50)
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("-" * 50)
    # Create string with all arg values joint
    args_to_add = [
        'resampler',
        'lesion_pool',
        'num_queries',
        'accumulation_steps',
        'skip_ka',
        'frequency_weighted',
    ]
    # Add abbreviations (first char of each word)
    args_to_add_abbr = []
    for arg in args_to_add:
        parts = arg.split('_')
        abbr = ''.join([p[0] for p in parts])
        args_to_add_abbr.append((arg, abbr))
    args_str = "_".join([f"{abbr}-{getattr(args, arg)}" for arg, abbr in args_to_add_abbr])
    save_dir = Path(args.save_dir) / args.version / args_str

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save args as json in save_dir
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    set_seeds(42)

    device = args.device
    # device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    reports_fn = "./cfgs/reports_clean_split.json"
    report_list = get_report_list(reports_fn, "train", skip_ka=args.skip_ka)

    transform = LightOCTAugment(p=0.5, drop_prob=args.drop_prob)
    dataset = NewVisionLanguageDataset(report_list, args.data_root, transform=transform)
    dataloader = DataLoader(
        dataset,
        args.batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    model_args = ModelArgs(
        num_queries=args.num_queries,
        resampler=args.resampler,
        lesion_pool=args.lesion_pool,
    )

    trainer = MedicalTrainer(
        args=model_args,
        lr=2e-4,
        num_epochs=args.num_epochs,
        accumulation_steps=args.accumulation_steps,
        device=device,
        save_dir=save_dir,
        dataloader=dataloader,
        reports=report_list,
        frequency_weighted=args.frequency_weighted,
    )

    print(f'Training on {len(dataloader)} samples...')

    trainer.train()


def get_data_stats(remaining_args):
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str)
    args = parser.parse_args(remaining_args)

    reports_fn = "./cfgs/reports_clean_split.json"
    report_list = get_report_list(reports_fn, "train", skip_ka=True)
    report_list += get_report_list(reports_fn, "val", skip_ka=True)

    transform = NoAugment()
    dataset = NewVisionLanguageDataset(report_list, args.data_root, transform=transform)
    dataloader = DataLoader(
        dataset,
        1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    slice_counts = []
    for bscan, info in tqdm.tqdm(dataloader):
        c_num_slices = bscan.shape[2]
        slice_counts.append(c_num_slices)  # Number of slices

    slice_counts = np.array(slice_counts)
    print(f"Total samples: {len(slice_counts)}")
    print(f"Total slices: {slice_counts.sum()}")
    print(f"Min slices: {slice_counts.min()}")
    print(f"Max slices: {slice_counts.max()}")
    print(f"Mean slices: {slice_counts.mean():.2f}")
    print(f"Median slices: {np.median(slice_counts)}")
    with open("_slice_counts_vlm.json", "w") as f:
        json.dump({
            'total_samples': len(slice_counts),
            'total_slices': int(slice_counts.sum()),
            'min_slices': int(slice_counts.min()),
            'max_slices': int(slice_counts.max()),
            'mean_slices': float(slice_counts.mean()),
            'median_slices': float(np.median(slice_counts)),
        }, f, indent=4)


########################################################################
# Inference


class MedicalInferencer(MedicalRunner):
    def __init__(
        self,
        args: ModelArgs,
        device="cuda",
        weights=Path(),
    ):
        super().__init__(args, device=device)
        self.load_weights(weights, device)
        print("Loaded weights successfully!")
        self.precompute_embeddings()

    def set_modes(self):
        print("Setting validation modes!")
        # All frozen
        self.vm.eval()
        self.resampler.eval()
        self.lesion_aggregator.eval()
        self.projection.eval()
        self.llm.eval()

    @torch.inference_mode()
    def infer(self, dataloader):
        predictions = []
        for step, (bscan, info) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                bscan = bscan.to(self.device).to(torch.bfloat16)
                combined_vis = self.get_vm_features(bscan)
                visual_embeds = self.projection(combined_vis)
                # DEBUG: randomize visual embeddings to test LLM
                #   integration without relying on VM outputs
                # visual_embeds = torch.randn_like(visual_embeds)

                inputs_embeds = torch.cat([
                    self.vision_start_embed,
                    visual_embeds,
                    self.vision_end_embed,
                    self.prompt_embeds,
                ], dim=1)

                attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

                # Autoregressive generation
                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=128,
                    do_sample=False,       # greedy
                )

            predicted_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.append((predicted_text[0], info['oct_findings'][0]))

            if step % 20 == 0:  # Print every 10 steps
                tqdm.tqdm.write(f"Step {step} | Predicted: <{predicted_text[0]}>")
                tqdm.tqdm.write(f"Step {step} | Ground Truth: <{info['oct_findings'][0]}>")
                tqdm.tqdm.write("-" * 50)

        return predictions


def main_val(remaining_args):
    print("Starting VALIDATION...")
    # Just predict some reports for some volumes in a dir
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--save_dir", type=str, default="__posttraining/val/")
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args(remaining_args)

    set_seeds(42)

    args_fn = Path(args.weights).parent / "args.json"
    model_name = Path(args.weights).parent.stem
    epoch = Path(args.weights).stem
    if args_fn.exists():
        args_to_add = {
            "num_queries", "dim", "resampler", "lesion_pool", "version",
            "skip_ka"
        }
        with open(args_fn, "r") as f:
            saved_args = json.load(f)
        for arg in args_to_add:
            if arg in saved_args:
                setattr(args, arg, saved_args[arg])
                print(f"Adding {arg} with value from training args: {saved_args[arg]}")

    save_dir = Path(args.save_dir) / args.version
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")

    model_args = ModelArgs(
        num_queries=args.num_queries,
        resampler=args.resampler,
        lesion_pool=args.lesion_pool,
    )

    reports_fn = "./cfgs/reports_clean_split.json"
    report_list = get_report_list(reports_fn, "val", skip_ka=args.skip_ka)

    inferencer = MedicalInferencer(
        args=model_args,
        device=device,
        weights=args.weights,
    )

    dataset = NewVisionLanguageDataset(report_list, args.data_root)
    dataloader = DataLoader(
        dataset,
        1,
        shuffle=False,
        num_workers=16,
    )

    print(f'Validating on {len(dataloader)} samples...')

    predictions = inferencer.infer(dataloader=dataloader)

    with open(save_dir / f"{model_name}__{epoch}.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def main_infer(remaining_args):
    print("Starting INFERENCE...")
    # Walks a folder structured as:
    #   DATASET_NAME/
    #       train/
    #           class_name_1/
    #               vol1.npz
    #               ...
    #           class_name_2/
    #               ...
    #       val/
    #           ...
    #       test/
    #           ...
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to DATASET_NAME folder, containing "
                             "train/val/test subfolders, each with "
                             "class_name subfolders of .npz volumes.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="__posttraining/infer/")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Comma-separated list of splits to process.")
    args = parser.parse_args(remaining_args)

    dataset_name = Path(args.data_root).stem

    set_seeds(42)

    args_fn = Path(args.weights).parent / "args.json"
    model_name = Path(args.weights).parent.stem
    epoch = Path(args.weights).stem

    if args_fn.exists():
        args_to_add = {
            "num_queries", "dim", "resampler", "lesion_pool", "version",
            "skip_ka", "use_proj",
        }
        with open(args_fn, "r") as f:
            saved_args = json.load(f)
        for arg in args_to_add:
            if arg in saved_args:
                setattr(args, arg, saved_args[arg])
                print(f"Adding {arg} with value from training args: {saved_args[arg]}")

    save_dir = Path(args.save_dir) / getattr(args, "version", "default")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")

    model_args = ModelArgs(
        num_queries=getattr(args, "num_queries", 16),
        resampler=getattr(args, "resampler", "xattn"),
        lesion_pool=getattr(args, "lesion_pool", "max_avg"),
        use_proj=getattr(args, "use_proj", False),
    )

    inferencer = MedicalInferencer(
        args=model_args,
        device=device,
        weights=args.weights,
    )

    data_root = Path(args.data_root)
    splits = args.splits.split(",")

    # Collect all (npz_path, class_name, split) tuples first
    samples = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"WARNING: split dir {split_dir} does not exist, skipping.")
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.stem
            for npz_fn in sorted(class_dir.glob("*.npz")):
                samples.append((npz_fn, class_name, split))

    print(f"Found {len(samples)} .npz files across splits {splits}")

    results = []
    for npz_fn, class_name, split in tqdm.tqdm(samples):
        volume = np.load(npz_fn)["vol"]
        volume = torch.from_numpy(volume).float()
        if volume.max() > 1.0:
            volume = volume / 255.0  # Normalize to [0, 1]
        if volume.shape[0] > MAX_SLICES:
            num_slices = volume.shape[0]
            indices = np.linspace(0, num_slices - 1, MAX_SLICES, dtype=int)
            volume = volume[indices]
        # [S, H, W] -> [1, 1, S, H, W]  (B, C, S, H, W)
        bscan = volume.unsqueeze(0).unsqueeze(0)
        # Resize to 512 H and W
        bscan = F.interpolate(bscan, size=(bscan.shape[2], 512, 512), mode='trilinear', align_corners=False)

        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            bscan = bscan.to(device).to(torch.bfloat16)
            combined_vis = inferencer.get_vm_features(bscan)
            visual_embeds = inferencer.projection(combined_vis)

            inputs_embeds = torch.cat([
                inferencer.vision_start_embed,
                visual_embeds,
                inferencer.vision_end_embed,
                inferencer.prompt_embeds,
            ], dim=1)
            attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

            generated_ids = inferencer.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                pad_token_id=inferencer.tokenizer.pad_token_id,
                eos_token_id=inferencer.tokenizer.eos_token_id,
                max_new_tokens=128,
                do_sample=False,
            )
        predicted_text = inferencer.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        results.append({
            "id": npz_fn.stem,
            "prediction": predicted_text,
            "class_name": class_name,
            "split": split,
        })

        tqdm.tqdm.write(f"[{split}/{class_name}] {npz_fn.stem} | Predicted: <{predicted_text}>")

    out_fn = save_dir / f"{dataset_name}__{model_name}__{epoch}.json"
    with open(out_fn, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} predictions to {out_fn}")



########################################################################
# Evaluation

def norm_txt(t):
    return t.lower().replace("ü", "ue").replace("ä", "ae").replace("ö", "oe").strip()


NEGATION_PATTERNS = re.compile(
    r'\b(kein|keine|keinen|keinem|keiner|ohne|nicht)\b',
    re.IGNORECASE
)

CLINICAL_GROUPS = {
    "OEDEM": [
        "ödem", "zysten", "zyste", "zystoides", "flüssigkeit", "srf", "irz",
        "cme", "makulaödem", "flüssigkeitsansammlung", "ablagerung",
        "netzhaut verdickt", "intraretinale zysten", "intraretinale zyste",
    ],
    "PEA": [
        "pigmentepithelabhebung", "pea", "ped", "abhebung", "irregularität",
        "unregelmäßigkeit", "irregularitaet", "üed", "pigmentepithel-irregularität",
    ],
    "FIBROSE_NARBE": [
        "fibrose", "fibrotisch", "fibrotische", "narbe", "narbem", "gliose",
        "fibrovask", "fibrovasc", "fibravask", "junius-kuhnt", "exsud. narbe",
    ],
    "ATROPHIE": [
        "atrophie", "atrophisch", "ga", "geographische atrophie", "pre-narbe",
        "retinales pigmentepithel-atrophie", "retinales pigmentepithel atrophie",
    ],
    "MEMBRAN": [
        "epiretinale membran", "erm", "vrt", "traktionssyndrom",
        "vitreomakulaeres", "vitreomakulär",
    ],
    "DRUSEN": [
        "drusen", "drusenoide", "drusem",  # typo in GT
    ],
    "TROCKEN": [
        "trocken", "trockenb",  # typo in GT
        "kein ödem", "ödem resorbiert", "unauffällig", "ohne befund",
        "eher trocken", "soweit beurteilbar trocken",
    ],
}

LOCATIONS = [
    "subretinal", "intraretinal", "parafoveal", "zentral", "parazentral",
    "nasal", "inferior", "foveal", "superior", "temporal",
]
NORM_LOCATIONS = [norm_txt(loc) for loc in LOCATIONS]


def has_negation(segment):
    return bool(NEGATION_PATTERNS.search(segment))


def extract_oct_graph(text):
    if not text or "k.a." in text.lower():
        return set()

    segments = re.split(r'[,.;]|\bund\b', text)
    found_triplets = []

    for segment in segments:
        norm_seg = norm_txt(segment)
        if not norm_seg:
            continue

        # Skip negated segments
        if has_negation(norm_seg):
            continue

        segment_findings = []
        for group, synonyms in CLINICAL_GROUPS.items():
            norm_syns = [norm_txt(s) for s in synonyms]
            if any(re.search(rf"\b{re.escape(syn)}\b", norm_seg) for syn in norm_syns):
                segment_findings.append(group)

        segment_locs = [loc for loc in NORM_LOCATIONS if loc in norm_seg]

        for finding in segment_findings:
            if segment_locs:
                for loc in segment_locs:
                    found_triplets.append((finding, "located_at", loc))
            else:
                found_triplets.append((finding, "exists", "present"))

    return set(found_triplets)


def compute_nlp_metrics(predictions):
    references = []
    hypotheses = []
    for pred_text, true_text in predictions:
        references.append([norm_txt(true_text).split()])
        hypotheses.append(norm_txt(pred_text).split())

    smoother = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_scores = [
        scorer.score(norm_txt(true_text), norm_txt(pred_text))['rougeL'].fmeasure
        for pred_text, true_text in predictions
    ]
    rouge_l = float(np.mean(rouge_scores))

    return {
        "bleu1": bleu1,
        "bleu4": bleu4,
        "rouge_l": rouge_l,
    }


def evaluate_model(remaining_args):
    print("Starting EVALUATION...")
    parser = ArgumentParser()
    parser.add_argument("--predictions", type=str)
    args = parser.parse_args(remaining_args)

    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    all_metrics = []
    both_empty = 0
    one_empty = 0

    # For micro F1
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred_text, true_text in predictions:
        pred_graph = extract_oct_graph(pred_text)
        true_graph = extract_oct_graph(true_text)

        if len(pred_graph) == 0 and len(true_graph) == 0:
            both_empty += 1
            # Exclude from metrics — report separately
            continue
        elif len(pred_graph) == 0 or len(true_graph) == 0:
            one_empty += 1
            precision, recall, f1 = 0.0, 0.0, 0.0
            tp, fp, fn = 0, len(pred_graph), len(true_graph)
        else:
            tp = len(pred_graph.intersection(true_graph))
            fp = len(pred_graph.difference(true_graph))
            fn = len(true_graph.difference(pred_graph))
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        all_metrics.append({
            "predicted": pred_text,
            "true": true_text,
            "pred_graph": str(pred_graph),
            "true_graph": str(true_graph),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    # Macro (per-sample average)
    avg_precision = np.mean([m["precision"] for m in all_metrics])
    avg_recall = np.mean([m["recall"] for m in all_metrics])
    all_f1 = [m["f1"] for m in all_metrics]
    avg_f1 = np.mean(all_f1)
    std_f1 = np.std(all_f1)

    # Micro (aggregate TP/FP/FN)
    micro_precision = total_tp / (total_tp + total_fp + 1e-8)
    micro_recall = total_tp / (total_tp + total_fn + 1e-8)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-8)

    n_total = len(predictions)
    n_evaluated = len(all_metrics)

    nlp_metrics = compute_nlp_metrics(predictions)

    print(f"\nSamples: {n_total} total | {n_evaluated} evaluated | "
          f"{both_empty} both-empty (excluded) | {one_empty} one-empty")
    print(f"\nMacro  — Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f}±{std_f1:.4f}")
    print(f"Micro  — Precision: {micro_precision:.4f} | Recall: {micro_recall:.4f} | F1: {micro_f1:.4f}")
    print(f"\nNLP metrics — BLEU-1: {nlp_metrics['bleu1']:.4f} | "
          f"BLEU-4: {nlp_metrics['bleu4']:.4f} | ROUGE-L: {nlp_metrics['rouge_l']:.4f}")


########################################################################
# Main


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('function', type=str)
    args, extras = parser.parse_known_args()

    globals()[args.function](extras)
