import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from einops import rearrange
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Any
from .patch_embed import PatchEmbed
from .pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed, get_sinusoid_encoding_table
from timm.models.vision_transformer import LayerScale, DropPath
from .ssast import BaseSSAST
import random
import math
from timm.models.layers import trunc_normal_
from collections import namedtuple
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


__all__ = [
    "MambaSSAST", "mamba_ssast_tiny", "mamba_ssast_small", "mamba_ssast_medium", "mamba_ssast_base", "mamba_ssast_large", "mamba_ssast_huge"
]


layernorm_wrapper = partial(nn.LayerNorm, eps=1e-6)


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-6,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    print("using regular mamba")
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MambaSSAST(BaseSSAST):
    def __init__(self,
                 img_size=(80, 200),
                 patch_size=(16, 4),
                 in_chans=1,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 mask_ratio=0.5,
                 masking_mode: str = "unstructured",
                 use_cls_token=True,
                 frequency_first=True,
                 norm_layer=layernorm_wrapper,
                 ssm_cfg=None,
                 rms_norm: bool = False,
                 residual_in_fp32: bool = False,
                 fused_add_norm: bool = False,
                 norm_epsilon: float = 1e-6,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, mask_ratio, masking_mode, use_cls_token, frequency_first, norm_layer)
        self.factory_kwargs = factory_kwargs
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mlp_ratio = mlp_ratio
        self.depth = depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    rms_norm=rms_norm,
                    norm_epsilon=norm_epsilon,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **self.factory_kwargs,
                )
            )
        self.enc_drop_path = nn.Identity()

        self.initialize_weights()


    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size(), cls_token=self.use_cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.use_cls_token:
            torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(
            partial(
                _init_weights,
                n_layer=self.depth,
            )
        )
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
    
    def _forward_blocks(self, x, inference_params=None):
        residual = None
        hidden_states = x
        for layer in self.blocks:
            # print("in forward_encoder, before layer:", hidden_states.shape)
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            # print("in forward_encoder, after layer:", hidden_states.shape)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.enc_drop_path(hidden_states)
            
            hidden_states = self.encoder_norm(residual.to(dtype=self.encoder_norm.weight.dtype))
            
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.encoder_norm, RMSNorm) else layer_norm_fn
            # print("in fused_add_norm, residual.shape:", residual.shape)
            hidden_states = fused_add_norm_fn(
                self.enc_drop_path(hidden_states),
                self.encoder_norm.weight,
                self.encoder_norm.bias,
                eps=self.encoder_norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


encoder_configs = {
    "tiny": {
        "depth": 12, "num_heads": 3, "embed_dim": 192
    },
    "small": {
        "depth": 12, "num_heads": 6, "embed_dim": 384
    },
    "medium": {
        "depth": 12, "num_heads": 8, "embed_dim": 512
    },
    "base": {
        "depth": 12, "num_heads": 12, "embed_dim": 768
    },
    "large": {
        "depth": 24, "num_heads": 16, "embed_dim": 1024
    },
    "huge": {
        "depth": 32, "num_heads": 16, "embed_dim": 1280
    }
}


def _get_mamba_ssast(encoder_name, **kwargs):
    img_size = kwargs.pop("img_size", (80, 200))
    patch_size = kwargs.pop("patch_size", (16, 4))
    frequency_first = kwargs.pop("frequency_first", True)
    enc_params = encoder_configs[encoder_name]

    residual_in_fp32 = kwargs.pop("residual_in_fp32", True)
    rms_norm = kwargs.pop("rms_norm", True)
    fused_add_norm = kwargs.pop("fused_add_norm", True)

    ssm_kwargs = kwargs.pop("ssm_kwargs", {
        "d_state":24,
        "d_conv":4,
        "expand":3,
    })
    print("PROVIDED SSM_KWARGS:", ssm_kwargs)
    return MambaSSAST(
        img_size=img_size,
        patch_size=patch_size,
        frequency_first=frequency_first,
        residual_in_fp32=residual_in_fp32,
        rms_norm=rms_norm,
        fused_add_norm=fused_add_norm,
        ssm_cfg=ssm_kwargs, 
        **enc_params, **kwargs
    )


def mamba_ssast_tiny(**kwargs):
    return _get_mamba_ssast("tiny", **kwargs)


def mamba_ssast_small(**kwargs):
    return _get_mamba_ssast("small", **kwargs)


def mamba_ssast_medium(**kwargs):
    return _get_mamba_ssast("medium", **kwargs)


def mamba_ssast_base(**kwargs):
    return _get_mamba_ssast("base", **kwargs)


def mamba_ssast_large(**kwargs):
    return _get_mamba_ssast("large", **kwargs)


def mamba_ssast_huge(**kwargs):
    return _get_mamba_ssast("huge", **kwargs)
