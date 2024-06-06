import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Any
from .patch_embed import PatchEmbed
from .pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed, get_sinusoid_encoding_table
from .layers import Block


__all__ = [
    "SSAST", "ssast_tiny", "ssast_small", "ssast_medium", "ssast_base", "ssast_large", "ssast_huge"
]


layernorm_wrapper = partial(nn.LayerNorm, eps=1e-6)


class BaseSSAST(nn.Module):
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
                 norm_layer=layernorm_wrapper
                 ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.frequency_first = frequency_first
        self.masking_mode = masking_mode
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            frequency_first=frequency_first
        )
        self.num_patches = self.patch_embed.num_patches
        total_patches = self.num_patches
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            total_patches += 1
        else:
            self.cls_token = None
        self.total_patches = total_patches
        self.blocks = None

        self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, embed_dim), requires_grad=False)
        self.encoder_norm = norm_layer(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pred = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, self.img_patch_dim()))

    def img_patch_dim(self):
        patch_size = self.patch_embed.patch_size
        return patch_size[0] * patch_size[1] * self.in_chans
    
    def patch_size(self):
        return self.patch_embed.patch_size
    
    def grid_size(self):
        return self.patch_embed.grid_size

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size(), cls_token=self.use_cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.use_cls_token:
            torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        # corresponds to how LayerNorm is initialized in Flax
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size[0]*patch_size[0]*in_chans)
        """
        ph, pw = self.patch_size()
        h, w = self.grid_size()

        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, ph, w, pw))
        # standardized patch layout irrespective of whether frequency channel comes first or second
        if self.frequency_first:
            x = torch.einsum('nchpwq->nwhqpc', x)
        else:
            x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, self.img_patch_dim()))
        return x
    
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (100 - mask_ratio*100) / 100)

        if self.masking_mode == "unstructured":
            noise = torch.rand(N, L, device=x.device)
        elif self.masking_mode == "timestep":
            grid_size = self.grid_size()
            noise = torch.rand(N, L//grid_size[1], device=x.device)
            noise = torch.repeat_interleave(noise, repeats=grid_size[1], dim=1)
        else:
            raise NotImplementedError(f"masking_mode={self.masking_mode} is not implemented")

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore
    
    def _forward_blocks(self, x, inference_params=None):
        for blk in self.blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x

    def forward(self, imgs, inference_params=None):
    
        target = self.patchify(imgs)
    
        x = self.patch_embed(imgs)
    
        if self.use_cls_token:
            x = x + self.pos_embed[:, 1:, :]
        else:
            x = x + self.pos_embed[:, :, :]

        mask, ids_restore = self.random_masking(x, self.mask_ratio)

        mask_tokens = self.mask_token.repeat(x.shape[0], x.shape[1], 1)
        x = x * (1 - mask.unsqueeze(-1)) + mask_tokens * mask.unsqueeze(-1)

        if self.use_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self._forward_blocks(x, inference_params=inference_params)

        # prediction
        x = self.pred(x)

        if self.use_cls_token:
            x = x[:, 1:, :]
        return x, target, None

    def forward_features(self, x, inference_params=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.use_cls_token:
            x = x + self.pos_embed[:, 1:, :]
        else:
            x = x + self.pos_embed[:, :, :]
        if self.use_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        x = self._forward_blocks(x, inference_params=inference_params)
        
        if self.use_cls_token:
            outcome = x[:, 1:, :]
        else:
            outcome = x[:, :, :]
        grid_size = self.grid_size()
        if self.frequency_first:
            f, t = grid_size
        else:
            t, f = grid_size
        outcome = rearrange(outcome, 'b (f t) d -> b t (f d)', f=f, d=self.embed_dim)
        return outcome


class SSAST(BaseSSAST):
    def __init__(self, img_size=(80, 200), patch_size=(16, 4), in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, mask_ratio=0.5, masking_mode: str = "unstructured", use_cls_token=True, frequency_first=True, norm_layer=layernorm_wrapper) -> None:
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, mask_ratio, masking_mode, use_cls_token, frequency_first, norm_layer)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer
                )
            )
        self.initialize_weights()


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


def _get_ssast(encoder_name, **kwargs):
    img_size = kwargs.pop("img_size", (80, 200))
    patch_size = kwargs.pop("patch_size", (16, 4))
    frequency_first = kwargs.pop("frequency_first", True)
    enc_params = encoder_configs[encoder_name]
    return SSAST(img_size=img_size, patch_size=patch_size, frequency_first=frequency_first, 
                    **enc_params, **kwargs)


def ssast_tiny(**kwargs):
    return _get_ssast("tiny", **kwargs)


def ssast_small(**kwargs):
    return _get_ssast("small", **kwargs)


def ssast_medium(**kwargs):
    return _get_ssast("medium", **kwargs)


def ssast_base(**kwargs):
    return _get_ssast("base", **kwargs)


def ssast_large(**kwargs):
    return _get_ssast("large", **kwargs)


def ssast_huge(**kwargs):
    return _get_ssast("huge", **kwargs)
