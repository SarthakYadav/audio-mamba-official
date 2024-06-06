import torch
import torch.nn as nn
import collections.abc
from itertools import repeat
from functools import partial
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Any
from timm.models.vision_transformer import LayerScale, DropPath
from timm.layers.weight_init import lecun_normal_, trunc_normal_tf_, trunc_normal_, variance_scaling_

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


layernorm_wrapper = partial(nn.LayerNorm, eps=1e-6)


# class PatchEmbed(PatchEmbedTimm):
#     """Image to Patch Embedding using conv layers"""

#     def __init__(self, img_size=224, 
#                  patch_size=16, in_chans=1, 
#                  embed_dim=768, norm_layer=None, flatten=True):
        
#         super(PatchEmbed, self).__init__(
#             img_size=img_size,
#             patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#             norm_layer=norm_layer,
#             flatten=flatten,
#             output_fmt=None,
#             bias=True
#         )

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.
            ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_layer = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj_layer = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop != 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop != 0 else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_layer(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute((2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = (q @ torch.swapaxes(k, -2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = torch.swapaxes((attn @ v), 1, 2).reshape(B, N, C)
        x = self.proj_layer(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int = None,
            drop: float = 0.,
            act_layer=nn.GELU
        ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop) if drop != 0 else nn.Identity()
        self.layer2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop) if drop != 0 else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.drop1(self.act1(self.layer1(x)))
        x = self.drop2(self.layer2(x))
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: int,
            qkv_bias: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values = None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=layernorm_wrapper,
            mlp_layer=Mlp

        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop, attn_drop=attn_drop, 
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def window_partition1d(x, window_size):
    B, W, C = x.shape
    x = x.reshape(B, W // window_size, window_size, C)
    # print(x.shape)
    windows = x.reshape(-1, window_size, C)
    return windows


def window_reverse1d(windows, window_size, W: int):
    B = int(windows.shape[0] / (W / window_size))
    if B == 0:
        # means originally input had a batch size of 1
        B = 1
    x = windows.reshape(B, W // window_size, window_size, -1)
    # print("in window_reverse1d, x.shape:", x.shape)
    x = x.reshape(B, W, -1)
    return x


def get_relative_position_index1d(win_w):
    coords_flatten = torch.stack(torch.meshgrid(torch.arange(win_w)))
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Ww, Ww

    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_w - 1
    return relative_coords.sum(-1)


class WindowedAttentionHead(nn.Module):
    def __init__(self, 
                 head_dim: int,
                 window_size: int,
                 shift_windows: bool = False,
                 attn_drop: float = 0.,
                 ):
        super().__init__()
        self.head_dim = head_dim
        self.window_size = window_size
        self.shift_windows = shift_windows
        self.attn_drop = attn_drop

        # the desired shape for position bias table is (2*self.window_size-1, 1),
        # in line with the original jax implementation
        # but we need this gymnastics to get correct fan_in and fan_out for variance_scaling_ init
        
        relative_position_bias_table = torch.zeros(2*self.window_size-1, 1)

        variance_scaling_(relative_position_bias_table.transpose(1,0).contiguous(), scale=0.02, 
                          mode='fan_in', distribution='truncated_normal')
        
        self.relative_position_bias_table = nn.Parameter(relative_position_bias_table.contiguous())

        self.register_buffer("relative_position_index", get_relative_position_index1d(self.window_size))

        self.scale = head_dim ** -0.5
        self.window_area = self.window_size * 1
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop != 0. else nn.Identity()

        if self.shift_windows:
            self.shift_size = self.window_size // 2
        else:
            self.shift_size = 0

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias

    def forward(self, q, k, v):
        B, W, C = q.shape
        mask = None
        cnt = 0
        if self.shift_size > 0:
            img_mask = torch.zeros((1, W, 1))
            for w in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)):
                img_mask[:, w, :] = cnt
                cnt += 1
            mask_windows = window_partition1d(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(-1, self.window_size)
            mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            mask = mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))

            q = torch.roll(q, shifts=-self.shift_size, dims=1)
            k = torch.roll(k, shifts=-self.shift_size, dims=1)
            v = torch.roll(v, shifts=-self.shift_size, dims=1)

        else:
            mask = None

        q = window_partition1d(q, self.window_size)
        k = window_partition1d(k, self.window_size)
        v = window_partition1d(v, self.window_size)

        attn = (q @ torch.swapaxes(k, -2, -1)) * self.scale
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            B_, N, _ = attn.shape
            num_win = mask.shape[0]
            attn = attn.reshape(B_//num_win, num_win, N, N) + mask[None, Ellipsis]
            attn = attn.reshape(-1, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        x = (attn @ v)

        shifted_x = window_reverse1d(x, self.window_size, W=W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x
        return x, attn


class AttentionHead(nn.Module):
    def __init__(self, head_dim, attn_drop) -> None:
        super().__init__()
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop != 0. else nn.Identity()
    
    def forward(self, q, k, v):
        B, W, C = q.shape
        attn = (q @ torch.swapaxes(k, -2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        return x, attn
    

class WindowedMultiHeadAttention(nn.Module):
    def __init__(self, 
                 dim: int,
                 window_sizes: Union[list, tuple, int],
                 shift_windows: bool = False,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 ):
        super().__init__()
        self.dim = dim
        self.shift_windows = shift_windows
        self.window_sizes = window_sizes
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        if type(self.window_sizes) == int:
            window_sizes = _ntuple(self.num_heads)(self.window_sizes)
        else:
            window_sizes = self.window_sizes
            assert len(window_sizes) == self.num_heads
        
        self.attn_heads = nn.ModuleList()
        for i in range(self.num_heads):
            ws_i = window_sizes[i]
            if ws_i == 0:
                self.attn_heads.append(AttentionHead(self.head_dim, self.attn_drop))
            else:
                self.attn_heads.append(WindowedAttentionHead(
                    self.head_dim,
                    window_size=ws_i,
                    shift_windows=self.shift_windows,
                    attn_drop=self.attn_drop
                ))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop != 0. else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute((2, 3, 0, 1, 4))
        q, k, v = qkv
        o = []
        for i in range(self.num_heads):
            head_i, attn_i = self.attn_heads[i](q[i], k[i], v[i])
            head_i = head_i.unsqueeze(0)
            o.append(head_i)
        o = torch.cat(o, dim=0)
        o = o.permute((1, 2, 0, 3)).reshape(B, N, -1)
        o = self.proj(o)
        o = self.proj_drop(o)
        return o
    

class MWMHABlock(nn.Module):
    def __init__(self, 
                 dim: int,
                 num_heads: int,
                 window_sizes: Union[list, tuple, int],
                 mlp_ratio: float = 4.,
                 shift_windows: bool = False,
                 qkv_bias: bool = False,
                 proj_drop: float = 0.,
                 attn_drop: float = 0.,
                 init_values: Any = None,
                 drop_path: float = 0.,
                 act_layer=nn.GELU,
                 norm_layer=layernorm_wrapper,
                 mlp_layer=Mlp
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = norm_layer(dim)
        self.attn = WindowedMultiHeadAttention(
            dim,
            window_sizes,
            shift_windows=shift_windows,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
