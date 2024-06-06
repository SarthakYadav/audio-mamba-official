import torch
import torch.nn as nn
from .. import utils
from typing import Optional, Callable, Union, Any


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding layer


    Arguments:
        img_size: tuple
            Spectrogram inputs shape, controlled by frequency_first parameter.
                - frequency_first == True , expected input is of shape [batch, in_chans, F, T]
                - otherwise [batch, in_chans, T, F]     (default)
                This is done for consistency with the original jax implementation
        patch_size: tuple
            Patch size parameter. Format depends on frequency_first parameter.
        in_chans: int
            number of input channels
        embed_dim: int
            embedding dimension
        norm_layer: callable
            normalization layer
        flatten: bool
            whether to flatten the output
        frequency_first: bool (default: False)
            Specifies whether input has frequency or time dimension first
    """
    def __init__(
            self,
            img_size: Optional[Union[tuple, int]] = (200, 80),
            patch_size: Optional[Union[tuple, int]] = (4, 16),
            in_chans: Optional[int] = 1,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            frequency_first: bool = False
        ) -> None:
        super().__init__()
        self.img_size = utils.to_2tuple(img_size)
        patch_size = utils.to_2tuple(patch_size)
        grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.frequency_first = frequency_first
        print(f"!!!!!!!!! ATTENTION: PatchEmbed is using frequency_first = {frequency_first} !!!!!!!!!")

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        outputs = self.proj(x)
        # print("PatchEmbed after proj outputs.shape", outputs.shape)

        # maintain ordering of dimensions as per original jax implementation
        if self.frequency_first:
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.permute(0, 2, 3, 1)

        # print("PatchEmbed outputs.shape", outputs.shape)
        if self.flatten:
            outputs = outputs.reshape(B, -1, self.embed_dim)

        outputs = self.norm_layer(outputs)
        return outputs
