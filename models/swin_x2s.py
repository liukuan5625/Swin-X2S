"""SwinUNETR with cross attention."""
from typing import MutableMapping, Sequence, Tuple, Union

import math
import copy
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

__all__ = ["SwinX2S"]
FeaturesDictType = MutableMapping[str, torch.Tensor]


class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            upsample_kernel_size: int,
            norm_name: str,
            res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetrBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            norm_name: str,
            res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inp):
        return self.layer(inp)


class UnetOutBlock(nn.Module):
    def __init__(
            self, spatial_dims: int, in_channels: int, out_channels: int, dropout: float = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            dropout=dropout,
            bias=True,
            act=None,
            norm=None,
            conv_only=False,
        )

    def forward(self, inp):
        return self.conv(inp)


class Attention(nn.Module):
    def __init__(self, num_heads=8, hidden_size=768, atte_dropout_rate=0.0):
        super(Attention, self).__init__()
        # self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(atte_dropout_rate)
        self.proj_dropout = nn.Dropout(atte_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x_1, x_2):
        mixed_query_layer_1 = self.query(x_1)
        mixed_key_layer_1 = self.key(x_1)
        mixed_value_layer_1 = self.value(x_1)
        query_layer_1 = self.transpose_for_scores(mixed_query_layer_1)
        key_layer_1 = self.transpose_for_scores(mixed_key_layer_1)
        value_layer_1 = self.transpose_for_scores(mixed_value_layer_1)
        mixed_query_layer_2 = self.query(x_2)
        mixed_key_layer_2 = self.key(x_2)
        mixed_value_layer_2 = self.value(x_2)
        query_layer_2 = self.transpose_for_scores(mixed_query_layer_2)
        key_layer_2 = self.transpose_for_scores(mixed_key_layer_2)
        value_layer_2 = self.transpose_for_scores(mixed_value_layer_2)

        attention_scores_1 = torch.matmul(query_layer_1, key_layer_2.transpose(-1, -2))
        attention_scores_1 = attention_scores_1 / math.sqrt(self.attention_head_size)
        attention_probs_1 = self.softmax(attention_scores_1)
        # weights_st = attention_probs_st if self.vis else None
        attention_probs_1 = self.attn_dropout(attention_probs_1)
        context_layer_1 = torch.matmul(attention_probs_1, value_layer_2)
        context_layer_1 = context_layer_1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_1 = context_layer_1.size()[:-2] + (self.all_head_size,)
        context_layer_1 = context_layer_1.view(*new_context_layer_shape_1)
        attention_output_1 = self.out(context_layer_1)
        attention_output_1 = self.proj_dropout(attention_output_1)

        attention_scores_2 = torch.matmul(query_layer_2, key_layer_1.transpose(-1, -2))
        attention_scores_2 = attention_scores_2 / math.sqrt(self.attention_head_size)
        attention_probs_2 = self.softmax(attention_scores_2)
        # weights_st = attention_probs_st if self.vis else None
        attention_probs_2 = self.attn_dropout(attention_probs_2)
        context_layer_2 = torch.matmul(attention_probs_2, value_layer_1)
        context_layer_2 = context_layer_2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_2 = context_layer_2.size()[:-2] + (self.all_head_size,)
        context_layer_2 = context_layer_2.view(*new_context_layer_shape_2)
        attention_output_2 = self.out(context_layer_2)
        attention_output_2 = self.proj_dropout(attention_output_2)

        return attention_output_1, attention_output_2


class Block(nn.Module):
    def __init__(self, hidden_size=768, mlp_dim=1536, dropout_rate=0.5, num_heads=8, atte_dropout_rate=0.0):
        super(Block, self).__init__()

        del mlp_dim
        del dropout_rate

        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(num_heads=num_heads, hidden_size=hidden_size, atte_dropout_rate=atte_dropout_rate)

    def forward(self, x_1, x_2):
        x_1 = self.attention_norm(x_1)
        x_2 = self.attention_norm(x_2)
        x_1, x_2 = self.attn(x_1, x_2)
        return x_1, x_2


class TransFusion(nn.Module):
    def __init__(
            self,
            hidden_size: int = 768,
            num_layers: int = 6,
            mlp_dim: int = 1536,
            dropout_rate: float = 0.5,
            num_heads: int = 8,
            atte_dropout_rate: float = 0.0,
            roi_size: Union[Sequence[int], int] = (64, 64, 64),
            scale: int = 16,
    ):
        super().__init__()
        if isinstance(roi_size, int):
            roi_size = [roi_size for _ in range(3)]

        patch_size = (1, 1, 1)
        self.size = (roi_size[0] // patch_size[0] // scale,
                     roi_size[1] // patch_size[1] // scale,
                     roi_size[2] // patch_size[2] // scale)
        n_patches = (self.size[0] * self.size[1] * self.size[2])
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.patch_embeddings = nn.Conv3d(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        for _ in range(num_layers):
            layer = Block(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                num_heads=num_heads,
                atte_dropout_rate=atte_dropout_rate,
            )
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x_1, x_2):
        x_1 = self.patch_embeddings(x_1)
        x_2 = self.patch_embeddings(x_2)
        x_1 = x_1.flatten(2).transpose(-1, -2)
        x_2 = x_2.flatten(2).transpose(-1, -2)
        x_1 = x_1 + self.position_embeddings
        x_2 = x_2 + self.position_embeddings
        x_1 = self.dropout(x_1)
        x_2 = self.dropout(x_2)
        for layer_block in self.layer:
            x_1, x_2 = layer_block(x_1, x_2)
        x_1 = self.encoder_norm(x_1)
        x_2 = self.encoder_norm(x_2)
        B, _, hidden = x_1.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        x_1 = x_1.permute(0, 2, 1).contiguous().view(B, hidden, self.size[0], self.size[1], self.size[2])
        x_2 = x_2.permute(0, 2, 1).contiguous().view(B, hidden, self.size[0], self.size[1], self.size[2])

        return x_1, x_2


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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = list(window_size)
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= min(self.window_size):
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = [0, 0]
            self.window_size[0] = self.input_resolution[0]
            self.window_size[1] = self.input_resolution[1]
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            # calculate attention mask for SW-MSA
            # PAD
            H, W = self.input_resolution
            H = (H // self.window_size[0] + 1) * self.window_size[0]
            W = (W // self.window_size[1] + 1) * self.window_size[1]

            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # PAD
        pad_l = pad_t = 0
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape
        dims = [B, hp, wp]

        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)

        # reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, dims[1], dims[2])  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, dims[1], dims[2])  # B H' W' C
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size[0] / self.window_size[1]
        flops += nW * self.attn.flops(self.window_size[0] * self.window_size[1])
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: Noneuse checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=[0, 0] if (i % 2 == 0) else [window_size[0] // 2, window_size[1] // 2],
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])
        self.norm = nn.LayerNorm(dim * 2)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x_out = self.norm(x).reshape(-1, self.input_resolution[0] // 2, self.input_resolution[1] // 2, self.dim * 2)
        return x, x_out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=[160, 224], patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=(160, 224), patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(7, 7), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging,  # PatchMerging if (i_layer < self.num_layers - 1) else None,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.embed_dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        out_list = []
        x_mid = self.norm(x).reshape(-1, self.img_size[0] // self.patch_size,
                                     self.img_size[1] // self.patch_size, self.embed_dim)
        out_list.append(x_mid)
        for i in range(self.num_layers):
            x, x_mid = self.layers[i](x)
            out_list.append(x_mid)

        return out_list

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class SwinX2S(nn.Module):
    """SwinUNETR with cross attention."""

    def __init__(
            self,
            img_size: Union[Sequence[int], int],
            in_channels: int,
            out_channels: int,
            feature_size: int = 96,
            depths: Sequence[int] = (2, 2, 18, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            norm_name: Union[Tuple, str] = "instance",
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            spatial_dims: int = 3,
            fusion_depths: Sequence[int] = (2, 2, 2, 2, 2, 2),
            window_size: Sequence[int] = (8, 8),
    ) -> None:
        """

        Args:
            img_size (int | tuple(int)): Input image size.
            in_channels (int): Number of input image channels.
            out_channels (int): Number of output image channels.
            feature_size(int): Patch embedding dimension. Default: 96
            depths (tuple(int)): Depth of each Swin Transformer layer.:
            num_heads (tuple(int)): Number of attention heads in different layers.
            norm_name (str): Type of norm. Default: instance
            drop_rate (float): Dropout rate. Default: 0
            attn_drop_rate (float): Attention dropout rate. Default: 0
            spatial_dims (int): Dimension of processed data. Default: 3
            fusion_depths (tuple(int)): Cross attention fusion depth. Default: (2, 2, 2, 2, 2, 2)
            window_size (tuple(int)): Window size. Default: (7, 7)
        """
        super(SwinX2S, self).__init__()
        self.img_size = img_size

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        self.cross_atte6 = TransFusion(
            hidden_size=feature_size * 16,
            num_layers=fusion_depths[5],
            mlp_dim=feature_size * 32,
            num_heads=num_heads[3],
            dropout_rate=drop_rate,
            atte_dropout_rate=attn_drop_rate,
            roi_size=img_size,
            scale=32,
        )

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            depths=depths,
            img_size=img_size[1:],
            window_size=window_size,
            patch_size=2,
            num_heads=num_heads,
        )

        self.expand_ = nn.ModuleList([nn.Conv3d(1, int(img_size[0] / 2 ** i), 1)
                                      for i in range(len(depths) + 2)])
        for e in self.expand_:
            nn.init.constant(e.weight, 1)
            nn.init.constant(e.bias, 0)

    def expand_dims(self, x_list, pos):
        x_out_list = []
        for i, x in enumerate(x_list):
            x_out = x.permute(0, 3, 1, 2).unsqueeze(1)
            x_out = self.expand_[i](x_out)
            if pos:
                x_out = torch.permute(x_out, [0, 2, 3, 1, 4])
            else:
                x_out = torch.permute(x_out, [0, 2, 1, 3, 4])
            x_out_list.append(x_out)

        return x_out_list

    def forward_view_encoder(self, x, pos=True):
        """Encode features."""
        x_hiddens = self.swinViT(x)
        x_hiddens.insert(0, x.permute(0, 2, 3, 1))
        x_hiddens = self.expand_dims(x_hiddens, pos)

        x_enc0 = self.encoder1(x_hiddens[0])
        x_enc1 = self.encoder2(x_hiddens[1])
        x_enc2 = self.encoder3(x_hiddens[2])
        x_enc3 = self.encoder4(x_hiddens[3])
        x_enc4 = self.encoder5(x_hiddens[4])  # xa_hidden[3]
        x_dec4 = self.encoder10(x_hiddens[5])
        return {"enc0": x_enc0, "enc1": x_enc1, "enc2": x_enc2, "enc3": x_enc3, "enc4": x_enc4, "dec4": x_dec4}

    def forward_view_decoder(self, x_encoded: FeaturesDictType) -> torch.Tensor:
        """Decode features."""
        x_dec3 = self.decoder5(x_encoded["dec4"], x_encoded["enc4"])
        x_dec2 = self.decoder4(x_dec3, x_encoded["enc3"])
        x_dec1 = self.decoder3(x_dec2, x_encoded["enc2"])
        x_dec0 = self.decoder2(x_dec1, x_encoded["enc1"])
        x_out = self.decoder1(x_dec0, x_encoded["enc0"])
        x_logits = self.out(x_out)
        return x_logits

    def forward_view_cross_attention(
            self, xa_encoded: FeaturesDictType, xb_encoded: FeaturesDictType) -> Tuple[
        FeaturesDictType, FeaturesDictType]:
        """Inplace cross attention between views."""
        xa_encoded["dec4"], xb_encoded["dec4"] = self.cross_atte6(xa_encoded["dec4"], xb_encoded["dec4"])
        return xa_encoded, xb_encoded

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> Sequence[torch.Tensor]:
        """Two views forward."""
        xa_encoded = self.forward_view_encoder(xa, True)
        xb_encoded = self.forward_view_encoder(xb, False)

        xa_encoded, xb_encoded = self.forward_view_cross_attention(xa_encoded, xb_encoded)
        return [self.forward_view_decoder(val) for val in [xa_encoded, xb_encoded]]

    def no_weight_decay(self):
        """Disable weight_decay on specific weights."""
        nwd = {"swinViT.absolute_pos_embed"}
        for n, _ in self.named_parameters():
            if "relative_position_bias_table" in n:
                nwd.add(n)
        return nwd

    def group_matcher(self, coarse=False):
        """Layer counting helper, used by timm."""
        return dict(
            stem=r"^swinViT\.absolute_pos_embed|patch_embed",  # stem and embed
            blocks=r"^swinViT\.layers(\d+)\.0"
            if coarse
            else [
                (r"^swinViT\.layers(\d+)\.0.downsample", (0,)),
                (r"^swinViT\.layers(\d+)\.0\.\w+\.(\d+)", None),
                (r"^swinViT\.norm", (99999,)),
            ],
        )


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format

    m = SwinX2S(img_size=[128, 128, 160],
                 in_channels=20,
                 out_channels=26,
                 feature_size=96,
                 window_size=[7, 7],
                 num_heads=(3, 6, 12, 24),
                 )
    x = torch.ones([1, 20, 128, 160])
    flops, params = profile(m, inputs=(x, x,))
    flops, params = clever_format([flops, params])
    print(flops, params)
