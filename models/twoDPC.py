from typing import Dict, List
import torch
from monai.networks.blocks.convolutions import Convolution
from torch import nn


class TwoDPermuteConcat(nn.Module):
    """
    Transvert Architecture
    Bayat, Amirhossein, et al. "Inferring the 3D standing spine posture from 2D radiographs.", 2020.

        Attributes:
        pa_encoder (nn.Module): encodes AP view x-rays
        lat_encoder (nn.Module): encodes LAT view x-rays
        decoder (nn.Module): takes encoded and fused AP and LAT view and generates a 3D volume
    """

    def __init__(
        self,
        input_image_size,
        encoder_in_channels,
        encoder_out_channels,
        encoder_strides,
        encoder_kernel_size,
        decoder_in_channels,
        decoder_out_channels,
        decoder_strides,
        decoder_kernel_size,
        ap_expansion_in_channels,
        ap_expansion_out_channels,
        ap_expansion_strides,
        ap_expansion_kernel_size,
        lat_expansion_in_channels,
        lat_expansion_out_channels,
        lat_expansion_strides,
        lat_expansion_kernel_size,
        activation,
        norm,
        dropout,
        dropout_rate,
        bias,
    ) -> None:
        super().__init__()
        # verify config
        assert (
            len(input_image_size) == 2
        ), f"expected images to be 2D but got {len(input_image_size)}-D"

        self.ap_encoder: nn.Module
        self.lat_encoder: nn.Module
        self.decoder: nn.Module

        self.ap_encoder = nn.Sequential(
            *self._encoder_layer(
                encoder_in_channels,
                encoder_out_channels,
                encoder_strides,
                encoder_kernel_size,
                activation,
                norm,
                dropout,
                dropout_rate,
            )
        )
        self.lat_encoder = nn.Sequential(
            *self._encoder_layer(
                encoder_in_channels,
                encoder_out_channels,
                encoder_strides,
                encoder_kernel_size,
                activation,
                norm,
                dropout,
                dropout_rate,
            )
        )

        self.ap_expansion = nn.Sequential(
            *self._expansion_layer(
                ap_expansion_in_channels,
                ap_expansion_out_channels,
                ap_expansion_strides,
                ap_expansion_kernel_size,
                activation,
                norm,
                dropout,
                dropout_rate,
            )
        )
        self.lat_expansion = nn.Sequential(
            *self._expansion_layer(
                lat_expansion_in_channels,
                lat_expansion_out_channels,
                lat_expansion_strides,
                lat_expansion_kernel_size,
                activation,
                norm,
                dropout,
                dropout_rate,
            )
        )

        self.decoder = nn.Sequential(
            *self._decoder_layers(
                decoder_in_channels,
                decoder_out_channels,
                decoder_strides,
                decoder_kernel_size,
                activation,
                norm,
                dropout,
                dropout_rate,
                bias,
            )
        )

    def _encoder_layer(
        self,
        encoder_in_channels,
        encoder_out_channels,
        encoder_strides,
        encoder_kernel_size,
        activation,
        norm,
        dropout,
        dropout_rate,
    ):
        layers: List[nn.Module] = []

        for in_channels, out_channels, strides in zip(
            encoder_in_channels,
            encoder_out_channels,
            encoder_strides,
        ):
            layers.append(
                Convolution(
                    spatial_dims=2,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    kernel_size=encoder_kernel_size,
                    act=activation,
                    norm=norm,
                    dropout=dropout_rate if dropout else None,
                )
            )

        return layers

    def _expansion_layer(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_size,
        activation,
        norm,
        dropout,
        dropout_rate,
    ):
        layers: List[nn.Module] = []

        for in_channels, out_channels, strides in zip(
            in_channels,
            out_channels,
            strides,
        ):
            layers.append(
                Convolution(
                    spatial_dims=3,
                    is_transposed=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    kernel_size=kernel_size,
                    act=activation,
                    norm=norm,
                    dropout=dropout_rate if dropout else None,
                )
            )
        return layers

    def _decoder_layers(
        self,
        decoder_in_channels,
        decoder_out_channels,
        decoder_strides,
        decoder_kernel_size,
        activation,
        norm,
        dropout,
        dropout_rate,
        bias,
    ):
        layers: List[nn.Module] = []
        for index, (in_channels, out_channels, strides, kernel_size) in enumerate(
            zip(
                decoder_in_channels,
                decoder_out_channels,
                decoder_strides,
                decoder_kernel_size,
            )
        ):
            if index == len(decoder_strides) - 1:
                conv_only = True
                # According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                # if a conv layer is directly followed by a batch norm layer, bias should be False.
            else:
                conv_only = False
            layers.append(
                Convolution(
                    spatial_dims=3,
                    is_transposed=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    kernel_size=kernel_size,
                    act=activation,
                    norm=norm,
                    dropout=dropout_rate if dropout else None,
                    bias=bias,
                    conv_only=conv_only,
                )
            )
        return layers

    def forward(self, image):
        N, C, _, _ = image.shape
        ap_image = image[:, :C // 2]
        lat_image = image[:, C // 2:]

        out_ap = self.ap_encoder(ap_image)
        out_lat = self.lat_encoder(lat_image)

        out_ap_expansion = self.ap_expansion(out_ap.unsqueeze(2))
        out_lat_expansion = self.lat_expansion(out_lat.unsqueeze(3))

        fused_cube = torch.cat(
            (out_ap_expansion, out_lat_expansion), dim=1
        )  # add new dimension assuming PIR orientation
        return self.decoder(fused_cube)