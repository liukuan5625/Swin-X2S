"""
adapted from monai.networks.nets.varautoencoder
The original Autoencoder in the monai.networks.nets.autoencoder does not have a
1D bottlneck latent vector layer i.e. it is fully convolutional.
In our application we need a low-dimensional  1D embedding of the Biplanar x-rays,
hence the need for full connected layer in between the encoder and the decoder
"""
import math
from torch import nn
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets.unet import Unet
from monai.networks.nets.unetr import UNETR
from .swin_x2s import SwinX2S
from .twoDPC import TwoDPermuteConcat
from .oneDC import OneDConcat
from .MStwoDPC import MultiScale2DPermuteConcat
from .TLP import CustomAutoEncoder, TLPredictor


class Register:
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f'Value of a Registry must be a callable!\nValue:{value}')
        if key is None:
            key = value.__name__
        if key in self._dict:
            print(f'Key {key} already in registry {self._name}')
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class"""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        return self._dict.keys()


def get_model(args) -> nn.Module:
    ARCHITECTURES = Register('architectures')
    name2arch = ARCHITECTURES

    for model in [
        SwinX2S,
        Unet,
        UNETR,
        SwinUNETR,
        AttentionUnet,
        OneDConcat,
        TwoDPermuteConcat,
        MultiScale2DPermuteConcat,
        CustomAutoEncoder
    ]:
        name2arch[model.__name__] = model

    model_name = args.model_name
    if args.model_name in ("Swin-X2S-Tiny", "Swin-X2S-Small", "Swin-X2S-Base", "Swin-X2S-Large"):
        model_name = "SwinX2S"
    """return encoder-decoder model"""
    if model_name in name2arch:
        return name2arch[model_name](**get_model_config(args))

    raise ValueError(f"invalid model name {args.model_name}")


def get_model_config(args):
    """model parameters"""
    model_name = args.model_name

    if model_name in ("Swin-X2S-Tiny", "Swin-X2S-Small", "Swin-X2S-Base", "Swin-X2S-Large"):
        return get_swin_x2s_config(args)
    elif model_name in (OneDConcat.__name__, "OneDConcatModel"):
        return get_1dconcatmodel_config(args)
    elif model_name == AttentionUnet.__name__:
        return get_attunet_config(args)
    elif model_name == SwinUNETR.__name__:
        return get_swinunetr_config(args)
    elif model_name == UNETR.__name__:
        return get_unetr_config(args)
    elif model_name in (TwoDPermuteConcat.__name__, "TwoDPermuteConcatModel"):
        return get_2dconcatmodel_config(args)
    elif model_name == Unet.__name__:
        return get_unet_config(args)
    elif model_name == MultiScale2DPermuteConcat.__name__:
        return get_multiscale2dconcatmodel_config(args)
    elif model_name == CustomAutoEncoder.__name__:
        return get_autoencoder_config(args)
    else:
        raise ValueError(f"invalid model name {model_name}")


def get_swin_x2s_config(args):
    if args.model_name.find('Tiny') != -1:
        feature_size = 32
        depths = (2, 2, 6, 2)
        num_heads = (1, 2, 4, 8)
    elif args.model_name.find('Small') != -1:
        feature_size = 64
        depths = (2, 2, 6, 2)
        num_heads = (2, 4, 8, 16)
    elif args.model_name.find('Base') != -1:
        feature_size = 96
        depths = (2, 2, 18, 2)
        num_heads = (3, 6, 12, 24)
    elif args.model_name.find('Large') != -1:
        feature_size = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)

    args.spatial_dims = 2
    model_config = {
        "img_size": [args.roi_x, args.roi_y, args.roi_z],
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "feature_size": feature_size,
        "depths": depths,
        "num_heads": num_heads,
        "fusion_depths": (1, 1, 1, 1, 1, 1),
        "drop_rate": args.dropout_rate,
        "attn_drop_rate": args.attn_drop_rate,
        "window_size": (7, 7),
    }
    return model_config


def get_unetr_config(args):
    args.spatial_dims = 3
    model_config = {
        "in_channels": args.in_channels * 2,
        "out_channels": args.out_channels,
        "img_size": [args.roi_x, args.roi_y, args.roi_z],
        "dropout_rate": args.dropout_rate,
    }
    return model_config


def get_autoencoder_config(args):
    args.spatial_dims = 3
    model_config = {
        "image_size": [args.roi_x, args.roi_y, args.roi_z],
        "latent_dim": 64,
        "spatial_dims": 3,
        "in_channels": args.in_channels * 2,
        "out_channels": args.out_channels,
        "channels": (8, 16, 32, 64),
        "strides": (2, 2, 2, 2),
        "dropout": args.dropout_rate,
    }
    return model_config


def get_swinunetr_config(args):
    """these parameters were found by searching through possible model sizes"""
    args.spatial_dims = 3
    model_config = {
        "img_size": [args.roi_x, args.roi_y, args.roi_z],
        "in_channels": args.in_channels * 2,
        "out_channels": args.out_channels,
        "depths": (2, 2, 2, 2),
        "num_heads": (3, 6, 12, 24),
        "feature_size": 48,
        "norm_name": "instance",
        "drop_rate": args.dropout_rate,
        "attn_drop_rate": 0.0,
        "dropout_path_rate": 0.0,
        "normalize": True,
        "use_checkpoint": False,
        "spatial_dims": args.spatial_dims,
        "downsample": "merging",
    }
    return model_config


def get_unet_config(args):
    """End-To-End Convolutional Neural Network for 3D Reconstruction of Knee Bones From Bi-Planar X-Ray Images
    https://arxiv.org/pdf/2004.00871.pdf
    """
    args.spatial_dims = 3
    model_config = {
        "spatial_dims": args.spatial_dims,
        "in_channels": args.in_channels * 2,
        "out_channels": args.out_channels,
        "channels": (8, 16, 32, 64, 128),
        "strides": (2, 2, 2, 2),
        "act": "RELU",
        "norm": "BATCH",
        "num_res_units": 2,
        "dropout": args.dropout_rate,
    }
    return model_config


def get_multiscale2dconcatmodel_config(args):
    """fully conv: image size does not matter"""
    args.spatial_dims = 2
    model_config = {
        "permute": True,
        "out_channels": args.out_channels,
        "dropout": args.dropout_rate,
        "encoder": {
            "initial_channel": 16,
            "in_channels": [],  # this will be filled in by autoconfig
            "out_channels": [2, 4, 8, 16, 32, 64],
            "encoder_count": 4,
            "kernel_size": 3,
            "act": "RELU",
            "norm": "BATCH",
        },
        "decoder_2D": {
            "in_channels": [],  # this will be filled in by autoconfig
            "out_channels": [4, 8, 16, 32, 64, 128],
            "kernel_size": 3,
            "act": "RELU",
            "norm": "BATCH",
        },
        "fusion_3D": {
            "in_channels": [],  # this will be filled in by autoconfig
            "out_channels": [32, 32, 32, 32, 32, 32],
            "kernel_size": 3,
            "act": "RELU",
            "norm": "BATCH",
        },
    }
    # constrain decoder configuration based on encoder
    enc_in = []
    for i in range(len(model_config["encoder"]["out_channels"])):
        if i == 0:
            enc_in.append(model_config["encoder"]["initial_channel"])
        else:
            enc_in.append(
                model_config["encoder"]["out_channels"][i - 1]
                * model_config["encoder"]["encoder_count"]
                + enc_in[i - 1]
            )
    model_config["encoder"]["in_channels"] = enc_in

    enc_out = [
        in_ch + out_ch * model_config["encoder"]["encoder_count"]
        for in_ch, out_ch in zip(
            model_config["encoder"]["in_channels"],
            model_config["encoder"]["out_channels"],
        )
    ]
    enc_out.reverse()
    dec_out = model_config["decoder_2D"]["out_channels"]
    dec_in = []
    for i in range(len(enc_out)):
        # decoder takes input from the adjacent encoder and previous decoder
        if i == 0:  # if this is the first decoder, then there is no previous decoder
            dec_in.append(enc_out[i])
        else:
            dec_in.append(enc_out[i] + dec_out[i - 1])
    model_config["decoder_2D"]["in_channels"] = dec_in

    fusion_in = []
    fusion_out = model_config["fusion_3D"]["out_channels"]
    for i in range(len(fusion_out)):
        if i == 0:
            fusion_in.append(2)  # AP and LAT view reshaped into 3D view
        else:
            fusion_in.append(
                2 + fusion_out[i - 1]
            )  # 2 channels from AP and LAT decoder, additional channel from earlier fusion decoder
    model_config["fusion_3D"]["in_channels"] = fusion_in

    return {'config': model_config}


def get_2dconcatmodel_config(args):
    """Inferring the 3D Standing Spine Posture from 2D Radiographs
    https://arxiv.org/abs/2007.06612
    default baseline for 64^3 volume
    """
    args.spatial_dims = 2
    expansion_depth = int(math.log2(args.roi_x)) - 2

    model_config = {
        "input_image_size": [args.roi_x, args.roi_z],
        "encoder_in_channels": [args.in_channels, 16, 32, 32, 32, 32],
        "encoder_out_channels": [16, 32, 32, 32, 32, 32],
        "encoder_strides": [2, 2, 1, 1, 1, 1],
        "encoder_kernel_size": 7,
        "ap_expansion_in_channels": [32, 32, 32, 32, 32][:expansion_depth],
        "ap_expansion_out_channels": [32, 32, 32, 32, 32][:expansion_depth],
        "ap_expansion_strides": ((2, 1, 1),) * expansion_depth,
        "ap_expansion_kernel_size": 3,
        "lat_expansion_in_channels": [32, 32, 32, 32, 32][:expansion_depth],
        "lat_expansion_out_channels": [32, 32, 32, 32, 32][:expansion_depth],
        "lat_expansion_strides": ((1, 2, 1),) * expansion_depth,
        "lat_expansion_kernel_size": 3,
        "decoder_in_channels": [64, 64, 64, 64, 64, 32, 16],
        "decoder_out_channels": [64, 64, 64, 64, 32, 16, args.out_channels],
        "decoder_strides": (1, 1, 1, 1, 2, 2, 1),
        "decoder_kernel_size": (3, 3, 3, 3, 3, 3, 7),
        "activation": "RELU",
        "norm": "BATCH",
        'dropout': args.dropout_rate,
        "dropout_rate": args.dropout_rate,
        "bias": True,
    }
    return model_config


def get_attunet_config(args):
    """Attention U-Net: Learning Where to Look for the Pancreas
    https://arxiv.org/abs/1804.03999
    """
    args.spatial_dims = 3
    model_config = {
        'spatial_dims': args.spatial_dims,
        "in_channels": args.in_channels * 2,
        "out_channels": args.out_channels,
        "channels": (8, 16, 32, 64, 128),
        "strides": (2, 2, 2, 2),
        "dropout": args.dropout_rate,
    }
    return model_config


def get_1dconcatmodel_config(args):
    """base model for 128^3 volume (2^7)"""
    args.spatial_dims = 2
    bottleneck_size = 256

    model_config = {
        "input_image_size": [args.roi_x, args.roi_z],
        "encoder_in_channels": [
                args.in_channels,
                32,
                64,
                128,
                256,
                512,
            ],
        "encoder_out_channels": [32, 64, 128, 256, 512, 1024],
        "encoder_strides": [2, 2, 2, 2, 2, 2],
        "decoder_in_channels": [bottleneck_size, 1024, 512, 256, 128, 64, 32],
        "decoder_out_channels": [1024, 512, 256, 128, 64, 32, args.out_channels],
        "decoder_strides": [
                2,
            ]
            * 7,
        "encoder_kernel_size": 3,
        "decoder_kernel_size": 3,
        "activation": "RELU",
        "norm": "BATCH",
        "dropout": args.dropout_rate,
        "dropout_rate": args.dropout_rate,
        "bias": True,
        "bottleneck_size": bottleneck_size,
    }

    assert model_config['decoder_in_channels'][0] == bottleneck_size, f"decoder first in-channel expected {bottleneck_size}, got {model_config['decoder_in_channels'][0]}"
    return model_config
