import argparse
import logging
import os

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.nn.parallel
from models import get_model
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from timm.utils import setup_default_logging
from torch.nn import KLDivLoss
from trainer import run_training

from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from utils.data_utils import get_loader
from utils.cbctdata_utils import get_cbct_loader

parser = argparse.ArgumentParser(description="Swin-X2S reconstruction pipeline")
parser.add_argument("--logdir", default="./totalall_swinx2sbase_view1/", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--use_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--pretrained_model_name", default="swin_small_patch4_window7_224_22k.pth", type=str, help="pretrained model name")
parser.add_argument("--data_dir", default="./dataset/Totalsegmentator_all/data_list.json", type=str, help="dataset directory")

parser.add_argument("--model_name", default="Swin-X2S-Base", type=str, help="feature size")
parser.add_argument("--roi_x", default=128, type=int, help="xray and ct resolution")
parser.add_argument("--roi_y", default=128, type=int, help="xray and ct resolution")
parser.add_argument("--roi_z", default=160, type=int, help="xray and ct resolution")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=31, type=int, help="number of output channels")

parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=250, type=int, help="max number of training epochs")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
parser.add_argument("--val_start", default=0, type=int, help="val start from epoch")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-2, type=float, help="regularization weight")
parser.add_argument("--layer_decay", default=1, type=float, help="layer-wise learning rate decay")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--dropout_rate", default=0., type=float, help="dropout rate")
parser.add_argument("--attn_drop_rate", default=0., type=float, help="drop path rate")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=20, type=int, help="number of warmup epochs")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--norm_name", default="batch", help="multi gpu use")
parser.add_argument("--loss_name", default="DiceCE", type=str, help="optimization algorithm")
parser.add_argument("--unsupervised", action="store_true", help="start unsupervised training")
parser.add_argument("--gpu", default=0, type=int, help="which gpu to use")
parser.add_argument("--workers", default=6, type=int, help="num of workers")

parser.add_argument("--is_cbct", default=False, type=bool, help="is cone beam ct?")
parser.add_argument("--mirror_prob", default=0.5, type=float, help="data augmentation")
parser.add_argument("--translate_prob", default=0.9, type=float, help="data augmentation")
parser.add_argument("--rotation_prob", default=0.9, type=float, help="data augmentation")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    main_worker(gpu=args.gpu, args=args)


def main_worker(gpu, args):
    ###  INITIALIZE
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    print(" gpu", args.gpu)
    setup_default_logging()
    logging.info(f"Batch size is: {args.batch_size}, epochs: {args.max_epochs}")

    ###  MODEL
    model = get_model(args)
    model.cuda(args.gpu)
    model_without_ddp = model

    pretrained_dir = args.pretrained_dir
    if args.use_pretrained:
        try:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name), map_location="cpu", weights_only=False)
            try:
                state_dict = model_dict["model"]
                state_dict['patch_embed.proj.weight'] = (
                    state_dict['patch_embed.proj.weight'][:, :, ::2, ::2].repeat(1, 10, 1, 1))[:, :args.in_channels]
                new_state_dict = {}
                for key in state_dict.keys():
                    new_key = 'swinViT.' + key
                    if new_key in model.state_dict().keys():
                        if model.state_dict()[new_key].shape == state_dict[key].shape:
                            new_state_dict[new_key] = state_dict[key]
                model.load_state_dict(new_state_dict, strict=False)
                logging.info("Using pretrained self-supervised Swin Transformer backbone weights !")
            except KeyError:
                state_dict = model_dict["state_dict"]
                model_dict = model.state_dict()
                matched_dict = {
                    k: v for k, v in state_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                model.load_state_dict(matched_dict, strict=False)
                logging.info("Using resumed weights !")
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters count: {pytorch_total_params}")

    param_groups = optim_factory.param_groups_layer_decay(
        model_without_ddp,
        weight_decay=args.reg_weight,
        layer_decay=args.layer_decay,
        verbose=False,
        no_weight_decay_list=model_without_ddp.no_weight_decay() if args.model_name.find("Swin-X2S") != -1 else (),
    )

    ###  LOSS
    to_onehot_y = True
    softmax = True
    if args.squared_dice:
        if args.loss_name == "Dice":
            self_crit = DiceLoss(
                to_onehot_y=to_onehot_y, softmax=softmax, squared_pred=True, smooth_nr=args.smooth_nr,
                smooth_dr=args.smooth_dr
            )
        elif args.loss_name == "DiceCE":
            self_crit = DiceCELoss(
                to_onehot_y=to_onehot_y, softmax=softmax, squared_pred=True, smooth_nr=args.smooth_nr,
                smooth_dr=args.smooth_dr
            )
        elif args.loss_name == "DiceFocal":
            self_crit = DiceFocalLoss(
                to_onehot_y=to_onehot_y, softmax=softmax, squared_pred=True, smooth_nr=args.smooth_nr,
                smooth_dr=args.smooth_dr
            )
    else:
        if args.loss_name == "Dice":
            self_crit = DiceLoss(to_onehot_y=to_onehot_y, softmax=softmax)
        elif args.loss_name == "DiceCE":
            self_crit = DiceCELoss(to_onehot_y=to_onehot_y, softmax=softmax)
        elif args.loss_name == "DiceFocal":
            self_crit = DiceFocalLoss(to_onehot_y=to_onehot_y, softmax=softmax)

    if args.model_name.find("Swin-X2S") != -1:
        mutual_crit = KLDivLoss(reduction="mean")  # CosineSimilarity(dim = 1)
    else:
        mutual_crit = None

    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    loss_func = {"self_crit": self_crit, "mutual_crit": mutual_crit}
    acc_func = {"dice_metric": DiceMetric(get_not_nans=True)}

    ###  OPTIMIZER
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=args.optim_lr)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=args.optim_lr)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=args.optim_lr, momentum=args.momentum, nesterov=True)
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        scheduler = None

    ######
    best_acc = 0
    start_epoch = 0
    if args.is_cbct:
        train_loader, val_loader = get_cbct_loader(args)
    else:
        train_loader, val_loader = get_loader(args)
    accuracy = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        acc_func=acc_func,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
