import argparse
import os
import nibabel as nib
import numpy as np
import torch
import torch.multiprocessing as mp
from inferers import double_sliding_window_inference
from utils.data_utils import get_loader
from utils.cbctdata_utils import get_cbct_loader
from utils.misc import resample_3d, cal_localization_error_and_identity_rate
from monai.networks.utils import one_hot
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from models import get_model
import re
import warnings
from torch.cuda.amp import autocast
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Swin X2S reconstruction pipeline")
parser.add_argument(
    "--weight_dir", default="./weights/", type=str, help="weight checkpoint directory",
)
parser.add_argument("--weight_model_name", default="totalall_swinx2sbase_view1.pt", type=str, help="weight model name")

parser.add_argument("--logdir", default="./testlog/", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--data_dir", default="./dataset/Totalsegmentator/data_list.json", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="total", type=str, help="experiment name")

parser.add_argument("--model_name", default="Swin-X2S-Base", type=str, help="feature size")
parser.add_argument("--roi_x", default=128, type=int, help="xray and ct resolution")
parser.add_argument("--roi_y", default=128, type=int, help="xray and ct resolution")
parser.add_argument("--roi_z", default=160, type=int, help="xray and ct resolution")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=31, type=int, help="number of output channels"
                    )
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--dropout_rate", default=0., type=float, help="dropout rate")
parser.add_argument("--attn_drop_rate", default=0., type=float, help="drop path rate")

parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--gpu", default=0, type=int, help="which gpu to use")

parser.add_argument("--is_cbct", default=False, help="start unsupervised training")
parser.add_argument("--magnification_factor", default=1500/1400, type=float, help="data augmentation")
parser.add_argument("--mirror_prob", default=0., type=float, help="data augmentation")
parser.add_argument("--translate_prob", default=0., type=float, help="data augmentation")
parser.add_argument("--rotation_prob", default=0., type=float, help="data augmentation")

hd_per = 95


def extract_numbers_from_string(string):
    numbers = re.findall(r"/d+/./d+|/d+", string)
    return numbers


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir

    if args.model_name.find("Swin-X2S") != -1:
        args.spatial_dims = 2
    elif args.model_name == "OneDConcat":
        args.spatial_dims = 2
    elif args.model_name == "AttentionUnet":
        args.spatial_dims = 3
    elif args.model_name == "SwinUNETR":
        args.spatial_dims = 3
    elif args.model_name == "UNETR":
        args.spatial_dims = 3
    elif args.model_name == "TwoDPermuteConcat":
        args.spatial_dims = 2
    elif args.model_name == "UNet":
        args.spatial_dims = 3
    elif args.model_name == "MultiScale2DPermuteConcat":
        args.spatial_dims = 2
    elif args.model_name == "CustomAutoEncoder":
        args.spatial_dims = 3

    main_worker(gpu=args.gpu, args=args)


def main_worker(gpu, args):
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.amp = not args.noamp

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = False
    args.test_mode = True
    if args.is_cbct:
        test_loader = get_cbct_loader(args)
    else:
        test_loader = get_loader(args)

    print("gpu", args.gpu)
    weight_dir = args.weight_dir
    model_name = args.weight_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_pth = os.path.join(weight_dir, model_name)

    model = get_model(args)
    model_dict = torch.load(weight_pth, map_location=device)["state_dict"]
    model.load_state_dict(model_dict, strict=True)
    model.cuda(args.gpu)
    model.eval()

    dice_func = DiceMetric(get_not_nans=True)
    hd_func = HausdorffDistanceMetric(percentile=95, get_not_nans=True)

    mean_dice = []
    mean_hd = []
    mean_loc_error = []
    mean_id_rate = []

    for idx, batch_data in enumerate(test_loader):
        with torch.no_grad():
            with (autocast(enabled=True)):
                test_inputs, test_labels = batch_data['img'], batch_data['seg']

                if args.is_cbct:
                    img_name = test_labels.meta['filename_or_obj'].replace('/', '//').replace('\\', '//').split('//')[-1].split('_seg')[0]
                else:
                    img_name = test_labels.meta['filename_or_obj'].replace('/', '//').replace('\\', '//').split('//')[-1].split('_label')[0]
                original_affine = torch.squeeze(test_labels.meta['affine']).numpy()
                scale = abs(original_affine[0, 0]) / args.magnification_factor

                _, _, h, w, d = test_labels.shape
                target_shape = (h, w, d)
                test_labels = test_labels.cpu().numpy()
                print("Inference on case {}".format(img_name))
                torch.cuda.empty_cache()
                output_list = []
                test_fuse = 0
                test_inputs = torch.stack(test_inputs, 1).cuda(args.gpu)

                if args.model_name.find("Swin-X2S") != -1:
                    is_cross = True
                else:
                    is_cross = False

                for i in range(1):
                    test_outputs_1, test_outputs_2 = double_sliding_window_inference(
                        test_inputs,
                        (args.roi_x, args.roi_y, args.roi_z),
                        1,
                        model,
                        args.spatial_dims,
                        overlap=args.infer_overlap,
                        mode="gaussian",
                        is_cross=is_cross,
                    )
                    test_fuse = test_fuse + test_outputs_1 + test_outputs_2
                output_list.append(test_fuse[0].cpu().numpy())
                test_inputs = test_inputs.cpu().numpy()

                for i, output in enumerate(output_list):
                    output = np.argmax(output, axis=0, keepdims=False)
                    output = resample_3d(output, target_shape)
                    if args.is_cbct:
                        output = np.flip(output, -1)
                        test_labels = np.flip(test_labels, -1)
                        test_inputs = np.flip(test_inputs, -1)
                    else:
                        output = np.flip(output, -3)
                        output = np.flip(output, -2)

                    target_ornt = nib.orientations.axcodes2ornt(tuple(nib.aff2axcodes(original_affine)))
                    out_ornt = [[0, 1], [1, 1], [2, 1]]
                    ornt_transf = nib.orientations.ornt_transform(out_ornt, target_ornt)

                    output = nib.orientations.apply_orientation(output, ornt_transf)
                    nib.save(
                        nib.Nifti1Image(output[:, ::-1, ::-1].astype(np.uint8), affine=original_affine),
                        os.path.join(output_directory, img_name + "_pred.nii.gz"),
                    )

                    test_labels_save = test_labels[0, 0]
                    nib.save(
                        nib.Nifti1Image(test_labels_save[:, ::-1, ::-1].astype(np.uint8), affine=original_affine),
                        os.path.join(output_directory, img_name + "_label.nii.gz"),
                    )

                    if args.spatial_dims == 2:
                        cor_test_inputs = (test_inputs[0, 0, 0: 1].repeat(128, 0) * 255).transpose(1, 0, 2)
                        nib.save(
                            nib.Nifti1Image(cor_test_inputs[:, ::-1, ::-1].astype(np.uint8), affine=original_affine),
                            os.path.join(output_directory, img_name + "_input_cor.nii.gz"),
                        )

                        sag_test_inputs = (test_inputs[0, 1, 0: 1].repeat(128, 0) * 255)
                        nib.save(
                            nib.Nifti1Image(sag_test_inputs[:, ::-1, ::-1].astype(np.uint8), affine=original_affine),
                            os.path.join(output_directory, img_name + "_input_sag.nii.gz"),
                        )
                    else:
                        cor_test_inputs = test_inputs[0, 0, 0] * 255
                        nib.save(
                            nib.Nifti1Image(cor_test_inputs[:, ::-1, ::-1].astype(np.uint8), affine=original_affine),
                            os.path.join(output_directory, img_name + "_input_cor.nii.gz"),
                        )

                        sag_test_inputs = test_inputs[0, 1, 0] * 255
                        nib.save(
                            nib.Nifti1Image(sag_test_inputs[:, ::-1, ::-1].astype(np.uint8), affine=original_affine),
                            os.path.join(output_directory, img_name + "_input_sag.nii.gz"),
                        )

                    test_labels_onehot = one_hot(torch.from_numpy(test_labels.copy()), num_classes=args.out_channels, dim=1)
                    output = np.expand_dims(np.expand_dims(output, axis=(0)), axis=0)
                    output_one_hot = one_hot(torch.from_numpy(np.ascontiguousarray(output)), num_classes=args.out_channels, dim=1)

                    # DICE
                    dice_func.reset()
                    dice_list = dice_func(y_pred=output_one_hot, y=test_labels_onehot)
                    dice_, dice_not_nans = dice_func.aggregate()
                    # HD
                    hd_func.reset()
                    hd_list = hd_func(y_pred=output_one_hot, y=test_labels_onehot)
                    hd_, hd_not_nans = hd_func.aggregate()
                    hd_ = hd_ * scale
                    # HD
                    loc_error_, id_rate_ = cal_localization_error_and_identity_rate(test_labels, output, scale)

                    print("{} img Mean Dice: {} %".format(img_name, dice_[0]*100))
                    print("{} img Mean HD: {} mm".format(img_name, hd_[0]))
                    print("{} img Mean L-error: {} mm".format(img_name, loc_error_))
                    print("{} img Identity rate: {} %".format(img_name, id_rate_*100))
                    mean_dice.append(dice_[0])
                    mean_hd.append(hd_[0])
                    mean_loc_error.append(loc_error_)
                    mean_id_rate.append(id_rate_)

    print("#########")
    print("ALL DONE.")
    print("Mean DICE: {} %".format(np.mean(mean_dice)*100))
    print("Mean HD: {} mm".format(np.mean(mean_hd)))
    print("Mean L-error: {} mm".format(np.mean(mean_loc_error)))
    print("Mean ID rate: {} %".format(np.mean(mean_id_rate) * 100))


if __name__ == "__main__":
    main()
