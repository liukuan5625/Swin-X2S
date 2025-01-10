# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import nibabel as nib
import numpy as np
import torch
import torch.multiprocessing as mp
from inferers import double_sliding_window_inference
from utils.data_utils import get_loader
from utils.misc import resample_3d, cal_localization_error_and_identity_rate
from monai.networks.utils import one_hot
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from models import get_model
import re
import warnings
from torch.cuda.amp import autocast
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Swin X2S segmentation pipeline")
parser.add_argument(
    "--weight_dir", default="./weights", type=str, help="weight checkpoint directory",
)
parser.add_argument("--weight_model_name", default="model.pt", type=str, help="weight model name")

parser.add_argument("--logdir", default="./testlog/", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--data_dir", default="./dataset/RibFrac/data_list_.json", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="RibFrac", type=str, help="experiment name")

parser.add_argument("--model_name", default="Swin-X2S-Base", type=str, help="feature size")
parser.add_argument("--roi_x", default=128, type=int, help="xray and ct resolution")
parser.add_argument("--roi_y", default=128, type=int, help="xray and ct resolution")
parser.add_argument("--roi_z", default=160, type=int, help="xray and ct resolution")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=25, type=int, help="number of output channels")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--dropout_rate", default=0., type=float, help="dropout rate")
parser.add_argument("--attn_drop_rate", default=0., type=float, help="drop path rate")

parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--spatial_dims", default=2, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")


hd_per = 95


def extract_numbers_from_string(string):
    numbers = re.findall(r"\d+\.\d+|\d+", string)
    return numbers


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    args.amp = not args.noamp

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = False
    args.test_mode = True
    test_loader = get_loader(args)

    print(args.rank, " gpu", args.gpu)
    weight_dir = args.weight_dir
    model_name = args.weight_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_pth = os.path.join(weight_dir, model_name)

    dice_func = DiceMetric(get_not_nans=True)
    hd_func = HausdorffDistanceMetric(percentile=95, get_not_nans=True)

    mean_dice = []
    mean_hd = []
    mean_loc_error = []
    mean_id_rate = []
    final_out_features = []

    for idx, batch_data in enumerate(test_loader):
        with torch.no_grad():
            with (autocast(enabled=True)):
                model = get_model(args)
                model_dict = torch.load(weight_pth)["state_dict"]
                model.load_state_dict(model_dict, strict=False)
                model.cuda(args.gpu)
                model.eval()

                if args.distributed:
                    torch.cuda.set_device(args.gpu)
                    if args.norm_name == "batch":
                        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                    model.cuda(args.gpu)
                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[args.gpu], output_device=args.gpu, broadcast_buffers=False,
                        find_unused_parameters=True
                    )

                test_inputs, test_labels = batch_data['img'], batch_data['seg']

                img_name = test_labels.meta['filename_or_obj'].split('\\')[-1].split('_label')[0]
                original_affine = torch.squeeze(test_labels.meta['affine']).numpy()
                scale = abs(original_affine[0, 0])

                _, _, h, w, d = test_labels.shape
                target_shape = (h, w, d)
                test_labels = test_labels.cpu().numpy()
                print("Inference on case {}".format(img_name))
                torch.cuda.empty_cache()
                output_list = []
                test_fuse = 0
                test_inputs = torch.stack(test_inputs, 1).cuda()

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

                for i, output in enumerate(output_list):
                    # final_out_features.append(np.float16(output[1:].reshape([54, -1])))
                    output = np.argmax(output, axis=0, keepdims=False)
                    output = resample_3d(output, target_shape)
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
                        cor_test_inputs = (test_inputs[0, 0, 0: 1].repeat(128, 1, 1) * 255).permute(1, 0, 2).cpu().numpy()
                        nib.save(
                            nib.Nifti1Image(cor_test_inputs[:, ::-1, ::-1].astype(np.uint8), affine=original_affine),
                            os.path.join(output_directory, img_name + "_input_cor.nii.gz"),
                        )

                        sag_test_inputs = (test_inputs[0, 1, 0: 1].repeat(128, 1, 1) * 255).cpu().numpy()
                        nib.save(
                            nib.Nifti1Image(sag_test_inputs[:, ::-1, ::-1].astype(np.uint8), affine=original_affine),
                            os.path.join(output_directory, img_name + "_input_sag.nii.gz"),
                        )
                    else:
                        cor_test_inputs = test_inputs[0, 0, 0].cpu().numpy() * 255
                        nib.save(
                            nib.Nifti1Image(cor_test_inputs[:, ::-1, ::-1].astype(np.uint8), affine=original_affine),
                            os.path.join(output_directory, img_name + "_input_cor.nii.gz"),
                        )

                        sag_test_inputs = test_inputs[0, 1, 0].cpu().numpy() * 255
                        nib.save(
                            nib.Nifti1Image(sag_test_inputs[:, ::-1, ::-1].astype(np.uint8), affine=original_affine),
                            os.path.join(output_directory, img_name + "_input_sag.nii.gz"),
                        )

                    test_labels_onehot = one_hot(torch.from_numpy(test_labels), num_classes=args.out_channels, dim=1)
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
