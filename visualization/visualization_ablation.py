import sys
sys.path.append("..")
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from palettable.cartocolors.qualitative import Bold_10
from monai import transforms
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot
from utils.misc import cal_localization_error_and_identity_rate
from PIL import Image

model_list = ["WOTRANS",
              "WOMUTUAL",
              "WOPRE",
              "WODATAAUG",
              "TINY",
              "SMALL",
              "BASE",
              "LARGE",]

colors_pal = Bold_10.mpl_colors
colors_pal = colors_pal * 6
custom_cmap_pal = ListedColormap(colors_pal)


def model_visualiztion(path_name, sample_name, out_channels, num_cor_slice, num_sag_slice):
    dice_func = DiceMetric(get_not_nans=True)
    hd_func = HausdorffDistanceMetric(percentile=95, get_not_nans=True)

    load_raw_nii = transforms.Compose([
        transforms.LoadImage(),
    ])
    raw_ct = load_raw_nii(os.path.join(path_name, sample_name + ".nii.gz"))

    load_nii = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(channel_dim='no_channel'),
        transforms.SpatialPad([128, 128, 160], mode='empty'),
    ])

    fig, axs = plt.subplots(2, len(model_list)+2, figsize=(len(model_list)+2, 2.5), dpi=3000)

    for i, model in enumerate(model_list):
        model_path = os.path.join(path_name, model)

        cor_img = load_nii(os.path.join(model_path, sample_name + "_input_cor.nii.gz"))[0]
        cor_img = np.uint8(np.flip(cor_img, -1))
        sag_img = load_nii(os.path.join(model_path, sample_name + "_input_sag.nii.gz"))[0]
        sag_img = np.uint8(np.flip(sag_img, -1))
        pred_ct = load_nii(os.path.join(model_path, sample_name + "_pred.nii.gz"))[0]
        pred_ct = np.uint8(np.flip(pred_ct, -1))
        seg_ct = load_nii(os.path.join(model_path, sample_name + "_label.nii.gz"))[0]
        scale = abs(seg_ct.meta['original_affine'][0, 0])
        seg_ct = np.uint8(np.flip(seg_ct, -1))
        masked_pred = np.ma.masked_where(pred_ct < 1, pred_ct)
        masked_seg = np.ma.masked_where(seg_ct < 1, seg_ct)

        pred_cor_img = np.array(Image.open(os.path.join(model_path, "pred_cor.png")))
        pred_cor_img = pred_cor_img[300:695, 592:908, :-1]
        pred_sag_img = np.array(Image.open(os.path.join(model_path, "pred_sag.png")))
        pred_sag_img = pred_sag_img[300:695, 592:908, :-1]
        label_cor_img = np.array(Image.open(os.path.join(model_path, "label_cor.png")))
        label_cor_img = label_cor_img[300:695, 592:908, :-1]
        label_sag_img = np.array(Image.open(os.path.join(model_path, "label_sag.png")))
        label_sag_img = label_sag_img[300:695, 592:908, :-1]

        if model in ["tlpredictor", "onedconcat"]:
            loc_error_, id_rate_ = cal_localization_error_and_identity_rate(pred_ct, seg_ct[:, :, 16: 144], scale)
            seg_onehot = one_hot(torch.from_numpy(seg_ct[:, :, 16: 144].copy()).unsqueeze(0), num_classes=out_channels, dim=0)
            pred_onehot = one_hot(torch.from_numpy(pred_ct[:, :, 16: 144].copy()).unsqueeze(0), num_classes=out_channels, dim=0)
        else:
            loc_error_, id_rate_ = cal_localization_error_and_identity_rate(pred_ct, seg_ct, scale)
            seg_onehot = one_hot(torch.from_numpy(seg_ct.copy()).unsqueeze(0), num_classes=out_channels, dim=0)
            pred_onehot = one_hot(torch.from_numpy(pred_ct.copy()).unsqueeze(0), num_classes=out_channels, dim=0)

        dice_func.reset()
        dice_func(y_pred=pred_onehot, y=seg_onehot)
        dice_, dice_not_nans = dice_func.aggregate()
        hd_func.reset()
        hd_func(y_pred=pred_onehot, y=seg_onehot)
        hd_, hd_not_nans = hd_func.aggregate()
        hd_ = hd_ * scale

        axs[0, 0].imshow(cor_img[:, num_cor_slice].T, cmap="gray")
        axs[0, 0].axis("off")
        axs[1, 0].imshow(sag_img[num_sag_slice].T, cmap="gray")
        axs[1, 0].axis("off")

        '''axs[0, len(model_list)+1].imshow(raw_ct[:, num_cor_slice].T, cmap="gray")
        axs[0, len(model_list)+1].imshow(masked_seg[:, num_cor_slice].T, cmap=custom_cmap_pal, vmin=0, vmax=out_channels)'''
        axs[0, len(model_list)+1].imshow(label_cor_img)
        axs[0, len(model_list)+1].axis("off")
        '''axs[1, len(model_list)+1].imshow(raw_ct[num_sag_slice].T, cmap="gray")
        axs[1, len(model_list)+1].imshow(masked_seg[num_sag_slice].T, cmap=custom_cmap_pal, vmin=0, vmax=out_channels)'''
        axs[1, len(model_list)+1].imshow(label_sag_img)
        axs[1, len(model_list)+1].axis("off")

        '''axs[0, i+1].imshow(raw_ct[:, num_cor_slice].T, cmap="gray")
        axs[0, i+1].imshow(masked_pred[:, num_cor_slice].T, cmap=custom_cmap_pal, vmin=0, vmax=out_channels)'''
        axs[0, i + 1].imshow(pred_cor_img)
        axs[0, i+1].axis("off")
        axs[0, i+1].text(0.25, 0.05, f'{dice_.item()*100:.2f}'+"%", transform=axs[0, i+1].transAxes, ha='center', va='center', fontsize=6, color='blue')
        axs[0, i+1].text(0.7, 0.05, f'{hd_.item():.2f}' + "mm", transform=axs[0, i+1].transAxes, ha='center',va='center', fontsize=6, color='orange')

        '''axs[1, i+1].imshow(raw_ct[num_sag_slice].T, cmap="gray")
        axs[1, i+1].imshow(masked_pred[num_sag_slice].T, cmap=custom_cmap_pal, vmin=0, vmax=out_channels)'''
        axs[1, i + 1].imshow(pred_sag_img)
        axs[1, i+1].axis("off")
        axs[1, i+1].text(0.25, 0.05, f'{id_rate_ * 100:.2f}' + "%", transform=axs[1, i+1].transAxes, ha='center', va='center', fontsize=6, color='green')
        axs[1, i+1].text(0.7, 0.05, f'{loc_error_:.2f}' + "mm", transform=axs[1, i+1].transAxes, ha='center', va='center', fontsize=6, color='purple')

    #plt.savefig('multi_figures.pdf')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0, hspace=0)
    plt.savefig(os.path.join(path_name, "img.svg"))
    # plt.show()


if __name__ == "__main__":
    # FIG1
    path_name = "./vis_result/ablation/FIG1"
    sample_name = "1.3.6.1.4.1.9328.50.4.0532"
    out_channels = 26
    model_visualiztion(path_name, sample_name, out_channels, num_cor_slice=44, num_sag_slice=64)

    # FIG2
    path_name = "./vis_result/ablation/FIG2"
    sample_name = "1.3.6.1.4.1.9328.50.4.0694"
    out_channels = 26
    model_visualiztion(path_name, sample_name, out_channels, num_cor_slice=45, num_sag_slice=64)
