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

colors_pal = Bold_10.mpl_colors
colors_pal = colors_pal * 6
custom_cmap_pal = ListedColormap(colors_pal)


def subset_visualiztion(path_name, sample_name, dataset_list, num_cor_slices, num_sag_slices):
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

    fig, axs = plt.subplots(2, len(num_cor_slices) + 2, figsize=(len(num_cor_slices) + 2, 2.5), dpi=3000)

    for i, (set, N) in enumerate(dataset_list):
        set_path = os.path.join(path_name, set)
        files = os.listdir(set_path)
        nii_files = [1, 2, 3, 4]
        for f in files:
            if f.find("input_cor") != -1:
                nii_files[0] = f
            if f.find("input_sag") != -1:
                nii_files[1] = f
            if f.find("pred.nii.gz") != -1:
                nii_files[2] = f
            if f.find("label.nii.gz") != -1:
                nii_files[3] = f

        cor_img = load_nii(os.path.join(set_path, nii_files[0]))[0]
        cor_img = np.uint8(np.flip(cor_img, -1))
        sag_img = load_nii(os.path.join(set_path, nii_files[1]))[0]
        sag_img = np.uint8(np.flip(sag_img, -1))
        pred_ct = load_nii(os.path.join(set_path, nii_files[2]))[0]
        pred_ct = np.uint8(np.flip(pred_ct, -1))
        seg_ct = load_nii(os.path.join(set_path, nii_files[3]))[0]
        scale = abs(seg_ct.meta['original_affine'][0, 0])
        seg_ct = np.uint8(np.flip(seg_ct, -1))

        pred_ct_copy = pred_ct
        seg_ct_copy = seg_ct

        pred_cor_img = np.array(Image.open(os.path.join(set_path, "pred_cor.png")))
        pred_cor_img = pred_cor_img[300:695, 592:908, :-1]
        pred_sag_img = np.array(Image.open(os.path.join(set_path, "pred_sag.png")))
        pred_sag_img = pred_sag_img[300:695, 592:908, :-1]
        label_cor_img = np.array(Image.open(os.path.join(set_path, "label_cor.png")))
        label_cor_img = label_cor_img[300:695, 592:908, :-1]
        label_sag_img = np.array(Image.open(os.path.join(set_path, "label_sag.png")))
        label_sag_img = label_sag_img[300:695, 592:908, :-1]

        loc_error_, id_rate_ = cal_localization_error_and_identity_rate(pred_ct, seg_ct, scale)
        seg_onehot = one_hot(torch.from_numpy(seg_ct.copy()).unsqueeze(0), num_classes=N, dim=0)
        pred_onehot = one_hot(torch.from_numpy(pred_ct.copy()).unsqueeze(0), num_classes=N, dim=0)
        if set == "Totalsegmentator_femur":
            pred_ct_copy = np.where(pred_ct == 1, 53, pred_ct)
            pred_ct_copy = np.where(pred_ct_copy == 2, 54, pred_ct_copy)
            seg_ct_copy = np.where(seg_ct == 1, 53, seg_ct)
            seg_ct_copy = np.where(seg_ct_copy == 2, 54, seg_ct_copy)
        elif set == "Totalsegmentator_pelvic":
            pred_ct_copy = np.where(pred_ct == 2, 51, pred_ct)
            pred_ct_copy = np.where(pred_ct_copy == 3, 52, pred_ct_copy)
            pred_ct_copy = np.where(pred_ct_copy == 1, 50, pred_ct_copy)
            seg_ct_copy = np.where(seg_ct == 2, 51, seg_ct)
            seg_ct_copy = np.where(seg_ct_copy == 3, 52, seg_ct_copy)
            seg_ct_copy = np.where(seg_ct_copy == 1, 50, seg_ct_copy)
        masked_pred = np.ma.masked_where(pred_ct_copy < 1, pred_ct_copy)
        masked_seg = np.ma.masked_where(seg_ct_copy < 1, seg_ct_copy)

        dice_func.reset()
        dice_func(y_pred=pred_onehot, y=seg_onehot)
        dice_, dice_not_nans = dice_func.aggregate()
        hd_func.reset()
        hd_func(y_pred=pred_onehot, y=seg_onehot)
        hd_, hd_not_nans = hd_func.aggregate()
        hd_ = hd_ * scale

        axs[0, 0].imshow(cor_img[:, num_cor_slices[i]].T, cmap="gray")
        axs[0, 0].axis("off")
        axs[1, 0].imshow(sag_img[num_sag_slices[i]].T, cmap="gray")
        axs[1, 0].axis("off")

        '''axs[0, len(num_cor_slices)+1].imshow(raw_ct[:, num_cor_slices[i]].T, cmap="gray")
        axs[0, len(num_cor_slices)+1].imshow(masked_seg[:, num_cor_slices[i]].T, cmap=custom_cmap_pal, vmin=0, vmax=55)'''
        axs[0, len(num_cor_slices)+1].imshow(label_cor_img)
        axs[0, len(num_cor_slices)+1].axis("off")
        '''axs[1, len(num_cor_slices)+1].imshow(raw_ct[num_sag_slices[i]].T, cmap="gray")
        axs[1, len(num_cor_slices)+1].imshow(masked_seg[num_sag_slices[i]].T, cmap=custom_cmap_pal, vmin=0, vmax=55)'''
        axs[1, len(num_cor_slices)+1].imshow(label_sag_img)
        axs[1, len(num_cor_slices)+1].axis("off")

        '''axs[0, i + 1].imshow(raw_ct[:, num_cor_slices[i]].T, cmap="gray")
        axs[0, i + 1].imshow(masked_pred[:, num_cor_slices[i]].T, cmap=custom_cmap_pal, vmin=0, vmax=55)'''
        axs[0, i + 1].imshow(pred_cor_img)
        axs[0, i + 1].axis("off")
        axs[0, i + 1].text(0.25, 0.05, f'{dice_.item() * 100:.2f}' + "%", transform=axs[0, i + 1].transAxes,
                           ha='center', va='center', fontsize=6, color='blue')
        axs[0, i + 1].text(0.7, 0.05, f'{hd_.item():.2f}' + "mm", transform=axs[0, i + 1].transAxes, ha='center',
                           va='center', fontsize=6, color='orange')

        '''axs[1, i + 1].imshow(raw_ct[num_sag_slices[i]].T, cmap="gray")
        axs[1, i + 1].imshow(masked_pred[num_sag_slices[i]].T, cmap=custom_cmap_pal, vmin=0, vmax=55)'''
        axs[1, i + 1].imshow(pred_sag_img)
        axs[1, i + 1].axis("off")
        axs[1, i + 1].text(0.25, 0.05, f'{id_rate_ * 100:.2f}' + "%", transform=axs[1, i + 1].transAxes, ha='center',
                           va='center', fontsize=6, color='green')
        axs[1, i + 1].text(0.7, 0.05, f'{loc_error_:.2f}' + "mm", transform=axs[1, i + 1].transAxes, ha='center',
                           va='center', fontsize=6, color='purple')

    # plt.savefig('multi_figures.pdf')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0)
    plt.savefig(os.path.join(path_name, "img.svg"))
    # plt.show()


if __name__ == "__main__":
    # Total
    dataset_list1 = [("Totalsegmentator_femur", 3),
                     ("Totalsegmentator_pelvic", 4),
                     ("Totalsegmentator_spine", 26),
                     ("Totalsegmentator_rib", 25),
                     ("Totalsegmentator_all", 55)]

    path_name = "./vis_result/subsets/FIG1"
    sample_name = "ct"
    subset_visualiztion(path_name, sample_name, dataset_list1, num_cor_slices=[55, 50, 45, 36, 45], num_sag_slices=[32, 40, 64, 28, 64])

    # 1K
    dataset_list2 = [("CTPelvic1K", 4)]
    path_name = "./vis_result/subsets/FIG2"
    sample_name = "1.3.6.1.4.1.9328.50.4.0510"
    subset_visualiztion(path_name, sample_name, dataset_list2, num_cor_slices=[30], num_sag_slices=[84])

    dataset_list3 = [("CTSpine1K", 26)]
    path_name = "./vis_result/subsets/FIG3"
    sample_name = "1.3.6.1.4.1.9328.50.4.0510"
    subset_visualiztion(path_name, sample_name, dataset_list3, num_cor_slices=[50], num_sag_slices=[64])

    # VERSE
    dataset_list4 = [("VERSE19", 26)]
    path_name = "./vis_result/subsets/FIG4"
    sample_name = "sub-verse230_ct"
    subset_visualiztion(path_name, sample_name, dataset_list4, num_cor_slices=[70], num_sag_slices=[64])

    # Rib
    dataset_list5 = [("RibFrac", 25)]
    path_name = "./vis_result/subsets/FIG5"
    sample_name = "RibFrac506"
    subset_visualiztion(path_name, sample_name, dataset_list5, num_cor_slices=[78], num_sag_slices=[14])
