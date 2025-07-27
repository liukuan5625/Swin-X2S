import math
import random
import sys

import numpy as np
import torch
import torch.utils.data
from monai import transforms
from monai.data import Dataset, load_decathlon_datalist
from torch.utils.data import DataLoader


def x2s_transform(
        input,
        args,
        param_dict
):
    mirror_prob = param_dict['mirror_params']
    translate_prob, center_shift = param_dict['translate_params']
    cor_idx, sag_idx, cor_rot_idx, sag_rot_idx, rot_angle = param_dict['rotation_params']

    def img_trans(img):
        cor_xray_data = img[:, :, cor_rot_idx].unsqueeze(0)
        sag_xray_data = img[:, :, sag_rot_idx].unsqueeze(0)

        xray_data = torch.cat([cor_xray_data, sag_xray_data])
        xray_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(spatial_size=[args.roi_z, args.roi_z], mode='bilinear'),
        ])
        xray_data = xray_trans(xray_data)

        # Intensity scale augmentation
        intensity_trans = transforms.Compose([
            transforms.RandScaleIntensity(factors=0.1, prob=0.25),
            transforms.RandShiftIntensity(offsets=0.1, prob=0.25),
        ])
        xray_data = intensity_trans(xray_data)

        # Random shift augmentation
        def img_shift(raw_img):
            cor = raw_img[:args.in_channels]
            sag = raw_img[args.in_channels:]
            cor_shape = [args.roi_z, args.roi_z]
            sag_shape = [args.roi_z, args.roi_z]
            crop_cor = torch.zeros([args.in_channels, args.roi_x, args.roi_z])
            crop_sag = torch.zeros([args.in_channels, args.roi_y, args.roi_z])
            deg_list = [90 * i / args.in_channels for i in range(args.in_channels)]
            for i, deg in enumerate(deg_list):
                cor_shift_center = [math.cos(deg * math.pi / 180) * center_shift[0] +
                                    math.sin(deg * math.pi / 180) * center_shift[1] + cor_shape[-2] // 2,
                                    center_shift[2] + cor_shape[-1] // 2]
                sag_shift_center = [math.cos((deg + 90) * math.pi / 180) * center_shift[0] +
                                    math.sin((deg + 90) * math.pi / 180) * center_shift[1] + sag_shape[-2] // 2,
                                    center_shift[2] + sag_shape[-1] // 2]
                crop_cor[i: i + 1] = transforms.Compose([
                    transforms.SpatialCrop(roi_center=cor_shift_center, roi_size=[args.roi_x, args.roi_z]),
                    transforms.SpatialPad(spatial_size=[args.roi_x, args.roi_z]),
                ])(cor[i: i + 1])
                crop_sag[i: i + 1] = transforms.Compose([
                    transforms.SpatialCrop(roi_center=sag_shift_center, roi_size=[args.roi_y, args.roi_z]),
                    transforms.SpatialPad(spatial_size=[args.roi_y, args.roi_z]),
                ])(sag[i: i + 1])
            return torch.cat([crop_cor, crop_sag])
        if translate_prob < args.translate_prob:
            xray_data = img_shift(xray_data)

        # Random flip augmentation
        if mirror_prob < args.mirror_prob:
            xray_data = torch.flip(xray_data, [-2])
        cor_xray_data = xray_data[:args.in_channels]
        sag_xray_data = xray_data[args.in_channels:]

        if args.spatial_dims == 3:
            cor_xray_data = transforms.RepeatChannel(args.roi_y)(cor_xray_data.unsqueeze(0))
            cor_xray_data = torch.permute(cor_xray_data, [1, 2, 0, 3])
            sag_xray_data = transforms.RepeatChannel(args.roi_x)(sag_xray_data.unsqueeze(0))
            sag_xray_data = torch.permute(sag_xray_data, [1, 0, 2, 3])
        return [cor_xray_data, sag_xray_data]

    def seg_trans(seg):
        ct_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.EnsureChannelFirst(channel_dim='no_channel'),
            transforms.Resize(spatial_size=[args.roi_z, args.roi_z, args.roi_z], mode='nearest'),
            transforms.Rotate(angle=[0, 0, rot_angle], mode='nearest', padding_mode='zeros')
        ])
        seg = ct_transform(seg)

        seg_shape = [args.roi_z, args.roi_z, args.roi_z]
        # Random shift augmentation
        if translate_prob < args.translate_prob:
            seg_shift = transforms.Compose([
                transforms.SpatialCrop(roi_center=[seg_shape[-3] // 2 + center_shift[0],
                                                   seg_shape[-2] // 2 + center_shift[1],
                                                   seg_shape[-1] // 2 + center_shift[2]],
                                       roi_size=[args.roi_x, args.roi_y, args.roi_z]),
                transforms.SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            ])
            seg = seg_shift(seg)

        # Random flip augmentation
        if mirror_prob < args.mirror_prob:
            seg = torch.flip(seg, [-3, -2])
            # seg = transforms.Flip([-3, -2])(seg)
        return seg

    _transform = transforms.Compose([
        transforms.LoadImaged(keys=["img", "seg"]),
        transforms.Lambdad(keys="img", func=img_trans),
        transforms.Lambdad(keys="seg", func=seg_trans),
    ])
    return _transform(input)


def val_x2s_transform(
        input,
        args,
        param_dict
):
    cor_idx, sag_idx, cor_rot_idx, sag_rot_idx, rot_angle = param_dict['rotation_params']

    def val_img_trans(img):
        assert args.roi_x == args.roi_y
        img_shape = [args.roi_z, args.roi_z]
        xray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(spatial_size=img_shape, mode='bilinear'),
            transforms.SpatialCrop(roi_center=[img_shape[-2] // 2, img_shape[-1] // 2],
                                   roi_size=[args.roi_y, args.roi_z]),
        ])
        cor_xray_data = img[:, :, cor_idx].unsqueeze(0)
        sag_xray_data = img[:, :, sag_idx].unsqueeze(0)
        cor_xray_data = xray_transform(cor_xray_data)
        sag_xray_data = xray_transform(sag_xray_data)

        if args.spatial_dims == 3:
            cor_xray_data = transforms.RepeatChannel(args.roi_y)(cor_xray_data.unsqueeze(0))
            cor_xray_data = torch.permute(cor_xray_data, [1, 2, 0, 3])
            sag_xray_data = transforms.RepeatChannel(args.roi_x)(sag_xray_data.unsqueeze(0))
            sag_xray_data = torch.permute(sag_xray_data, [1, 0, 2, 3])
        return [cor_xray_data, sag_xray_data]

    def val_seg_trans(seg):
        seg_shape = [args.roi_z, args.roi_z, args.roi_z]
        ct_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.EnsureChannelFirst(channel_dim='no_channel'),
            transforms.Resize(spatial_size=seg_shape, mode='nearest'),
            transforms.SpatialCrop(roi_center=[seg_shape[-3] // 2, seg_shape[-2] // 2, seg_shape[-1] // 2],
                                   roi_size=[args.roi_x, args.roi_y, args.roi_z]),
        ])
        seg = ct_transform(seg)
        return seg

    _val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["img", "seg"]),
        transforms.Lambdad(keys="img", func=val_img_trans),
        transforms.Lambdad(keys="seg", func=val_seg_trans),
    ])
    return _val_transform(input)


def get_cbct_loader(args):
    def cor_sag_find(item, args):
        name = item['img'].split('\\')[-1].split('-')
        start_angle = -1 * float(name[3][1:])
        proj_num = int(name[4][1:])
        device = name[5]
        angle_list = np.loadtxt('./utils/' + device + '-angle' + str(proj_num) + '.txt')
        cor_idx = np.argmin(np.abs(angle_list + start_angle))
        sag_idx = np.argmin(np.abs(angle_list + start_angle + np.pi / 2))
        rotation_prob = torch.rand([1])
        if rotation_prob < args.rotation_prob:
            sag_rot_idx = np.random.randint(0, min(cor_idx - sag_idx, proj_num - cor_idx + sag_idx))  # not 2pi
            cor_rot_idx = np.argmin(np.abs(angle_list - angle_list[sag_rot_idx] - np.pi / 2))
            cor_rot_angle = angle_list[cor_rot_idx] - angle_list[cor_idx]
            sag_rot_angle = angle_list[sag_rot_idx] - angle_list[sag_idx]
            rot_angle = -(cor_rot_angle + sag_rot_angle) / 2
            return cor_idx, sag_idx, cor_rot_idx, sag_rot_idx, rot_angle
        else:
            return cor_idx, sag_idx, cor_idx, sag_idx, 0

    class CustomTransformDataset(Dataset):
        def __init__(self, data, transform=None):
            super().__init__(data, transform)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            center_shift = [int(torch.randn([1]) * args.roi_x) / 10,
                            int(torch.randn([1]) * args.roi_y) / 10,
                            int(torch.randn([1]) * args.roi_z) / 10]
            mirror_prob = torch.rand([1])
            translate_prob = torch.rand([1])
            cor_idx, sag_idx, cor_rot_idx, sag_rot_idx, rot_angle = cor_sag_find(item, args)
            param_dict = {'mirror_params': mirror_prob, 'translate_params': (translate_prob, center_shift),
                          'rotation_params': (cor_idx, sag_idx, cor_rot_idx, sag_rot_idx, rot_angle)}
            return self.transform(item, args, param_dict)

    if args.test_mode:
        test_data_list = load_decathlon_datalist(args.data_dir, is_segmentation=True, data_list_key="test")
        test_ds = CustomTransformDataset(data=test_data_list, transform=val_x2s_transform)
        test_loader = DataLoader(dataset=test_ds,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0 if sys.platform.startswith('win') else args.workers,
                                 pin_memory=False,
                                 persistent_workers=False if sys.platform.startswith('win') else True, )
        return test_loader

    else:
        train_data_list = load_decathlon_datalist(args.data_dir, is_segmentation=True, data_list_key="training")
        train_ds = CustomTransformDataset(data=train_data_list, transform=x2s_transform)
        train_loader = DataLoader(dataset=train_ds,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0 if sys.platform.startswith('win') else args.workers,
                                  pin_memory=False,
                                  persistent_workers=False if sys.platform.startswith('win') else True,
                                  drop_last=False)
        val_data_list = load_decathlon_datalist(args.data_dir, is_segmentation=True, data_list_key="validation")
        val_ds = CustomTransformDataset(data=val_data_list, transform=val_x2s_transform)
        val_loader = DataLoader(dataset=val_ds,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=0 if sys.platform.startswith('win') else args.workers,
                                pin_memory=False,
                                persistent_workers=False if sys.platform.startswith('win') else True,
                                drop_last=False)
        return train_loader, val_loader
