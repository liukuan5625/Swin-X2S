import os
import json
import argparse
import numpy as np
import torch
import nibabel as nib
from monai import transforms
from scipy.ndimage import distance_transform_edt as distance


def get_parser():
    parser = argparse.ArgumentParser(description="Digitally Reconstructed Radiograph Description")
    parser.add_argument('--id_exe_file',
                        default="IdentityDirection.exe",
                        type=str,
                        help='Identity direction exe')
    parser.add_argument('--drr_exe_file',
                        default="DigitallyReconstructedRadiograph.exe",
                        type=str,
                        help='digitally reconstructed radiograph exe')
    parser.add_argument('--input_file',
                        default="./dataset/CTSpine1k",
                        type=str,
                        help='data in folder will be processed')
    parser.add_argument('--midout_file',
                        default="./dataset/CTSpine1k_mid",
                        type=str,
                        help='processed data output folder')
    parser.add_argument('--output_file',
                        default="./dataset/CTSpine1k_final",
                        type=str,
                        help='processed data output folder')
    parser.add_argument('--size',
                        type=str,
                        default="160",
                        help='pixel spacing of the output image')
    parser.add_argument('--num_aug',
                        type=str,
                        default="5",
                        help='pixel spacing of the output image')
    return parser


def identity_preprocess(args):
    """
    To make every CT to the same affine matrix for drr generation
    """

    if args.id_exe_file is not None and args.input_file is not None:
        input_file_name = os.path.join(args.input_file, "data")
        label_file_name = os.path.join(args.input_file, "label")

        for input_file, label_file in zip(os.listdir(input_file_name), os.listdir(label_file_name)):
            input_file = os.path.join(input_file_name, input_file)
            label_file = os.path.join(label_file_name, label_file)

            file_name = []
            if os.path.isdir(input_file):
                for d in os.listdir(input_file):
                    if d.endswith("nii.gz"):
                        file_name.append("".join(d.split(".")[:-2]))
                        input_xray_data_path = os.path.join(input_file, d)

                        # RAW CT
                        ct_command_line = args.id_exe_file + " " + input_xray_data_path
                        os.system(ct_command_line)

            # SPINE SEG CT
            if os.path.isdir(label_file):
                for l in os.listdir(label_file):
                    if l.endswith("nii.gz"):
                        input_ct_data_path = os.path.join(label_file, l)

                        ct_command_line = args.id_exe_file + " " + input_ct_data_path
                        os.system(ct_command_line)

    else:
        raise AssertionError("Give exe file name and input file name and output file name.")


def drr_generator(args):
    if args.drr_exe_file is not None and args.input_file is not None and args.midout_file is not None:

        input_file_name = os.path.join(args.input_file, "data")
        label_file_name = os.path.join(args.input_file, "label")

        for input_file, label_file in zip(os.listdir(input_file_name), os.listdir(label_file_name)):
            input_file = os.path.join(input_file_name, input_file)
            label_file = os.path.join(label_file_name, label_file)

            file_name = []
            if os.path.isdir(input_file):
                data_f = os.listdir(input_file)
                if '.DS_Store' in data_f:
                    data_f.remove('.DS_Store')

                data_f.sort(key=lambda x: sum(ord(c) for c in x))
                for d in data_f:
                    split_id = d.rfind(".nii.gz")
                    if split_id != -1:
                        file_name.append(d[:split_id])

                        input_xray_data_path = os.path.join(input_file, d)
                        output_xray_data_file = file_name[-1]

                        output_xray_data_file_path = os.path.join(args.midout_file, output_xray_data_file)
                        os.makedirs(output_xray_data_file_path, exist_ok=True)

                        dev_degree_list = [i * 4.5 for i in range(20)]
                        rot_degree_list = [0, 90]
                        for rot_deg in rot_degree_list:
                            output_xray_data_file_path_rot = os.path.join(output_xray_data_file_path,
                                                                          "coronal" if rot_deg == 0 else "sagittal")
                            os.makedirs(output_xray_data_file_path_rot, exist_ok=True)

                            for dev_deg in dev_degree_list:
                                output_xray_data = output_xray_data_file + "_" + str(dev_deg) + "deg" + ".png"
                                output_xray_data_path = os.path.join(output_xray_data_file_path_rot,
                                                                     output_xray_data)

                                command_line = args.drr_exe_file + " " + input_xray_data_path + \
                                               " -o " + output_xray_data_path + \
                                               " -size " + args.size + " " + args.size + \
                                               " -rx -90 -ry " + str(rot_deg + dev_deg)
                                os.system(command_line)

            # SPINE SEG CT
            index = 0
            if os.path.isdir(label_file):
                label_f = os.listdir(label_file)
                if '.DS_Store' in label_f:
                    label_f.remove('.DS_Store')

                label_f.sort(key=lambda x: sum(ord(c) for c in x))
                for l in label_f:
                    if l.endswith("nii.gz"):
                        input_ct_data_path = os.path.join(label_file, l)
                        output_ct_data_file = file_name[index]
                        output_ct_data_file_path = os.path.join(args.midout_file, output_ct_data_file)
                        if not os.path.exists(output_ct_data_file_path): raise AssertionError("Wrong ct file.")

                        output_ct_data = l
                        output_ct_data_path = os.path.join(output_ct_data_file_path, output_ct_data)

                        ct_command_line = args.drr_exe_file + " " + input_ct_data_path + \
                                          " -ct_o " + output_ct_data_path + \
                                          " -ct_size " + args.size + " " + args.size + " " + args.size + \
                                          " -is_seg_ct " + "1"
                        os.system(ct_command_line)
                        index += 1

    else:
        raise AssertionError("Give exe file name and input file name and output file name.")


def dataset_generator(args):
    assert args.midout_file is not None
    assert args.output_file is not None
    os.makedirs(args.output_file, exist_ok=True)

    xray_list = []
    xray_transforms = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.ScaleIntensityRange(a_min=0, a_max=255, b_min=0, b_max=1),
        transforms.ToNumpy()
    ])

    ct_seg_list = []
    ct_seg_name_list = []
    ct_seg_affine_list = []

    ct_transforms = transforms.Compose([
        transforms.LoadImage(image_only=False),
        transforms.ToNumpy()
    ])

    for data_file in os.listdir(args.midout_file):
        data_file_path = os.path.join(args.midout_file, data_file)

        file_path = os.listdir(data_file_path)
        file_path.sort(key=lambda x: len(x))
        xray2_list = []
        for file in file_path:
            file_name = os.path.join(data_file_path, file)

            # CT DATASET (LABEL)
            if file.endswith("nii.gz"):
                if file.find("seg") != -1:
                    seg_ct = ct_transforms(file_name)
                    seg_ct_meta = seg_ct[-1]
                    seg_ct = seg_ct[0]
                    seg_ct = np.flip(seg_ct, axis=[-1])
                    seg_name = seg_ct_meta['filename_or_obj']
                    seg_affine = seg_ct_meta["affine"]

                    seg_tran = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.EnsureChannelFirst(channel_dim='no_channel'),
                        transforms.SpatialPad(spatial_size=(seg_ct.shape[-1], seg_ct.shape[-1], seg_ct.shape[-1])),
                        transforms.Resize([160, 160, 160], mode='nearest'),
                        # transforms.CenterSpatialCrop([128, 128, 160]),
                        transforms.ToNumpy()
                    ])
                    seg_affine[:3, :3] = seg_ct.shape[-1] / 160 * seg_affine[:3, :3]
                    ct_seg_list.append(seg_tran(seg_ct)[0].astype(np.uint8))
                    ct_seg_name_list.append(seg_name.split('\\'))
                    ct_seg_affine_list.append(seg_affine)

            # XRAY DATASET (DATA)
            else:
                assert os.path.isdir(file_name)
                xray_datas = []

                xray_list_name = os.listdir(file_name)
                xray_list_name.sort(key=lambda x: float((x.split("_")[-1]).split("deg")[0]))

                for xray_file in xray_list_name:
                    xray_path = os.path.join(file_name, xray_file)
                    xray_data = xray_transforms(xray_path)

                    xray_tran = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.EnsureChannelFirst(channel_dim='no_channel'),
                        transforms.SpatialPad(spatial_size=(xray_data.shape[-1], xray_data.shape[-1])),
                        transforms.Resize([160, 160], mode='bilinear'),
                        # transforms.CenterSpatialCrop([128, 160]),
                    ])
                    xray_data = xray_tran(xray_data)[0]

                    xray_crops = []
                    for i in range(1):
                        xray_crops.append(xray_data)
                    xray_datas.append(np.stack(xray_crops))
                xray2_list.append(np.stack(xray_datas).transpose(1, 0, 2, 3))
        xray_list.append(np.stack(xray2_list).transpose(1, 0, 2, 3, 4))
    xray_list = np.concatenate(xray_list, 0).tolist()

    ###
    xray_process_list = []
    ct_process_seg_list = []
    ct_process_seg_name_list = []
    ct_process_seg_affine_list = []

    for index, label in enumerate(ct_seg_list):
        idxs = np.unique(label)[1:]
        idxs.sort()

        # 过滤异常类别
        if len(idxs) != 0:
            if idxs[-1] > 25:
                # assert 1 == 0
                continue

        ct_process_seg_list.append(label.astype(np.uint8))
        xray_process_list.append(xray_list[index])
        ct_process_seg_name_list.append(ct_seg_name_list[index])
        ct_process_seg_affine_list.append(ct_seg_affine_list[index])

    del xray_list
    del ct_seg_list
    ct_process_seg_list = np.stack(ct_process_seg_list)
    xray_process_list = np.stack(xray_process_list)

    ###
    list = []
    with open(os.path.join(args.input_file, "data_split.txt"), 'r') as file:
        # 按行读取
        for line in file:
            line_d = line.strip()
            if len(line_d) != 0:
                list.append(line_d.split('.nii.gz')[0])
    train_idx = list.index('trainset:')
    val_idx = list.index('test_public:')
    test_idx = list.index('test_private:')
    split_dic = {'training': list[train_idx + 1: val_idx],
                 'validation': list[val_idx + 1: test_idx],
                 'test': list[test_idx + 1:]}

    data_dic = {"training": [], "validation": [], "test": []}
    for index, (image, label) in enumerate(zip(xray_process_list, ct_process_seg_list)):
        name_list = ct_process_seg_name_list[index]

        image_save_dir = name_list[-2] + '_image' + '.npy'
        np.save(os.path.join(args.output_file, image_save_dir), image)

        affine_mat = ct_process_seg_affine_list[index]
        label_save_dir = name_list[-2] + '_label' + '.nii.gz'
        nib.save(nib.Nifti1Image(label.astype(np.uint8), affine=affine_mat),
                 os.path.join(args.output_file, label_save_dir))

        dic = {'img': image_save_dir, 'seg': label_save_dir}
        if name_list[-2] in split_dic['training']:
            data_dic['training'].append(dic)
        elif name_list[-2] in split_dic['validation']:
            data_dic['validation'].append(dic)
        elif name_list[-2] in split_dic['test']:
            data_dic['test'].append(dic)
        else:
            print(dic)
            assert 1 == 0

    with open(os.path.join(args.output_file, 'data_list.json'), 'w') as file:
        json.dump(data_dic, file)


def dataset_augmentation(args):
    with open(os.path.join(args.output_file, 'data_list.json'), 'r', encoding='utf-8') as f:
        data_dic = json.load(f)

    new_training_list = []
    for dic in data_dic['training']:
        img_dir = os.path.join(args.output_file, dic['img'])
        seg_dir = os.path.join(args.output_file, dic['seg'])
        new_training_list.append(dic)

        for i in range(int(args.num_aug)):
            zoom_scale = torch.rand([1]) + 0.5
            rotate_angle = torch.randn([1]) * 5 * torch.pi / 180

            img_data = transforms.LoadImage()(img_dir)
            img_data = img_data.reshape(-1, int(args.size), int(args.size))
            img_trans = transforms.Compose([
                # Random zoom augmentation
                transforms.Zoom(zoom=float(zoom_scale), mode='bicubic', padding_mode='constant'),
                # Random rotate augmentation
                transforms.Rotate(angle=float(rotate_angle), mode='bicubic', padding_mode='zeros'),
            ])
            aug_img_data = img_trans(img_data).reshape([2, -1, int(args.size), int(args.size)])
            aug_img_dir = dic['img'].split('.npy')[0] + '_aug' + str(i) + '.npy'
            np.save(os.path.join(args.output_file, aug_img_dir), aug_img_data.numpy())

            seg_data = transforms.LoadImage()(seg_dir)
            seg_trans = transforms.Compose([
                transforms.EnsureChannelFirst(channel_dim='no_channel'),
                # Random zoom augmentation
                transforms.Zoom(zoom=[float(zoom_scale), float(zoom_scale), float(zoom_scale)],
                                mode='nearest', padding_mode='constant'),
                # Random rotate augmentation
                transforms.Rotate(angle=[float(rotate_angle), -float(rotate_angle), 0],
                                  mode='nearest', padding_mode='zeros'),
            ])
            aug_seg_data = seg_trans(seg_data)[0]
            aug_seg_dir = dic['seg'].split('.nii.gz')[0] + '_aug' + str(i) + '.nii.gz'
            aff = aug_seg_data.meta['original_affine']
            aff[:3, :3] = aff[:3, :3] * float(zoom_scale)
            nib.save(nib.Nifti1Image(aug_seg_data.numpy().astype(np.uint8), affine=aff),
                     os.path.join(args.output_file, aug_seg_dir))
            new_training_list.append({'img': aug_img_dir, 'seg': aug_seg_dir})

    data_dic = {"training": new_training_list, "validation": data_dic['validation'], "test": data_dic['test']}
    with open(os.path.join(args.output_file, 'data_list_.json'), 'w') as file:
        json.dump(data_dic, file)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    identity_preprocess(args)
    drr_generator(args)
    dataset_generator(args)
    dataset_augmentation(args)
