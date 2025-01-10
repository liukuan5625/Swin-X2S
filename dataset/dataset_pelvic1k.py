import os
import json
import argparse
import numpy as np
import torch
import nibabel as nib
from monai import transforms
import pickle
import shutil


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
    parser.add_argument('--raw_file',
                        default="./dataset/CTPelvic1Kraw",
                        type=str,
                        help='data in folder will be processed')
    parser.add_argument('--input_file',
                        default="./dataset/CTPelvic1K",
                        type=str,
                        help='data in folder will be processed')
    parser.add_argument('--midout_file',
                        default="./dataset/CTPelvic1K_mid",
                        type=str,
                        help='processed data output folder')
    parser.add_argument('--output_file',
                        default="./dataset/CTPelvic1K_final",
                        type=str,
                        help='processed data output folder')
    parser.add_argument('--size',
                        type=str,
                        default="160",
                        help='pixel spacing of the output image')
    parser.add_argument('--num_aug',
                        type=str,
                        default="1",
                        help='pixel spacing of the output image')
    return parser


def raw_dataset_process(args):
    """
        Convert the original dataset into the required path format
        require splits_final.pkl
    """
    if args.raw_file is not None and args.input_file is not None:
        data_name = os.listdir(os.path.join(args.raw_file, "data"))
        data_name_list = []
        for d in data_name:
            new_d = d.split('.nii.gz')[0]
            if new_d.find('dataset') != -1:
                new_d = ''.join(new_d.split('_')[1:])
            new_d = ''.join(new_d.split('.'))
            new_d = ''.join(new_d.split('_'))
            new_d = ''.join(new_d.split('-'))
            new_d = new_d.split('data')[0]
            data_name_list.append(new_d)

        label_name = os.listdir(os.path.join(args.raw_file, "label"))
        label_name_list = []
        for l in label_name:
            new_l = l.split('_mask')[0]
            if new_l.find('dataset') != -1:
                new_l = ''.join(new_l.split('_')[1:])
            new_l = new_l[:26]
            new_l = ''.join(new_l.split('.'))
            new_l = ''.join(new_l.split('_'))
            new_l = ''.join(new_l.split('-'))
            label_name_list.append(new_l)

        raw_pkl_path = os.path.join(args.raw_file, "splits_final.pkl")
        with open(raw_pkl_path, 'rb') as f:
            raw_name = pickle.load(f)[0]

        data_path = os.path.join(args.input_file, "data")
        train_data_path = os.path.join(data_path, "train")
        val_data_path = os.path.join(data_path, "val")
        test_data_path = os.path.join(data_path, "test")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(train_data_path, exist_ok=True)
        os.makedirs(val_data_path, exist_ok=True)
        os.makedirs(test_data_path, exist_ok=True)

        label_path = os.path.join(args.input_file, "label")
        train_label_path = os.path.join(label_path, "train")
        val_label_path = os.path.join(label_path, "val")
        test_label_path = os.path.join(label_path, "test")
        os.makedirs(label_path, exist_ok=True)
        os.makedirs(train_label_path, exist_ok=True)
        os.makedirs(val_label_path, exist_ok=True)
        os.makedirs(test_label_path, exist_ok=True)

        for split_key in raw_name.keys():
            split_file = raw_name[split_key]
            for file in split_file:
                if file.find('dataset') != -1:
                    if file.find('.') == -1:
                        continue

                save_name = ''.join(file.split('_')[2:])[:26]
                new_save_name = ''.join(save_name.split('.'))
                new_save_name = ''.join(new_save_name.split('_'))
                new_save_name = ''.join(new_save_name.split('-'))

                raw_ct_name = data_name[data_name_list.index(new_save_name)]
                raw_ct_file_path = os.path.join(os.path.join(args.raw_file, 'data'), raw_ct_name)
                raw_ct_save_path = os.path.join(os.path.join(data_path, split_key), save_name + '.nii.gz')
                shutil.copy(raw_ct_file_path, raw_ct_save_path)

                seg_ct_name = label_name[label_name_list.index(new_save_name)]
                seg_ct_file_path = os.path.join(os.path.join(args.raw_file, 'label'), seg_ct_name)
                seg_ct_save_path = os.path.join(os.path.join(label_path, split_key), save_name + '_seg.nii.gz')
                seg_ct = transforms.LoadImage()(seg_ct_file_path)
                affine_mat = seg_ct.meta['original_affine']
                seg_ct = np.where(seg_ct > 3, 0, seg_ct)
                nib.save(nib.Nifti1Image(seg_ct.astype(np.uint8), affine=affine_mat), seg_ct_save_path)

    else:
        raise AssertionError("Give exe file name and input file name and output file name.")


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

        input_file_names = os.path.join(args.input_file, "data")
        label_file_names = os.path.join(args.input_file, "label")

        for input_split, label_split in zip(os.listdir(input_file_names), os.listdir(label_file_names)):
            input_file_name = os.path.join(input_file_names, input_split)
            label_file_name = os.path.join(label_file_names, label_split)

            data_f = os.listdir(input_file_name)
            data_f.sort(key=lambda x: sum(ord(c) for c in x))
            label_f = os.listdir(label_file_name)
            label_f.sort(key=lambda x: sum(ord(c) for c in x))
            for data, label in zip(data_f, label_f):
                input_file = os.path.join(input_file_name, data)
                label_file = os.path.join(label_file_name, label)

                # XRAY DRR
                output_xray_data_file = data.split('.nii.gz')[0]
                output_xray_data_file_path = os.path.join(os.path.join(args.midout_file, input_split),
                                                          output_xray_data_file)
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

                        command_line = args.drr_exe_file + " " + input_file + \
                                       " -o " + output_xray_data_path + \
                                       " -size " + args.size + " " + args.size + \
                                       " -rx -90 -ry " + str(rot_deg + dev_deg)
                        os.system(command_line)

                # SPINE SEG CT
                output_ct_data_file = label.split('_seg.nii.gz')[0]
                output_ct_data_file_path = os.path.join(os.path.join(args.midout_file, label_split),
                                                        output_ct_data_file)
                if not output_ct_data_file_path: raise AssertionError("Wrong ct file.")

                output_ct_data_path = os.path.join(output_ct_data_file_path, data)
                ct_command_line = args.drr_exe_file + " " + label_file + \
                                  " -ct_o " + output_ct_data_path + \
                                  " -ct_size " + args.size + " " + args.size + " " + args.size + \
                                  " -is_seg_ct " + "1"
                os.system(ct_command_line)

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

    split_index = []
    split_file_name = os.listdir(args.midout_file)
    for split_file in split_file_name:
        split_file_path = os.path.join(args.midout_file, split_file)

        for data_file in os.listdir(split_file_path):
            data_file_path = os.path.join(split_file_path, data_file)

            file_path = os.listdir(data_file_path)
            file_path.sort(key=lambda x: len(x))
            file_path = [file_path[2], file_path[0], file_path[1]]
            xray2_list = []
            for file in file_path:
                file_name = os.path.join(data_file_path, file)

                # CT DATASET (LABEL)
                if file.endswith("nii.gz"):
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
                        transforms.ToNumpy()
                    ])
                    ct_seg_list.append(seg_tran(seg_ct)[0])
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
                            transforms.Resize([160, 160], mode='bicubic'),
                        ])
                        xray_data = xray_tran(xray_data)[0]

                        xray_crops = []
                        for i in range(1):
                            xray_crops.append(xray_data)
                        xray_datas.append(np.stack(xray_crops))
                    xray2_list.append(np.stack(xray_datas).transpose(1, 0, 2, 3))
            if len(xray2_list) != 0:
                xray_list.append(np.stack(xray2_list).transpose(1, 0, 2, 3, 4))
        split_index.append([split_file_path, len(ct_seg_list)])
    xray_list = np.concatenate(xray_list, 0).tolist()

    ###
    training_list = []
    validation_list = []
    test_list = []
    for index, (image, label) in enumerate(zip(xray_list, ct_seg_list)):
        name_list = ct_seg_name_list[index]

        image_save_dir = name_list[-2] + '_image' + '.npy'
        np.save(os.path.join(args.output_file, image_save_dir), image)

        affine_mat = ct_seg_affine_list[index]
        label_save_dir = name_list[-2] + '_label' + '.nii.gz'
        nib.save(nib.Nifti1Image(label.astype(np.uint8), affine=affine_mat),
                 os.path.join(args.output_file, label_save_dir))

        dic = {'img': image_save_dir, 'seg': label_save_dir}
        if name_list[-3].find('train') != -1:
            training_list.append(dic)
        elif name_list[-3].find('val') != -1:
            validation_list.append(dic)
        elif name_list[-3].find('test') != -1:
            test_list.append(dic)
        else:
            print(name_list)
            assert 1 == 0

    data_dic = {"training": training_list, "validation": validation_list, "test": test_list}
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

    raw_dataset_process(args)
    identity_preprocess(args)
    drr_generator(args)
    dataset_generator(args)
    dataset_augmentation(args)
