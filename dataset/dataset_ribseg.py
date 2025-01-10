import os
import json
import argparse
import numpy as np
import torch
import nibabel as nib
from monai import transforms
import SimpleITK as sitk
import skimage
from skimage.morphology import dilation, square


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
                        default="./dataset/RibFrac",
                        type=str,
                        help='data in folder will be processed')
    parser.add_argument('--midout_file',
                        default="./dataset/RibFrac_mid",
                        type=str,
                        help='processed data output folder')
    parser.add_argument('--output_file',
                        default="./dataset/RibFrac_final",
                        type=str,
                        help='processed data output folder')
    parser.add_argument('--size',
                        type=str,
                        default="160",
                        help='pixel spacing of the output image')
    parser.add_argument('--num_aug',
                        type=str,
                        default="2",
                        help='pixel spacing of the output image')
    return parser


def identity_preprocess(args):
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


def ribseg_generator(args):
    def rank(x):
        return np.max(np.where(ct_label == x.label), 1)[-1]

    if args.input_file is not None and args.midout_file is not None:
        label_file_names = os.path.join(args.input_file, "ribfrac-seg")
        for label_name in os.listdir(label_file_names):
            if label_name.find("-seg") != -1:
                label_file = os.path.join(label_file_names, label_name)
                # SEG CT
                ct_image = nib.load(label_file)
                ct_data = np.array(ct_image.get_fdata())
                ct_data = sitk.GetImageFromArray(ct_data.astype('int8'))
                ct_data = sitk.BinaryDilate(ct_data, (2, 2, 2), sitk.sitkBall)
                ct_data = sitk.BinaryFillhole(ct_data, fullyConnected=False)
                ct_data = sitk.BinaryErode(ct_data, (2, 2, 2), sitk.sitkBall)
                ct_data = sitk.BinaryMorphologicalClosing(ct_data, (2, 2, 2), sitk.sitkBall)
                ct_data = sitk.BinaryFillhole(ct_data, fullyConnected=False)
                ct_data = sitk.GetArrayFromImage(ct_data)
                ct_data = np.logical_or(ct_data, ct_image.get_fdata())
                ct_center = np.mean(np.nonzero(ct_data), 1)

                ct_label = skimage.measure.label(ct_data, connectivity=1)
                ct_rib = skimage.measure.regionprops(ct_label)
                ct_rib.sort(key=lambda x: x.area, reverse=True)

                left = []
                right = []
                for i in ct_rib[:24]:
                    center = np.mean(np.where(ct_label == i.label), 1)
                    if center[0] < ct_center[0]:
                        if i.area > 250:
                            left.append(i)
                    else:
                        if i.area > 250:
                            right.append(i)

                left.sort(key=lambda x: x.area, reverse=True)
                left = left[:12]
                left.sort(key=rank, reverse=True)
                if len(left) != 12:
                    print("warning, left", label_file)

                right.sort(key=lambda x: x.area, reverse=True)
                right = right[:12]
                right.sort(key=rank, reverse=True)
                if len(right) != 12:
                    print("warning, right", label_file)

                save_image = np.zeros_like(ct_data)
                for i, l in enumerate(left[:12]):
                    save_image = np.where(ct_label == l.label, i + 1, save_image)
                for i, l in enumerate(right[:12]):
                    save_image = np.where(ct_label == l.label, i + 13, save_image)

                nib.save(nib.Nifti1Image(save_image.astype(np.uint8), affine=ct_image.affine), label_file)
            else:
                continue


def drr_generator(args):
    if args.drr_exe_file is not None and args.input_file is not None and args.midout_file is not None:
        train_file_names1 = os.path.join(os.path.join(args.input_file, "ribfrac-train-images"), "Part1")
        train_file_names2 = os.path.join(os.path.join(args.input_file, "ribfrac-train-images"), "Part2")
        train_file_names_ls = [os.path.join(train_file_names1, i) for i in os.listdir(train_file_names1)] + \
                              [os.path.join(train_file_names2, i) for i in os.listdir(train_file_names2)]
        train_names_ls = [i.split("\\")[-1].split("-image")[0] for i in train_file_names_ls]

        val_file_names = os.path.join(args.input_file, "ribfrac-val-images")
        val_file_names_ls = [os.path.join(val_file_names, i) for i in os.listdir(val_file_names)]
        val_names_ls = [i.split("\\")[-1].split("-image")[0] for i in val_file_names_ls]

        test_file_names = os.path.join(args.input_file, "ribfrac-test-images")
        test_file_names_ls = [os.path.join(test_file_names, i) for i in os.listdir(test_file_names)]
        test_names_ls = [i.split("\\")[-1].split("-image")[0] for i in test_file_names_ls]

        label_file_names = os.path.join(args.input_file, "ribfrac-seg")
        for label_name in os.listdir(label_file_names):
            if label_name.find("-seg") != -1:
                label_file = os.path.join(label_file_names, label_name)
                name = label_name.split("-rib-seg")[0]
                if name in train_names_ls:
                    index = train_names_ls.index(name)
                    input_file = train_file_names_ls[index]
                    output_xray_data_file_path = os.path.join(os.path.join(args.midout_file, "train"), name)
                    output_ct_data_file_path = os.path.join(os.path.join(args.midout_file, "train"), name)
                elif name in val_names_ls:
                    index = val_names_ls.index(name)
                    input_file = val_file_names_ls[index]
                    output_xray_data_file_path = os.path.join(os.path.join(args.midout_file, "val"), name)
                    output_ct_data_file_path = os.path.join(os.path.join(args.midout_file, "val"), name)
                elif name in test_names_ls:
                    index = test_names_ls.index(name)
                    input_file = test_file_names_ls[index]
                    output_xray_data_file_path = os.path.join(os.path.join(args.midout_file, "test"), name)
                    output_ct_data_file_path = os.path.join(os.path.join(args.midout_file, "test"), name)
                else:
                    raise AssertionError("Give exe file name and input file name and output file name.")

                # X-RAY
                os.makedirs(output_xray_data_file_path, exist_ok=True)
                dev_degree_list = [i * 4.5 for i in range(20)]
                rot_degree_list = [0, 90]
                for rot_deg in rot_degree_list:
                    output_xray_data_file_path_rot = os.path.join(output_xray_data_file_path,
                                                                  "coronal" if rot_deg == 0 else "sagittal")
                    os.makedirs(output_xray_data_file_path_rot, exist_ok=True)

                    for dev_deg in dev_degree_list:
                        output_xray_data = name + "_" + str(dev_deg) + "deg" + ".png"
                        output_xray_data_path = os.path.join(output_xray_data_file_path_rot,
                                                             output_xray_data)

                        command_line = args.drr_exe_file + " " + input_file + \
                                       " -o " + output_xray_data_path + \
                                       " -size " + args.size + " " + args.size + \
                                       " -rx -90 -ry " + str(rot_deg + dev_deg)
                        os.system(command_line)

                # SEG CT
                output_ct_data_path = os.path.join(output_ct_data_file_path, name + "_seg.nii.gz")
                ct_command_line = args.drr_exe_file + " " + label_file + \
                                  " -ct_o " + output_ct_data_path + \
                                  " -ct_size " + args.size + " " + args.size + " " + args.size + \
                                  " -is_seg_ct " + "1"
                os.system(ct_command_line)

            else:
                continue


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
    identity_preprocess(args)
    ribseg_generator(args)
    drr_generator(args)
    dataset_generator(args)
    dataset_augmentation(args)
