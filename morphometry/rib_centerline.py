import argparse
import os
import numpy as np
import json
import SimpleITK as sitk
import skimage
from pathlib import Path
import kimimaro
from torch import tensor
from scipy.interpolate import UnivariateSpline
from xrayto3d_morphometry import (
    find_furthest_pt,
    get_array_and_volume,
    get_oriented_camera,
    clDice,
    Normalized_Line_Dice,
    Line_Seg_Chamfer_DistanceError,
)
import vedo
from monai.metrics import DiceMetric


def skel_extraction(vol):
    skels = kimimaro.skeletonize(
        vol.astype(np.int16),
        teasar_params={
            'scale': 4,
            'const': 500,  # physical units
            'pdrf_exponent': 4,
            'pdrf_scale': 100000,
            'soma_detection_threshold': 1100,  # physical units
            'soma_acceptance_threshold': 3500,  # physical units
            'soma_invalidation_scale': 1.0,
            'soma_invalidation_const': 300,  # physical units
            'max_paths': 50,  # default None
        },
        # object_ids=[25024949], # process only the specified labels
        # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
        # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
        dust_threshold=10,  # skip connected components with fewer than this many voxels
        anisotropy=(1, 1, 1),  # default True
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        fill_holes=False,  # default False
        fix_avocados=False,  # default False
        progress=False,  # default False, show progress bar
        parallel=1,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=10,  # how many skeletons to process before updating progress bar
    )
    return skels


def smooth_3d_array(points, num=None, **kwargs):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    points = np.zeros((num, 3))
    if num is None:
        num = len(x)
    w = np.arange(0, len(x), 1)
    sx = UnivariateSpline(w, x, **kwargs)
    sy = UnivariateSpline(w, y, **kwargs)
    sz = UnivariateSpline(w, z, **kwargs)
    wnew = np.linspace(0, len(x), num)
    points[:, 0] = sx(wnew)
    points[:, 1] = sy(wnew)
    points[:, 2] = sz(wnew)
    return points


def dilation(vol_ct, vol_rib):
    '''
    input:
        vol_ct: raw ct volume
        vol_rib: predicted rib volume
    output:
        res:  delated rib volume's largest component
    '''
    img_array = sitk.GetImageFromArray(vol_rib.astype(np.int8))
    img_dilated = sitk.BinaryDilate(img_array, (5, 5, 5))
    mask_dilated = sitk.GetArrayFromImage(img_dilated)
    vol_rib_dilated = np.multiply(vol_ct, mask_dilated).astype(np.int8)

    vol_tmp = skimage.measure.label(vol_rib_dilated, connectivity=1)
    vol_region = skimage.measure.regionprops(vol_tmp)
    vol_region.sort(key=lambda x: x.area, reverse=True)
    res = np.in1d(vol_tmp, [x.label for x in vol_region[:1]]).reshape(vol_rib_dilated.shape)
    return res


def get_centerline(np_array):
    np_array_tmp = np_array.copy()
    rib_array = dilation(np_array, np_array_tmp).astype(np.int8)
    try:
        tmp_cl = skel_extraction(rib_array)[1]
        cl_skl = tmp_cl
        seed = find_furthest_pt(tmp_cl, 0, single=False)[0]
        longest_path = find_furthest_pt(tmp_cl, seed, single=False)[1][0]
        org = tmp_cl.vertices[longest_path]
        aug = smooth_3d_array(org, num=500, s=2000)
        cl_org = org
        cl_smooth = aug
    except:
        return None, None, None
        # print("has no", i)
    return cl_skl, cl_org, cl_smooth


def main(nifti_file,
         out_file,
         is_show=False,
         screenshot=False,
         screenshot_out_dir="./screenshots",
         ):
    pred_files = []
    label_files = []
    files = os.listdir(nifti_file)
    for file in files:
        if file.find("_label") != -1:
            name = file.split("_label")[0]
            pred_files.append(name + "_pred.nii.gz")
            label_files.append(name + "_label.nii.gz")

    pred_files.sort(key=lambda x: sum(ord(c) for c in x))
    label_files.sort(key=lambda x: sum(ord(c) for c in x))
    dice_func = DiceMetric()
    metrics_dicts = {}
    for pred, label in zip(pred_files, label_files):
        diff_list = {}

        flag = False
        pred_vol_list = []
        pred_cl_smooth_list = []
        label_vol_list = []
        label_cl_smooth_list = []
        for i in range(26, 50):
            pred_array, pred_vol = get_array_and_volume(
                os.path.join(nifti_file, pred), label_idx=i, reorient=False
            )
            if np.sum(pred_array) == 0:
                continue
            pred_cl, pred_cl_org, pred_cl_smooth = get_centerline(pred_array)
            if pred_cl is None:
                continue
            pred_vol_list.append(pred_vol)
            pred_cl_smooth_list.append(pred_cl_smooth)

            label_array, label_vol = get_array_and_volume(
                os.path.join(nifti_file, label), label_idx=i, reorient=False
            )
            if np.sum(label_array) == 0:
                continue
            label_cl, label_cl_org, label_cl_smooth = get_centerline(label_array)
            if label_cl is None:
                continue
            label_vol_list.append(label_vol)
            label_cl_smooth_list.append(label_cl_smooth)

            diff_dict = {}
            diff_dict["Dice"] = float(dice_func(tensor(np.expand_dims(pred_array, [0, 1])),
                                                tensor(np.expand_dims(label_array, [0, 1])))[0, 0])
            diff_dict["clDice"] = clDice(pred_array, label_array)
            diff_dict["nlDice"] = Normalized_Line_Dice(pred_cl_smooth, label_cl_smooth)
            diff_dict["LSCDError"] = Line_Seg_Chamfer_DistanceError(pred_array, pred_cl_smooth,
                                                                    label_array, label_cl_smooth)
            # print(diff_dict)
            diff_list[str(i)] = diff_dict
            flag = True

        if is_show:
            if not flag:
                continue
            label_mesh_obj = vedo.Mesh()
            for l in label_vol_list:
                label_mesh_obj += l.isosurface(value=0.9, flying_edges=True)
            label_mesh_obj = vedo.merge(label_mesh_obj.unpack())
            label_points = vedo.Points(np.concatenate(label_cl_smooth_list, 0), r=5, c='yellow')
            pred_mesh_obj = vedo.Mesh()
            for p in pred_vol_list:
                pred_mesh_obj += p.isosurface(value=0.9, flying_edges=True)
            pred_mesh_obj = vedo.merge(pred_mesh_obj.unpack())
            pred_points = vedo.Points(np.concatenate(pred_cl_smooth_list, 0), r=5, c='red')

            # 渲染所有线段
            plotter = vedo.Plotter()
            plotter.clear()
            cam = get_oriented_camera(pred_mesh_obj, axis=2, camera_dist=400)
            plotter.show(
                label_mesh_obj.c("green", 0.5),
                pred_mesh_obj.c("blue", 0.5),
                pred_points,
                label_points,
                camera=cam,
                resetcam=False,
                axes=1,
            )
            if screenshot:
                outfile = Path(f"{screenshot_out_dir}/sample.png").with_name(
                    f"{Path(nifti_file).stem}.png"
                )
                vedo.screenshot(str(outfile))

        save_name = label.split("_label")[0]
        metrics_dicts[save_name] = diff_list
    print("completed")

    with open(out_file, 'w') as file:
        json.dump(metrics_dicts, file)


def single_processing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti_file", default="./outputs/totalall")
    parser.add_argument("--output_file", default="./visualization/vis_result/morph/totalall_rib")
    parser.add_argument("--is_show", default=False, action="store_true")
    parser.add_argument("--screenshot", default=False, action="store_true")
    args = parser.parse_args()

    for f in os.listdir(args.nifti_file):
        input = os.path.join(args.nifti_file, f)
        output = os.path.join(args.output_file, f + ".json")
        main(input, output, args.is_show, args.screenshot)


if __name__ == "__main__":
    single_processing()
