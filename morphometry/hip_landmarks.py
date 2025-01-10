"""hip landmark extraction script"""
import argparse
from pathlib import Path
import vedo
import numpy as np
from functools import partial
import json
import os
from multiprocessing import Pool
import csv
from monai.networks.utils import one_hot
from torch import tensor
from monai.metrics import DiceMetric
from xrayto3d_morphometry import (
    align_along_principal_axes,
    get_asis_estimate,
    get_quadrant_cuts,
    get_ischial_points_estimate,
    get_maximal_pelvic_width,
    get_mesh_from_segmentation,
    get_array_and_mesh_from_segmentation,
    get_midpoint,
    get_oriented_camera,
    get_transverse_plane_height,
    move_to,
    move_to_origin,
    get_app_plane_rotation_matrix,
    get_farthest_point_along_axis,
    get_distance_between_points,
    get_nifti_stem,
    file_type_gt_or_pred,
)

np.set_printoptions(precision=3, suppress=True)


def get_landmark_formatted_header():
    """return landmark header for readability"""
    header = (
        "id,gt_or_pred"
        + ",asis_l_x,asis_l_y,asis_l_z"
        + ",asis_r_x,asis_r_y,asis_r_z"
        + ",pt_l_x,pt_l_y,pt_l_z"
        + ",pt_r_x,pt_r_y,pt_r_z"
        + ",is_l_x,is_l_y,is_l_z"
        + ",is_r_x,is_r_y,is_r_z"
        + ",psis_l_x,psis_l_y,psis_l_z"
        + ",psis_r_x,psis_r_y,psis_r_z"
    )
    return header


def write_log_header(filepath, filename):
    """write output log header"""
    outdir = Path(f"{filepath}/")
    outdir.mkdir(exist_ok=True)
    with open(outdir / f"{filename}", "w", encoding="utf-8") as f:
        header = get_landmark_formatted_header()
        f.write(f"{header}/n")


def get_landmark_formatted_row(nifti_file, landmarks):
    """output formatted string containing comma-separated landmarks"""
    return f"{get_nifti_stem(str(nifti_file))[:5]},{file_type_gt_or_pred(str(nifti_file))},{landmarks['ASIS_L'][0]:.3f},{landmarks['ASIS_L'][1]:.3f},{landmarks['ASIS_L'][2]:.3f},{landmarks['ASIS_R'][0]:.3f},{landmarks['ASIS_R'][1]:.3f},{landmarks['ASIS_R'][2]:.3f},{landmarks['PT_L'][0]:.3f},{landmarks['PT_L'][1]:.3f},{landmarks['PT_L'][2]:.3f},{landmarks['PT_R'][0]:.3f},{landmarks['PT_R'][1]:.3f},{landmarks['PT_R'][2]:.3f},{landmarks['IS_L'][0]:.3f},{landmarks['IS_L'][1]:.3f},{landmarks['IS_L'][2]:.3f},{landmarks['IS_R'][0]:.3f},{landmarks['IS_R'][1]:.3f},{landmarks['IS_R'][2]:.3f},{landmarks['PSIS_L'][0]:.3f},{landmarks['PSIS_L'][1]:.3f},{landmarks['PSIS_L'][2]:.3f},{landmarks['PSIS_R'][0]:.3f},{landmarks['PSIS_R'][1]:.3f},{landmarks['PSIS_R'][2]:.3f}"


def get_landmarks(mesh_obj, shift=None):
    """return landmarks as dict"""
    if shift is not None:
        move_to(mesh_obj, shift)
    else:
        _, shift = move_to_origin(mesh_obj)
    # rotate around principal axis
    aligned_mesh_obj, T = align_along_principal_axes(mesh_obj)
    #      ______________________________________________________
    #     |    Axes    |      X      |      Y      |      Z      |
    #     |  Positive  |    Left     |    Inferior |   Anterior  |
    #     |  Negative  |    Right    |    Superior |  Posterior  |
    #     |______________________________________________________|
    mwp_midpoint = get_maximal_pelvic_width(aligned_mesh_obj)[-1]
    tph = get_transverse_plane_height(aligned_mesh_obj, mwp_midpoint, alpha=0.6)[0]
    bottom_left, top_left, bottom_right, top_right = get_quadrant_cuts(
        aligned_mesh_obj, transverse_plane_pos=(0, tph, 0)
    )

    asis_p1, asis_p2, pt_p1, pt_p2, ps, asis_plane = get_asis_estimate(
        bottom_left, top_left, bottom_right, top_right
    )
    # sanity check: for some cases, aligning along principal axes results
    # in change in mesh orientation, bring them back to correct orientation
    # by checking ASIS and MWP orientation
    asis_midpoint = get_midpoint(asis_p1, asis_p2)
    asis_x, asis_y, asis_z = asis_midpoint.pos()
    mwp_x, mwp_y, mwp_z = mwp_midpoint.pos()
    redo_asis_estimate = False
    if asis_y < mwp_y:
        aligned_mesh_obj.rotate_x(
            angle=180, around=aligned_mesh_obj.center_of_mass()
        )
        redo_asis_estimate = True
    if redo_asis_estimate:
        mwp_midpoint = get_maximal_pelvic_width(aligned_mesh_obj)[-1]
        tph = get_transverse_plane_height(aligned_mesh_obj, mwp_midpoint, alpha=0.6)[0]
        bottom_left, top_left, bottom_right, top_right = get_quadrant_cuts(
            aligned_mesh_obj, transverse_plane_pos=(0, tph, 0)
        )

        asis_p1, asis_p2, pt_p1, pt_p2, ps, asis_plane = get_asis_estimate(
            bottom_left, top_left, bottom_right, top_right
        )

    # second iteration: apply transformation and get asis estimate
    T = get_app_plane_rotation_matrix(
        pt_p1.pos(),
        pt_p2.pos(),
        asis_p1.pos(),
        asis_p2.pos(),
    )
    aligned_mesh_obj.apply_transform(T)
    bottom_left.apply_transform(T)
    top_left.apply_transform(T)
    bottom_right.apply_transform(T)
    top_right.apply_transform(T)
    asis_p1, asis_p2, pt_p1, pt_p2, ps, asis_plane = get_asis_estimate(
        bottom_left, top_left, bottom_right, top_right
    )

    # get ischial points
    ps_x, ps_y, ps_z = ps.pos()
    asis_midpoint = get_midpoint(asis_p1, asis_p2)
    app_x, app_y, app_z = get_midpoint(ps, asis_midpoint).pos()
    is_1, is_2 = get_ischial_points_estimate(aligned_mesh_obj, ps_y, app_y)

    # get PSIS points
    cut_plane_origin = (0, asis_p1.pos()[1], 0)
    superior_boundary = get_farthest_point_along_axis(
        top_left.points(), axis=1, negative=True
    )[0]
    psis_found = True
    while True:
        top_left_cut = (
            aligned_mesh_obj.clone()
            .cut_with_plane(normal=(0, 1, 0), origin=cut_plane_origin, invert=True)
            .cut_with_plane(normal=(1, 0, 0))
        )
        top_right_cut = (
            aligned_mesh_obj.clone()
            .cut_with_plane(normal=(0, 1, 0), origin=cut_plane_origin, invert=True)
            .cut_with_plane(normal=(1, 0, 0), invert=True)
        )
        try:
            psis_p1 = vedo.Point(
                get_farthest_point_along_axis(
                    top_left_cut.points(), axis=2, negative=True
                )[0]
            )
            psis_p2 = vedo.Point(
                get_farthest_point_along_axis(
                    top_right_cut.points(), axis=2, negative=True
                )[0]
            )
        except ValueError:
            print(f"error in obtaining psis")
            psis_found = False
            break

        if (
            (
                (abs(psis_p1.pos()[0]) < 10)
                or (abs(psis_p2.pos()[0]) < 10)
            )
            or (
                get_distance_between_points(
                    psis_p1.pos(), psis_p2.pos()
                )
                < 20
            )
            or (abs(cut_plane_origin[1] - 5) >= abs(superior_boundary[1]))
        ):
            cut_plane_origin = (0, cut_plane_origin[1] - 2, 0)
        else:
            break

    #  return coordinates in original mesh space, not aligned mesh space
    asis_p1_idx = aligned_mesh_obj.closest_point(
        asis_p1.pos(), return_point_id=True
    )
    asis_p2_idx = aligned_mesh_obj.closest_point(
        asis_p2.pos(), return_point_id=True
    )
    pt_p1_idx = aligned_mesh_obj.closest_point(
        pt_p1.pos(), return_point_id=True
    )
    pt_p2_idx = aligned_mesh_obj.closest_point(
        pt_p2.pos(), return_point_id=True
    )
    is_p1_idx = aligned_mesh_obj.closest_point(is_1, return_point_id=True)
    is_p2_idx = aligned_mesh_obj.closest_point(is_2, return_point_id=True)

    if psis_found:
        psis_p1_idx = aligned_mesh_obj.closest_point(
            psis_p1.pos(), return_point_id=True
        )
        psis_p2_idx = aligned_mesh_obj.closest_point(
            psis_p2.pos(), return_point_id=True
        )
    else:
        psis_p1_idx = None
        psis_p2_idx = None

    return {
        "ASIS_L": asis_p1_idx,
        "ASIS_R": asis_p2_idx,
        "PT_L": pt_p1_idx,
        "PT_R": pt_p2_idx,
        "IS_L": is_p1_idx,
        "IS_R": is_p2_idx,
        "PSIS_L": psis_p1_idx,
        "PSIS_R": psis_p2_idx,
    }, aligned_mesh_obj, shift


def main(
        nifti_file,
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
    diff_list = {}
    for pred, label in zip(pred_files, label_files):
        pred_array, pred_mesh_obj = get_array_and_mesh_from_segmentation(os.path.join(nifti_file, pred),
                                                                         label_idx=[50, 51, 52])
        if pred_mesh_obj == 0:
            continue
        pred_mesh_obj.rotate_x(180, around=pred_mesh_obj.center_of_mass())
        try:
            pred_landmark_indices, pred_aligned_mesh_obj, shift = get_landmarks(pred_mesh_obj)
        except:
            continue
        pred_landmarks = {
            key: pred_mesh_obj.points()[pred_landmark_indices[key]]
            if pred_landmark_indices[key] is not None
            else np.asarray((0.0, 0.0, 0.0))
            for key in pred_landmark_indices
        }
        pred_landmarks_list = [pred_landmarks[key] for key in pred_landmarks]

        label_array, label_mesh_obj = get_array_and_mesh_from_segmentation(os.path.join(nifti_file, label),
                                                                           label_idx=[50, 51, 52])
        if label_mesh_obj == 0:
            continue
        label_mesh_obj.rotate_x(180, around=label_mesh_obj.center_of_mass())
        try:
            label_landmark_indices, label_aligned_mesh_obj, shift = get_landmarks(label_mesh_obj, shift)
        except:
            continue
        label_landmarks = {
            key: label_mesh_obj.points()[label_landmark_indices[key]]
            if label_landmark_indices[key] is not None
            else np.asarray((0.0, 0.0, 0.0))
            for key in label_landmark_indices
        }
        label_landmarks_list = [label_landmarks[key] for key in label_landmarks]

        diff_dict = {}
        dice = dice_func(tensor(np.expand_dims(pred_array, [0, 1])),
                         tensor(np.expand_dims(label_array, [0, 1])))
        diff_dict["Dice"] = float(np.mean(np.array(dice)))

        for p, l in zip(pred_landmarks, label_landmarks):
            diff_dict[p] = float(np.linalg.norm(pred_landmarks[p] - label_landmarks[l], 2))

        if is_show:
            cam = get_oriented_camera(pred_mesh_obj, axis=2, camera_dist=400)

            plotter = vedo.Plotter()
            plotter.clear()
            plotter.show(
                pred_mesh_obj.c("green", 0.5),
                label_mesh_obj.c("blue", 0.5),
                vedo.Points(pred_landmarks_list, c="red", alpha=0.5, r=24),
                vedo.Points(label_landmarks_list, c="yellow", alpha=0.5, r=24),
                resetcam=False,
                camera=cam,
                axes=1,
            )
            if screenshot:
                outfile = Path(f"{screenshot_out_dir}/sample.png").with_name(
                    f"{Path(nifti_file).stem}.png"
                )
                vedo.screenshot(str(outfile))

        save_name = label.split("_label")[0]
        diff_list[save_name] = diff_dict

    with open(out_file, 'w') as file:
        json.dump(diff_list, file)


def single_processing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti_file", default="./outputs/totalall")
    parser.add_argument("--output_file", default="./visualization/vis_result/morph/totalall_pelvic")
    parser.add_argument("--is_show", default=False, action="store_true")
    parser.add_argument("--screenshot", default=False, action="store_true")
    args = parser.parse_args()

    for f in os.listdir(args.nifti_file):
        input = os.path.join(args.nifti_file, f)
        output = os.path.join(args.output_file, f+".json")
        main(input, output, args.is_show, args.screenshot)


if __name__ == "__main__":
    single_processing()
