"""visualize femur morphometry algorithm"""
from pathlib import Path
import argparse
import os
from functools import partial
from multiprocessing import Pool

"""femur morphometry utils"""
from typing import Tuple
import numpy as np
import vedo
import json
from torch import tensor
from monai.metrics import DiceMetric
from xrayto3d_morphometry import (
    get_nifti_stem,
    cma_es_search_candidate_cut_plane,
    get_angle_between_vectors,
    get_best_solution,
    get_closest_point_from_line,
    get_distance_to_line_segment,
    get_farthest_point_from_line_segment,
    get_line_segment,
    get_mesh_from_segmentation,
    get_array_and_mesh_from_segmentation,
    get_points_along_directions,
    get_vector_from_points,
    lerp,
    get_segmentation_volume,
    extract_volume_surface,
    read_volume,
    get_segmentation_labels,
)

femur_label_dict = {"head": 4, "neck": 3, "sub_troc": 2}


def get_subtrochanter_center(nifti_filename, label_id):
    """assume subtrochanter segmenetation exists"""
    vol = get_segmentation_volume(nifti_filename, label_id)
    if vol is None:
        return
    else:
        return extract_volume_surface(vol).center_of_mass()


def seg_contain_subtrochanter(nifti_filename) -> bool:
    """return True if the nifti segmentation contains subtrochanter
    region label
    """
    seg_vol = read_volume(nifti_filename)
    label_indexes = get_segmentation_labels(seg_vol)
    return femur_label_dict["sub_troc"] in label_indexes


def get_neck_shaft_angle(diaphysis_line, neck_line):
    """given line segment representing diaphsis axis and neck axis, evaluate neck shaft angle"""
    diaphysis_normal = get_vector_from_points(*diaphysis_line)
    neck_normal = get_vector_from_points(*neck_line)
    nsa = get_angle_between_vectors(diaphysis_normal, neck_normal)
    return (
        180.0 - nsa if nsa < 90.0 else nsa
    )  # sometimes the normals are pointed in opposite direction, detect this and correct


def get_femoral_head_offset(diaphysis_line, femoral_head: vedo.Sphere):
    """femoral head offset: perpendicular distance of femoral head center (center of rotation) from the diaphysis line"""
    return get_distance_to_line_segment(femoral_head.center.tolist(), *diaphysis_line)


def fit_femoral_head_sphere(p_c, pi_c_normal, mesh_obj) -> vedo.Sphere:
    """fit a sphere on a point cloud representing femoral head

    Args:
        p_c ([float,float,float]]): Point representing farthest point from diaphysis axis
        pi_c_normal ([float,float,float]): vector representing direction of the tangent plane at p_c
        mesh_obj (vedo.Mesh): femur mesh

    Returns:
        vedo.Sphere: Fitted sphere approximating the femoral head
    """
    pc_points = get_points_along_directions(p_c, pi_c_normal, 50, positive_only=True)
    candidate_femoral_head_cuts = [
        mesh_obj.clone().cut_with_plane(p, pi_c_normal).boundaries() for p in pc_points
    ]
    candidate_sphere_points = []
    for cut in candidate_femoral_head_cuts:
        candidate_sphere_points.extend(cut.points().tolist())
    head_sphere: vedo.Sphere = vedo.fit_sphere(candidate_sphere_points)  # type: ignore
    return head_sphere


def get_femur_morphometry(
        nifti_filename, subtrochanter_centroid: Tuple[float, float, float], label_id, robust=False
):
    """return key:val containing femur morphometry"""
    array, mesh_obj = get_array_and_mesh_from_segmentation(nifti_filename, label_id)
    # diaphysis axis
    cma_obj = cma_es_search_candidate_cut_plane(
        mesh_obj, subtrochanter_centroid, (0, 1, 0), verbose=False
    )
    diaphysis_direction = get_best_solution(cma_obj)
    if robust:
        # use the initial diaphysis axis to traverse through
        additional_diaphysis_center = get_points_along_directions(
            subtrochanter_centroid,
            diaphysis_direction,
        )
        additional_diaphysis_direction = [
            get_best_solution(
                cma_es_search_candidate_cut_plane(
                    mesh_obj, center, diaphysis_direction, verbose=False
                )
            )
            for center in additional_diaphysis_center
        ]
        diaphysis_direction = np.mean(additional_diaphysis_direction, axis=0)
    l_a = get_line_segment(subtrochanter_centroid, diaphysis_direction, 400)
    p_c, _ = get_farthest_point_from_line_segment(mesh_obj.points(), *l_a)  # type: ignore
    p_m = get_closest_point_from_line(mesh_obj.center_of_mass(), *l_a)  # type: ignore
    pi_c_normal = get_vector_from_points(p_c, p_m)

    # fit femoral head
    femoral_head = fit_femoral_head_sphere(p_c, pi_c_normal, mesh_obj)

    # fit femoral neck
    l_n = get_vector_from_points(femoral_head.center, p_m)
    p_n = lerp(femoral_head.center.tolist(), p_m, alpha=0.5)
    neck_es = cma_es_search_candidate_cut_plane(mesh_obj, p_n, l_n, verbose=False)
    neck_normal = get_best_solution(neck_es)
    if robust:
        # use initial neck axis to traverse through
        additional_p_n = get_points_along_directions(p_n, neck_normal)
        additional_neck_normal = [
            get_best_solution(
                cma_es_search_candidate_cut_plane(
                    mesh_obj, center, neck_normal, verbose=False
                )
            )
            for center in additional_p_n
        ]
        neck_normal = np.mean(additional_neck_normal, axis=0)
    if neck_normal[0] < 0:
        neck_normal = -neck_normal
    nsa = get_neck_shaft_angle(l_a, get_line_segment(p_n, neck_normal, 10))
    fhr = femoral_head.radius
    fo = get_femoral_head_offset(l_a, femoral_head)
    fhc_x, fhc_y, fhc_z = femoral_head.center.tolist()
    fda_x, fda_y, fda_z = (
        get_angle_between_vectors(diaphysis_direction, (1, 0, 0)),
        get_angle_between_vectors(diaphysis_direction, (0, 1, 0)),
        get_angle_between_vectors(diaphysis_direction, (0, 0, 1)),
    )
    fna_x, fna_y, fna_z = (
        get_angle_between_vectors(neck_normal, (1, 0, 0)),
        get_angle_between_vectors(neck_normal, (0, 1, 0)),
        get_angle_between_vectors(neck_normal, (0, 0, 1)),
    )

    show_points = [femoral_head.center,
                   femoral_head.center + 10 * neck_normal,
                   femoral_head.center - 10 * pi_c_normal]
    line1 = vedo.Line(femoral_head.center, femoral_head.center + 10 * neck_normal)
    line2 = vedo.Line(femoral_head.center, femoral_head.center - 10 * pi_c_normal)
    line = [line1, line2]
    # show_points = [[fhc_x, fhc_y, fhc_z]]
    vis_obj = [vedo.Points(show_points, c="blue", r=24), line]
    dict = {
        "nsa": nsa,
        "fhr": fhr,
        "fo": fo,
        "fhc": np.array([fhc_x, fhc_y, fhc_z]),
        "fda": np.array([fda_x, fda_y if fda_y < 90 else 180.0 - fda_y, fda_z]),
        "fna": np.array([fna_x, fna_y, fna_z])
    }
    return mesh_obj, vis_obj, array, dict


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
        for i in [53, 54]:
            diff_dict = {}
            label_subtrochanter_center = get_subtrochanter_center(os.path.join(nifti_file, label), i)
            if label_subtrochanter_center is None:
                continue
            try:
                label_mesh_obj, label_vis_obj, label_array, label_dict = get_femur_morphometry(
                    os.path.join(nifti_file, label), label_subtrochanter_center, i
                )
                # pred_subtrochanter_center = get_subtrochanter_center(os.path.join(nifti_file, pred), i)
                pred_mesh_obj, pred_vis_obj, pred_array, pred_dict = get_femur_morphometry(
                    os.path.join(nifti_file, pred), label_subtrochanter_center, i
                )
            except:
                continue

            dice = dice_func(tensor(np.expand_dims(pred_array, [0, 1])),
                             tensor(np.expand_dims(label_array, [0, 1])))
            diff_dict["Dice"] = float(np.mean(np.array(dice)))
            for p, l in zip(pred_dict, label_dict):
                if p == "fhc":
                    diff_dict[p] = float(np.linalg.norm((pred_dict[p] - label_dict[l]), 2))
                elif p == "fda" or p == "fna":
                    pr = np.cos(np.radians(pred_dict[p]))
                    lr = np.cos(np.radians(label_dict[p]))
                    cos_theta = np.dot(pr, lr) / np.linalg.norm(pr) / np.linalg.norm(lr)
                    angle = float(np.degrees(np.arccos(cos_theta)))
                    diff_dict[p] = angle if angle < 90 else 180 - angle
                else:
                    diff_dict[p] = float(abs(pred_dict[p] - label_dict[l]))
            diff_list[str(i)] = diff_dict

            if is_show:
                plotter = vedo.Plotter()
                plotter.clear()
                plotter.show(
                    pred_mesh_obj.c("red", 0.4),
                    pred_vis_obj,
                    label_mesh_obj.c("blue", 0.4),
                    label_vis_obj,
                    axes=1,
                )

            if screenshot:
                outfile = Path(f"{screenshot_out_dir}/sample.png").with_name(
                    f"{Path(nifti_file).stem}.png"
                )
                vedo.screenshot(str(outfile))

        save_name = label.split("_label")[0]
        metrics_dicts[save_name] = diff_list

    with open(out_file, 'w') as file:
        json.dump(metrics_dicts, file)


def single_processing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti_file", default="./outputs/totalall")
    parser.add_argument("--output_file", default="./visualization/vis_result/morph/totalall_femur")
    parser.add_argument("--is_show", default=False, action="store_true")
    parser.add_argument("--screenshot", default=False, action="store_true")
    args = parser.parse_args()

    for f in os.listdir(args.nifti_file):
        input = os.path.join(args.nifti_file, f)
        output = os.path.join(args.output_file, f + ".json")
        main(input, output, args.is_show, args.screenshot)


if __name__ == '__main__':
    single_processing()
