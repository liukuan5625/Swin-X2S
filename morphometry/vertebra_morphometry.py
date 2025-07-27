import argparse
from typing import Sequence
from pathlib import Path
import vedo
import numpy as np
import json
import os
from torch import tensor
from sklearn.linear_model import QuantileRegressor
from sklearn.cluster import KMeans
from xrayto3d_morphometry import (
    get_mesh_from_segmentation,
    get_array_and_mesh_from_segmentation,
    move_to,
    move_to_origin,
    get_nifti_stem,
    file_type_gt_or_pred,
    add_tuple,
    subtract_tuple,
    multiply_tuple_scalar,
    brute_force_search_get_closest_points_between_point_clouds,
    get_vector_from_points,
    get_angle_between_vectors,
    project_points_onto_line,
    get_distance_between_points,
    get_distance2_between_points,
    lerp,
    get_oriented_camera,
)
from monai.metrics import DiceMetric
BONE_COLOR = (255.0, 193.0, 149.0)


def fit_l1(data, alpha=0.2):
    """fit a L1 regularized linear model
    data: (N,3) where N is number of data points
    alpha: regularization strength
    """
    quantile_reg = QuantileRegressor(solver="interior-point", alpha=alpha).fit(
        data[:, :2], data[:, 2]
    )
    fitted_data = []
    for datum in data:
        x, y, z = datum
        (z_new,) = (
            quantile_reg.predict(
                [
                    [x, y],
                ]
            )
            .flatten()
            .tolist()
        )
        fitted_data.append([x, y, z_new])
    return fitted_data


def get_axis_lines(origin, boundary_points, scale=40):
    vb_axes = vedo.pca_ellipsoid(boundary_points)
    # print(vb_axes.va, vb_axes.vb)
    vb_ax1_p0 = add_tuple(
        tuple(origin), multiply_tuple_scalar(tuple(vb_axes.axis1), scale)
    )
    vb_ax1_p1 = subtract_tuple(
        tuple(origin), multiply_tuple_scalar(tuple(vb_axes.axis1), scale)
    )
    vb_ax2_p0 = add_tuple(
        tuple(origin), multiply_tuple_scalar(tuple(vb_axes.axis2), scale)
    )
    vb_ax2_p1 = subtract_tuple(
        tuple(origin), multiply_tuple_scalar(tuple(vb_axes.axis2), scale)
    )
    axis1_line = (vb_ax1_p0, vb_ax1_p1)
    axis2_line = (vb_ax2_p0, vb_ax2_p1)
    return axis1_line, axis2_line


def get_slope_intercept_from_two_points_z_y(p0: Sequence[float], p1: Sequence[float]):
    "z is the independent dimension, y is the dependent dimension"
    p0_x, p0_y, p0_z = p0
    p1_x, p1_y, p1_z = p1
    # y = mz+c, m = y2 - y1 / z2 - z1, c = y1 - m*z1
    m = (p1_y - p0_y) / (p1_z - p0_z)
    c = p1_y - m * p1_z
    return m, c


def get_slope_intercept_from_two_points_y_z(p0: Sequence[float], p1: Sequence[float]):
    "y is the independent dimension, z is the dependent dimension"
    p0_x, p0_y, p0_z = p0
    p1_x, p1_y, p1_z = p1
    # z = my+c, m = z2 - z1 / y2 - y1, c = z1 - m*y1
    m = (p1_z - p0_z) / (p1_y - p0_y)
    c = p1_z - m * p1_y
    return m, c


def get_symmetry_plane(vert_mesh, shift=None):
    vert_mesh = vert_mesh.clone(deep=True)
    mirrored_vert_mesh = vert_mesh.clone(deep=True).mirror("z")
    if shift is None:
        _, shift = move_to_origin(vert_mesh)
        move_to_origin(mirrored_vert_mesh)
    else:
        move_to(vert_mesh, shift)
        shift[-1] = -shift[-1]
        move_to(mirrored_vert_mesh, shift)

    mirrored_vert_points = vedo.Points(mirrored_vert_mesh.points())
    vert_mesh_points = vedo.Points(
        vert_mesh.clone(deep=True).points()
    )
    aligned_pts1 = mirrored_vert_points.clone().align_to(vert_mesh_points, rigid=True, invert=False).c('blue')

    # draw arrows to see where points end up
    rand_idx = np.random.randint(0, len(vert_mesh.points()), 100)
    sampled_vmp = vert_mesh.points()[rand_idx]
    sampled_apts1 = aligned_pts1.points()[rand_idx]
    avg_points = [lerp(a, b, 0.5) for a, b in zip(sampled_vmp, sampled_apts1)]
    sym_plane = vedo.fit_plane(avg_points, signed=True)
    return vert_mesh, sym_plane, shift


def get_fitted_line_along_y(ap_or_lat_line: vedo.Line, boundary_points: vedo.Points):
    """update fitted line to avoid longer lines than required"""
    vb_anterior_proj = project_points_onto_line(
        boundary_points, *ap_or_lat_line.points()
    )
    anterior_up_proj = [
        (x, y, z) for x, y, z in vb_anterior_proj if y < ap_or_lat_line.center[1]
    ]
    anterior_down_proj = [
        (x, y, z) for x, y, z in vb_anterior_proj if y > ap_or_lat_line.center[1]
    ]

    anterior_up_most_proj_id = np.argmax(
        [
            get_distance2_between_points(ap_or_lat_line.center, p)
            for p in anterior_up_proj
        ]
    )
    anterior_up_most_proj = anterior_up_proj[anterior_up_most_proj_id]

    anterior_down_most_proj_id = np.argmax(
        [
            get_distance2_between_points(ap_or_lat_line.center, p)
            for p in anterior_down_proj
        ]
    )
    anterior_down_most_proj = anterior_down_proj[anterior_down_most_proj_id]
    # update line
    ap_or_lat_line = vedo.Line(anterior_down_most_proj, anterior_up_most_proj)
    return ap_or_lat_line


def get_fitted_line_along_z(sup_or_inf_line: vedo.Line, boundary_points: vedo.Points):
    """update fitted line to avoid longer lines than required"""
    vb_anterior_proj = project_points_onto_line(
        boundary_points, *sup_or_inf_line.points()
    )
    anterior_up_proj = [
        (x, y, z) for x, y, z in vb_anterior_proj if z < sup_or_inf_line.center[2]
    ]
    anterior_down_proj = [
        (x, y, z) for x, y, z in vb_anterior_proj if z > sup_or_inf_line.center[2]
    ]

    anterior_up_most_proj_id = np.argmax(
        [
            get_distance2_between_points(sup_or_inf_line.center, p)
            for p in anterior_up_proj
        ]
    )
    anterior_up_most_proj = anterior_up_proj[anterior_up_most_proj_id]

    anterior_down_most_proj_id = np.argmax(
        [
            get_distance2_between_points(sup_or_inf_line.center, p)
            for p in anterior_down_proj
        ]
    )
    anterior_down_most_proj = anterior_down_proj[anterior_down_most_proj_id]
    # update line
    sup_or_inf_line = vedo.Line(anterior_down_most_proj, anterior_up_most_proj)
    return sup_or_inf_line


def get_vertebra_measurements(vert_mesh, sym_plane):
    # initial orientation
    vert_mesh.compute_normals()

    # setup symmetry plane: mirroring and registration
    # sym_plane = get_symmetry_plane(vert_mesh)

    cut_mesh = vert_mesh.clone().cut_with_plane(
        normal=(sym_plane.normal)
    )
    sym_plane_boundaries = cut_mesh.boundaries()
    sym_plane_points = sym_plane_boundaries.points().tolist()

    # use kmeans to sepearte the vertebral body and spinous process boundary points
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(np.array(sym_plane_points))
    c0_x, c0_y, c0_z = kmeans.cluster_centers_[0]
    c1_x, c1_y, c1_z = kmeans.cluster_centers_[1]
    vb_label = 1 if c1_y > c0_y else 0
    sp_label = int(not bool(vb_label))
    vertebral_body_points = [
        p
        for p in sym_plane_points
        if kmeans.predict(
            [
                p,
            ]
        )[0]
        == vb_label
    ]
    spinous_process_points = [
        p
        for p in sym_plane_points
        if kmeans.predict(
            [
                p,
            ]
        )[0]
        == sp_label
    ]

    # smooth vertebral body points
    vertebral_body_points = [
        np.mean(sym_plane_boundaries.closest_point(p, n=10), axis=0).tolist()
        for p in vertebral_body_points
    ]

    vbc = np.mean(vertebral_body_points, axis=0)
    spc = np.mean(spinous_process_points, axis=0)
    v0, s0, vcl = brute_force_search_get_closest_points_between_point_clouds(
        vertebral_body_points, spinous_process_points
    )
    # sanity check: if vcl is very small, say less than 2mm
    # then, the kmeans clustering algorithm above failed to separate vertebral body points and spinous process points
    pq_unit_vec = get_vector_from_points(v0, s0)
    vb_axis1, vb_axis2 = get_axis_lines(vbc, vertebral_body_points)
    sp_axis1, sp_axis2 = get_axis_lines(spc, spinous_process_points)

    # calculate spinous process morphometry: spa, spl
    project_spp = project_points_onto_line(spinous_process_points, *sp_axis1)
    anterior_project_spp = [(x, y, z) for x, y, z in project_spp if z < spc[2]]
    posterior_project_spp = [(x, y, z) for x, y, z in project_spp if z > spc[2]]
    posterior_most_spp_id = np.argmax(
        [get_distance_between_points(spc, p) for p in posterior_project_spp]
    )
    posterior_most_spp = posterior_project_spp[posterior_most_spp_id]
    anterior_most_spp_id = np.argmax(
        [get_distance_between_points(spc, p) for p in anterior_project_spp]
    )
    anterior_most_spp = anterior_project_spp[anterior_most_spp_id]
    spl = get_distance_between_points(anterior_most_spp, posterior_most_spp)

    # find upper endplate and lower endplate points
    # vert_normals = vert_mesh.normals(recompute=False)
    vert_mesh.compute_normals()
    vert_normals = vert_mesh.pointdata['Normals']
    vb_normals = [
        vert_normals[vert_mesh.closest_point(p, return_point_id=True)]
        for p in vertebral_body_points
    ]
    # calculate dot product of vertebral body normals wrt vertebra foramen
    vbn_projections = [np.dot(vbn, pq_unit_vec) for vbn in vb_normals]
    vb_endplate = [
        vertebral_body_points[i]
        for i, vbnp in enumerate(vbn_projections)
        if ((vbnp < 0.5) and (vbnp > -0.5))
    ]
    vb_anteroposterior = [
        vertebral_body_points[i]
        for i, vbnp in enumerate(vbn_projections)
        if ((vbnp > 0.5) or (vbnp < -0.5))
    ]

    # separate endplates
    m, c = get_slope_intercept_from_two_points_z_y(*vb_axis1)
    vb_up = np.asarray([(x, y, z) for x, y, z in vb_endplate if (z * m + c) > y])
    vb_lp = np.asarray([(x, y, z) for x, y, z in vb_endplate if (z * m + c) < y])

    # separate anteroposterior boundaries
    m, c = get_slope_intercept_from_two_points_y_z(*vb_axis2)
    vb_ap = np.asarray([(x, y, z) for x, y, z in vb_anteroposterior if (y * m + c) > z])
    vb_pp = np.asarray([(x, y, z) for x, y, z in vb_anteroposterior if (y * m + c) < z])

    a_bs = vedo.fit_line(np.asarray(vb_up))
    a_bi = vedo.fit_line(np.asarray(vb_lp))
    # update projectin lines to stay flush
    a_bs = get_fitted_line_along_z(a_bs, vb_up)
    a_bi = get_fitted_line_along_z(a_bi, vb_lp)

    a_bs_0, a_bs_1 = a_bs.points()
    a_bi_0, a_bi_1 = a_bi.points()
    a_bm_0 = lerp(a_bs_0, a_bi_0, 0.5)
    a_bm_1 = lerp(a_bs_1, a_bi_1, 0.5)
    a_bm = vedo.fit_line(np.asarray([a_bm_0, a_bm_1]))

    a_ba = vedo.fit_line(np.asarray(vb_ap))
    a_bp = vedo.fit_line(np.asarray(vb_pp))

    # update projection lines to stay flush (with no spikes https://github.com/naamiinepal/xrayto3D-morphometry/issues/18)
    a_ba = get_fitted_line_along_y(a_ba, vb_ap)
    a_bp = get_fitted_line_along_y(a_bp, vb_pp)

    # spa is the angle between sp_axis1 and a_bs
    a_bs_vec = get_vector_from_points(*a_bs.points())
    spl_vec = get_vector_from_points(*sp_axis1)
    spa = get_angle_between_vectors(spl_vec, a_bs_vec)

    # vertbral body measurements
    anterior_vb_height = get_distance_between_points(*a_ba.points())
    posterior_vb_height = get_distance_between_points(*a_bp.points())
    superior_vb_length = get_distance_between_points(*a_bs.points())
    inferior_vb_length = get_distance_between_points(*a_bi.points())

    visualization_objects = {
        vedo.Points([vbc, spc, v0, s0]),
        vedo.Points(
            [*a_bs.points(), *a_bi.points(), *a_ba.points(), *a_bp.points()],
            r=8,
            c="red",
        ),
        vedo.Points([posterior_most_spp, anterior_most_spp], r=8, c="red"),
        vedo.Points(spinous_process_points),
        a_bs.lw(5),
        a_bi.lw(5),
        a_ba.lw(5),
        a_bp.lw(5),
        a_bm,
        vedo.Line(sp_axis1),
        vedo.Line(v0, s0),
    }
    return {
        "spl": spl,
        "spa": 180.0 - spa,
        "avbh": anterior_vb_height,
        "pvbh": posterior_vb_height,
        "svbl": superior_vb_length,
        "ivbl": inferior_vb_length,
        "vcl": vcl,
    }, visualization_objects


def main(
    nifti_file, out_file, is_show=False, screenshot=False, screenshot_out_dir="./screenshots"
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

    """single file processing entry point"""
    dice_func = DiceMetric()
    metrics_dicts = {}
    for pred, label in zip(pred_files, label_files):
        diff_list = {}
        for i in range(1, 26):
            pred_vert_array, pred_vert_mesh = get_array_and_mesh_from_segmentation(
                os.path.join(nifti_file, pred), label_idx=i, largest_component=True, reorient=False
            )
            label_vert_array, label_vert_mesh = get_array_and_mesh_from_segmentation(
                os.path.join(nifti_file, label), label_idx=i, largest_component=True, reorient=False
            )

            if np.sum(pred_vert_array) < 250 or np.sum(label_vert_array) < 250: continue

            try:
                label_vert_mesh, label_sym_plane, shift = get_symmetry_plane(label_vert_mesh)
                label_metrics_dict, label_visualization_objects = get_vertebra_measurements(label_vert_mesh, label_sym_plane)

                pred_vert_mesh, pred_sym_plane, shift = get_symmetry_plane(pred_vert_mesh, shift)
                pred_metrics_dict, pred_visualization_objects = get_vertebra_measurements(pred_vert_mesh, pred_sym_plane)

            except Exception as e:
                print("e")
                continue

            dice = dice_func(tensor(np.expand_dims(pred_vert_array, [0, 1])),
                             tensor(np.expand_dims(label_vert_array, [0, 1])))[0, 0]

            diff_dict = {}
            diff_dict["Dice"] = float(dice)
            for p, l in zip(pred_metrics_dict, label_metrics_dict):
                diff_dict[p] = abs(pred_metrics_dict[p] - label_metrics_dict[l])
            diff_list[str(i)] = diff_dict

            if is_show:
                topview_cam = get_oriented_camera(pred_vert_mesh, axis=1, camera_dist=-200)
                topview_cam["viewup"] = (-1, 0, 0)
                sideview_cam = get_oriented_camera(pred_vert_mesh, axis=0, camera_dist=200)
                sideview_cam["viewup"] = (0, 1, 0)

                plotter = vedo.Plotter()
                plotter.clear()
                plotter.show(
                    pred_vert_mesh.clone()
                    .cut_with_plane(normal=pred_sym_plane.normal, invert=True)
                    .c("green", alpha=0.5),
                    # sym_plane.opacity(0.5),
                    *pred_visualization_objects,

                    label_vert_mesh.clone()
                    .cut_with_plane(normal=label_sym_plane.normal, invert=True)
                    .c("blue", alpha=0.5),
                    # sym_plane.opacity(0.5),
                    *label_visualization_objects,

                    axes=3,
                    camera=sideview_cam,
                    resetcam=False,
                    interactive=True
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
    parser.add_argument("--output_file", default="./visualization/vis_result/morph/totalall_spine")
    parser.add_argument("--is_show", default=False, action="store_true")
    parser.add_argument("--screenshot", default=False, action="store_true")
    args = parser.parse_args()

    for f in os.listdir(args.nifti_file):
        input = os.path.join(args.nifti_file, f)
        output = os.path.join(args.output_file, f + ".json")
        main(input, output, args.is_show, args.screenshot)


if __name__ == "__main__":
    single_processing()
