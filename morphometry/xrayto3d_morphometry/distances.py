from typing import List, Sequence, Union, Tuple

import numpy as np
import vedo
from point_cloud_utils import chamfer_distance

from .geom_ops import get_distance2_to_line_segment, get_distance_between_points
from .tuple_ops import add_tuple, multiply_tuple_scalar, subtract_tuple


def get_avg_pointcloud_distance(a, b, verbose=False):
    """average chamfer distance between point clouds"""
    d = 0
    for p in a.points():
        cpt = b.closest_point(p)
        d += vedo.mag2(p - cpt)  # square of residual distance

    residual = d / a.npoints
    if verbose:
        vedo.printc("ave. squared distance =", residual)
    return residual


def get_closest_points_between_point_clouds(
    p0: List[Sequence[float]], p1: List[Sequence[float]]
):
    """wrapper around `pointcloud_utils.chamfer_distance`"""
    d, corr_p0_to_p1, corr_p1_to_p0 = chamfer_distance(p0, p1, return_index=True)
    return d, corr_p0_to_p1, corr_p1_to_p0


def brute_force_search_get_closest_points_between_point_clouds(
    p0: List[Sequence[float]], p1: List[Sequence[float]]
):
    """closest points between point clouds"""
    mi = get_distance_between_points(p0[0], p1[0])
    p1_candidate = p0[0]
    p2_candidate = p1[0]
    ln_p0 = len(p0)
    ln_p1 = len(p1)
    for i in range(ln_p0):
        for j in range(ln_p1):
            d = get_distance_between_points(p0[i], p1[j])
            if d < mi:  # Update min_dist and points
                mi = d
                p1_candidate, p2_candidate = p0[i], p1[j]
    return p1_candidate, p2_candidate, mi


def get_closest_point_from_line(
    p0: Sequence[float], line_p0: Sequence[float], line_p1: Sequence[float]
):
    """closest point from line segment"""
    line_p0_numpy = np.asarray(line_p0, dtype=np.float32)
    line_p1_numpy = np.asarray(line_p1, dtype=np.float32)
    p0_numpy = np.asarray(p0, dtype=np.float32)

    ap = p0_numpy - line_p0_numpy
    ab = line_p1_numpy - line_p0_numpy
    result = line_p0_numpy + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result.tolist()


def project_points_onto_line(points: List, line_p0: Sequence[float], line_p1: Sequence[float]):
    """projection of points on line"""
    projections_on_line = [
        get_closest_point_from_line(p, line_p0, line_p1) for p in points
    ]
    return projections_on_line


def get_closest_point_from_plane(
    mesh_obj: vedo.Mesh, plane: Union[vedo.Plane, Sequence[float]]
):
    return get_extrema_from_plane(mesh_obj, plane, apply_fn=np.argmin)


def get_farthest_point_from_plane(
    mesh_obj: vedo.Mesh, plane: Union[vedo.Plane, Sequence[float]]
):
    return get_extrema_from_plane(mesh_obj, plane, apply_fn=np.argmax)


def get_farthest_point_from_line_segment(
    points: np.ndarray, line_p0: Sequence[float], line_p1: Sequence[float]
):
    """return the point farthest from line segment represented by start and end point"""
    distance_to_line = [
        get_distance2_to_line_segment(p, line_p0, line_p1) for p in points
    ]
    candidate_point_idx = np.argmax(distance_to_line)
    return points[candidate_point_idx], candidate_point_idx


def get_closest_point_from_line_segment(
    points: np.ndarray, line_p0: Sequence[float], line_p1: Sequence[float]
):
    """return the point farthest from line segment represented by start and end point"""
    distance_to_line = [
        get_distance2_to_line_segment(p, line_p0, line_p1) for p in points
    ]
    candidate_point_idx = np.argmin(distance_to_line)
    return points[candidate_point_idx], candidate_point_idx


def get_line_segment(
    center: Sequence[float], direction: Sequence[float], distance=400
) -> Tuple[Sequence, Sequence]:
    """return line segment defined by start and end point coordinates"""
    if isinstance(center, vedo.Points):
        center = center.GetPosition()
    if isinstance(direction, vedo.Points):
        direction = direction.GetPosition()

    line_p0 = add_tuple(center, multiply_tuple_scalar(direction, distance))
    line_p1 = subtract_tuple(center, multiply_tuple_scalar(direction, distance))
    return line_p0, line_p1


def get_farthest_point_along_axis(
    points: np.ndarray, axis: int, negative: bool = False
):
    """find the coordinates and index of the point at furthest distance along an axis"""
    if negative:
        candidate_point_idx = np.argmin(points[:, abs(axis)])
    else:
        candidate_point_idx = np.argmax(points[:, abs(axis)])
    return points[candidate_point_idx], candidate_point_idx


def get_extrema_from_plane(
    mesh_obj: vedo.Mesh, plane: Union[vedo.Plane, Sequence[float]], apply_fn
):
    """if plane is a vector, treat it as a normal to the plane"""
    if isinstance(plane, Sequence):
        plane = vedo.Plane(normal=plane)
    mesh_obj.distance_to(plane)
    candidate_point_idx = apply_fn(mesh_obj.pointdata["Distance"])
    return mesh_obj.points()[candidate_point_idx], candidate_point_idx
