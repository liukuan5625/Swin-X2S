import vedo
from typing import List, Sequence
from .tuple_ops import add_tuple, multiply_tuple_scalar


def get_oriented_camera(mesh_obj: vedo.Mesh, axis, camera_dist=200):
    """return a camera dict with
    focal point: mesh_obj center of mass
    camera_position at distance camera_dist from focal_point
    and oriented along CT coordinate system.
    The view-up vector was empirically determined to obtain sagittal, transverse and anteroposterior views from the camera.
    """
    x0, x1, y0, y1, z0, z1 = mesh_obj.bounds()
    focal_point = mesh_obj.center_of_mass()

    position = list(focal_point).copy()
    position[axis] -= camera_dist

    distance = abs(camera_dist)
    clipping_range = (x0, x1) if axis == 0 else (y0, y1) if axis == 1 else (z0, z1)
    viewup = (0, 1, 0) if axis == 2 else (0, 0, 1) if axis == 1 else (0, 0, 1)
    return {
        "position": position,
        "focal_point": focal_point,
        "viewup": viewup,
        "distance": distance,
        "clipping_range": clipping_range,
    }


def get_direction_axes(center, axis1, axis2, axis3, scale=20) -> List[vedo.Mesh]:
    axis1 = tuple(axis1)
    axis2 = tuple(axis2)
    axis3 = tuple(axis3)
    center = tuple(center)

    a = vedo.Arrow(
        center, add_tuple(center, multiply_tuple_scalar(axis1, scale)), c="r"
    )
    b = vedo.Arrow(
        center, add_tuple(center, multiply_tuple_scalar(axis2, scale)), c="g"
    )
    c = vedo.Arrow(
        center, add_tuple(center, multiply_tuple_scalar(axis3, scale)), c="b"
    )
    return [a, b, c]


def get_direction_axes_from_ellipsoid(ellipsoid: vedo.Ellipsoid) -> List[vedo.Mesh]:
    return get_direction_axes(
        ellipsoid.center, ellipsoid.axis1, ellipsoid.axis2, ellipsoid.axis3
    )


def get_arrow_actor(origin: Sequence[float], direction: Sequence[float], scale=20):
    return vedo.Arrow(
        origin, add_tuple(origin, multiply_tuple_scalar(direction, scale))
    )
