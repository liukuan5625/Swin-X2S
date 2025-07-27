"""thin wrapper around vtkMath"""

from typing import Sequence

import numpy as np
import vedo
from vtkmodules.all import vtkLine, vtkMath

from .tuple_ops import add_tuple, multiply_tuple_scalar


def get_distance_between_points(p1: Sequence[float], p2: Sequence[float]):
    """return euclidean distance between points"""
    return np.sqrt(vtkMath.Distance2BetweenPoints(p1, p2))


def get_distance2_between_points(p1: Sequence[float], p2: Sequence[float]):
    """ squared euclidean distance"""
    return vtkMath.Distance2BetweenPoints(p1, p2)


def get_distance2_to_line_segment(p0: Sequence[float], line_p0: Sequence[float], line_p1: Sequence[float]):
    """return squared distance to line segment"""
    return vtkLine.DistanceToLine(p0, line_p0, line_p1)


def get_distance_to_line_segment(p0: Sequence[float], line_p0: Sequence[float], line_p1: Sequence[float]):
    """get distance to line segment"""
    return np.sqrt(get_distance2_to_line_segment(p0, line_p0, line_p1))


def get_angle_between_vectors(v1, v2, degrees=True):
    angle = vtkMath.AngleBetweenVectors(v1, v2)
    if degrees:
        angle = vtkMath.DegreesFromRadians(angle)
    return angle


def get_vector_from_points(p1, p2):
    """return a vector pointing from p1 to p2 (not p2 to p1, order(direction) is important in vectors)"""
    v = np.subtract(p2, p1)
    v_norm = vtkMath.Norm(v)
    return v / v_norm


def lerp(p0: Sequence[float], p1: Sequence[float], alpha: float):
    """linear interpolation"""
    return tuple(a*alpha + b*(1.0 - alpha) for a, b in zip(p0, p1))


def get_midpoint(a: vedo.Points, b: vedo.Points):
    # midpoint = tuple((i + j)/2.0 for i, j in zip(a.GetPosition(), b.GetPosition()))
    # midpoint = lerp(a.GetPosition(), b.GetPosition(), 0.5)
    midpoint = lerp(a.pos(), b.pos(), 0.5)
    return vedo.Point(pos=midpoint)


def get_points_along_directions(point: Sequence[float], direction: Sequence[float],
                                num_points: int = 6, positive_only=False):
    """given a point and direction, sample points along the direction"""
    candidate_points = []
    if positive_only:
        for i in np.linspace(0, num_points, num_points):
            candidate_points.append(add_tuple(point, multiply_tuple_scalar(direction, i)))
    else:
        for i in np.linspace(-num_points//2, num_points//2, num_points):
            candidate_points.append(add_tuple(point, multiply_tuple_scalar(direction, i)))
    return candidate_points
