"""optimization utils"""
import sys
from typing import Sequence, Tuple

import cma
import numpy as np
import vedo

from .tuple_ops import add_tuple, multiply_tuple_scalar


def get_cross_section_area(
    mesh_obj: vedo.Mesh, plane_origin: Sequence[float], plane_normal: Sequence[float]
):
    """return cross section area cut by a plane on a mesh at `plane_origin`"""
    sliced_mesh = mesh_obj.clone().cut_with_plane(
        origin=plane_origin, normal=plane_normal
    )
    return sliced_mesh.boundaries().triangulate().area()


def get_best_solution(cma_obj: cma.CMAEvolutionStrategy):
    """wrapper to CMAEvolutionStrategy object"""
    solution, _, _ = cma_obj.best.get()
    return solution


def cma_es_search_candidate_cut_plane(
    mesh_obj: vedo.Mesh,
    init_plane_origin: Sequence[float],
    init_plane_normal: Sequence[float],
    init_std=1.0,
    verbose=True,
) -> cma.CMAEvolutionStrategy:
    """use CMA to find the optimal direction of cut plane"""
    es = cma.CMAEvolutionStrategy(
        np.asarray(init_plane_normal), init_std, {"bounds": [-1.5, 1.5]}
    )
    es.opts.set(
        {"tolfunhist": 1e-1}
    )  # stop when the cross section area does not reduce by 0.1

    while not es.stop():
        solutions = es.ask()
        loss_values = [
            get_cross_section_area(
                mesh_obj, plane_normal=tuple(x), plane_origin=init_plane_origin
            )
            for x in solutions
        ]
        es.tell(solutions, loss_values)
        if verbose:
            es.disp()
    if verbose:
        es.result_pretty()
    return es


# does not work


def cma_es_search_minimal_width_neck(
    mesh_obj: vedo.Mesh,
    init_plane_origin: Sequence[float],
    init_plane_normal: Sequence[float],
    init_std=1.0,
    verbose=False,
):
    """search along the `plane_normal` to find the location of minimal neck width"""
    distance = 0.0
    starting_parameters = (*init_plane_normal, distance)
    # lower-bound and upper-bound the normals but not the distance
    es = cma.CMAEvolutionStrategy(
        np.asarray(starting_parameters),
        init_std,
        {"bounds": [[-1.5, -1.5, -1.5, None], [1.5, 1.5, 1.5, None]]},
    )
    es.opts.set({"tolfunhist": 1e-1})

    while not es.stop():
        solutions = es.ask()
        loss_values = []

        for x in solutions:
            nx, ny, nz, d = x
            plane_normal = tuple((nx, ny, nz))
            plane_origin = add_tuple(
                init_plane_origin, multiply_tuple_scalar(plane_normal, d)
            )
            loss_values.append(
                get_cross_section_area(
                    mesh_obj, plane_normal=plane_normal, plane_origin=plane_origin
                )
            )
        es.tell(solutions, loss_values)
        if verbose:
            es.disp()
            print(es.best.get())
    if verbose:
        es.result_pretty()
    return es


def grid_search_candidate_cut_plane(
    mesh_obj: vedo.Mesh,
    init_plane_origin: Sequence[float],
    init_plane_normal: Sequence[float],
    num_cuts=5,
    range_min=-0.5,
    range_max=0.5,
    verbose=False,
) -> Tuple[Sequence[float], float]:
    """grid search through candidate cut plane normals at given position to find the one with smallest cross-sectional area"""
    best_cut_plane_normal = tuple()
    smallest_csarea = sys.float_info.max
    for incr_k in np.linspace(range_min, range_max, num_cuts):
        for incr_i in np.linspace(range_min, range_max, num_cuts):
            for incr_j in np.linspace(range_min, range_max, num_cuts):
                candidate_cut_plane_normal = tuple(
                    v + inc
                    for v, inc in zip(init_plane_normal, (incr_i, incr_j, incr_k))
                )
                csa = get_cross_section_area(
                    mesh_obj,
                    plane_normal=candidate_cut_plane_normal,
                    plane_origin=init_plane_origin,
                )
                if csa <= smallest_csarea:
                    # additional sanity check: is the cut plane actually circular
                    # Cross section consistency check:
                    # 1. Circular fitting
                    # boundary_points = mesh_obj.clone().cut_with_plane(origin=init_plane_origin,normal=candidate_cut_plane_normal).boundaries().points()
                    # c,R,n = vedo.fit_circle(boundary_points)

                    if verbose:
                        print(f"found better candidate with cs-area {csa:.3f}")
                    smallest_csarea = csa
                    best_cut_plane_normal = candidate_cut_plane_normal
    return best_cut_plane_normal, smallest_csarea
