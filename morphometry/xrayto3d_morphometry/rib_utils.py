import os
import collections
from collections import namedtuple
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.spatial import KDTree
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import h5py
import kimimaro
import cloudvolume as cv
from skimage.morphology import skeletonize, skeletonize_3d, binary_dilation, ball


FakeSkeleton = namedtuple(
    "FakeSkeleton", ["vertices", "edges", "radii", "vertex_types", "ind_map"]
)


def skeletonize_around_pt(cvol, pt, segid, kimi_params, bbox_width=(125, 125, 100)):

    bbox = u.make_bbox([pt], bbox_width)

    seg = cvol[
        bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1], bbox[0][2] : bbox[1][2]
    ]

    return skeletonize_seg(seg, segid, kimi_params)


def skeletonize_seg(seg, segid, kimi_params):
    return kimimaro.skeletonize(seg, object_ids=[segid], **kimi_params)[segid]


def label_skeleton(
    skel,
    count_thr=3,
    dist_thr=200,
    radius_buffer=50,
    path_inds=None,
    path_radii=None,
    root=None,
):
    """
    Labels the spines of p_A skeleton using (mostly) the number of
    paths passing through each node. Performs other post-processing
    of p_A hard threshold to try to reach the base of each spine.
    Label 1 => dendritic spine
    Label 0 => dendritic shaft
    """
    if (path_inds is None) or (path_radii is None):
        path_inds, path_radii = compute_path_info(skel, root=root)
    path_counts = node_path_counts(path_inds)

    num_nodes = len(skel.vertices)
    labels = np.ones((num_nodes,), dtype=np.uint8)
    # Labeling an initial set of shaft seeds
    labels[path_counts > count_thr] = 0

    dists, seed_inds = path_distance_transform(path_inds, labels)

    to_label = (dists < dist_thr) & (skel.radii > skel.radii[seed_inds] - radius_buffer)
    prelim_labels = labels.copy()
    prelim_labels[to_label] = 0

    return clean_prelim_labels(path_inds, prelim_labels, labels)


def label_skeleton_pathcount(skel, count_thr=6, root=None):
    """
    Labels the spines of p_A skeleton using (solely) the number
    of paths passing through each node.
    """
    path_inds, path_radii = compute_path_info(skel, root=root)
    num_nodes = len(skel.vertices)
    counts = node_path_counts(path_inds, num_nodes)

    labels = np.ones((num_nodes,), dtype=np.uint8)
    labels[counts > count_thr] = 0
    # removing nodes from p_A root (these have been p_A spine in the past)
    # labels[counts == counts.max()] = 1

    return labels


def extract_spine_near_pts(
    skel,
    labels,
    pts,
    path_inds=None,
    root=None,
    cv_skel=True,
    return_all_paths=False,
    kdt=None,
):
    """
    Extracts the stretches of spine-labeled skeleton closest to p_A given
    list of landmark points (often synapses).
    """
    if path_inds is None:
        path_inds, _ = compute_path_info(skel, root=root, cv_skel=cv_skel)

    if kdt is None:
        kdt = KDTree(skel.vertices)

    spines = list()
    for pt in pts:
        closest_node = find_closest_node(skel, pt, kdtree=kdt)
        path = find_containing_paths(path_inds, closest_node, max_len=True)
        spine, root = extract_spine_from_path(path, labels, closest_node)
        all_prongs = find_containing_paths(path_inds, root)
        if len(all_prongs) > 1 and return_all_paths:
            # multi-pronged spine
            spine_prongs = [
                extract_spine_from_path(prong, labels, root)[0] for prong in all_prongs
            ]
            spine = np.unique(np.hstack(spine_prongs))

        spines.append((spine, root))

    return spines


def extract_spine_by_node(
    skel, labels, nodes, path_inds=None, root=None, cv_skel=True, return_all_paths=False
):
    if path_inds is None:
        path_inds, _ = compute_path_info(skel, root=root, cv_skel=cv_skel)

    spines = list()
    for node in nodes:
        path = find_containing_paths(path_inds, node, max_len=True)
        spine, root = extract_spine_from_path(path, labels, node)
        all_prongs = find_containing_paths(path_inds, root)
        if len(all_prongs) > 1 and return_all_paths:
            # multi-pronged spine
            spine_prongs = [
                extract_spine_from_path(prong, labels, root)[0] for prong in all_prongs
            ]
            spine = np.unique(np.hstack(spine_prongs))

        spines.append((spine, root))

    return spines


def compute_path_info(skel, root=None, cv_skel=True):
    path_inds = paths(skel, root=root, cv_skel=cv_skel)
    path_radii = [skel.radii[inds] for inds in path_inds]

    return path_inds, path_radii


def skel_by_inds(skel, inds):
    assert len(inds) > 0, "empty inds"
    new_verts = skel.vertices[inds]
    new_vtypes = skel.vertex_types[inds]
    new_radii = skel.radii[inds]

    ind_map = np.empty((max(inds) + 1,), dtype=inds.dtype)
    ind_map[inds] = np.arange(len(inds))
    edge_inds = np.all(np.isin(skel.edges, inds), axis=1)
    new_edges = ind_map[skel.edges[edge_inds]]

    return FakeSkeleton(new_verts, new_edges, new_radii, new_vtypes, ind_map)


def translate_skel_ids(skel, ids):
    """
    Maps vertex ids to those of p_A view (FakeSkeleton)
    """
    return skel.ind_map[ids]


def paths_containing_pair(skel, root, other, single=False, max_len=False, cv_skel=True):
    path_inds = paths(skel, root=root, return_inds=True, cv_skel=cv_skel)
    cpaths = [inds for inds in path_inds if np.all(np.isin((root, other), inds))]

    if single:
        assert len(cpaths) == 1, "both nodes contained in multiple paths"
        return cpaths[0]

    if max_len:
        return max(cpaths, key=len)

    return cpaths


def find_furthest_pt(skel, root, single=True):
    num_nodes = len(skel.vertices)
    edges = skel.edges
    g = sp.coo_matrix(
        (
            np.ones(
                len(edges),
            ),
            (edges[:, 0], edges[:, 1]),
        ),
        shape=(num_nodes, num_nodes),
    )
    o = csgraph.breadth_first_order(g, root, directed=False, return_predecessors=False)

    furthest_node = o[-1]

    o2, preds = csgraph.breadth_first_order(
        g, furthest_node, directed=False, return_predecessors=True
    )

    path_inds = reconstruct_all_paths(preds)
    paths = [inds for inds in path_inds if root in inds]

    if single:
        assert len(paths) == 1, "Too many paths"
        return furthest_node, paths[0]

    else:
        return furthest_node, paths


def medium_radius_cut_pt(skel, path_inds, simplest_vers=True):
    radii = skel.radii[path_inds]
    half_len = len(path_inds) // 2
    min_close_rad_i = np.argmin(radii[:half_len])
    max_far_rad_i = np.argmax(radii[half_len:]) + half_len

    if simplest_vers:
        local_cut_i = (min_close_rad_i + max_far_rad_i) // 2
        return path_inds[local_cut_i], local_cut_i

    linear_dist = (radii[min_close_rad_i:max_far_rad_i] - radii[min_close_rad_i]) / (
        radii[max_far_rad_i] - radii[min_close_rad_i]
    )

    # shifting window to acct for large bumps (e.g. multi-headed spines)
    min_i = np.max(np.nonzero(linear_dist < 1 / 5)[0], initial=0)

    try:
        local_cut_i = (
            np.nonzero(linear_dist[min_i:] > 1 / 3)[0].min() + min_close_rad_i + min_i
        )
    except Exception:
        local_cut_i = min_close_rad_i + min_i

    return path_inds[local_cut_i], local_cut_i


def reconstruct_all_paths(preds):
    leaves = np.nonzero(~np.isin(np.arange(len(preds)), preds))[0]
    return [reconstruct_path(preds, leaf) for leaf in leaves]


def reconstruct_path(preds, leaf):
    path = []
    curr = leaf
    while curr >= 0:
        path.append(curr)
        curr = preds[curr]
    return path


def node_path_counts(path_inds, num_nodes=None):
    if num_nodes is None:
        num_nodes = max(map(max, path_inds)) + 1

    counts = np.zeros((num_nodes,), dtype=np.uint16)

    for inds in path_inds:
        counts[inds] += 1

    return counts


def paths(skel, root=None, return_inds=True, cv_skel=True):

    if cv_skel:
        paths = []
        for tree in skel.components():
            paths += single_tree_paths(skel, tree, root=root, return_inds=return_inds)
    else:
        # FakeSkeleton, assumed to be single tree
        paths = single_tree_paths(
            skel, skel, root=root, cv_skel=False, return_inds=return_inds
        )
    return paths


def single_tree_paths(skel, tree, root=None, return_inds=True, cv_skel=True):

    if cv_skel:
        tree = tree.consolidate()

    if root is not None:
        # check whether this node exists within the tree
        if not isinstance(root, collections.Iterable):
            root = tuple(skel.vertices[root])

        tree_lookup = {tuple(v): i for (i, v) in enumerate(tree.vertices)}
        # If it's not in the tree, nullify it
        root = tree_lookup.get(root, None)

    edges = tree.edges
    num_nodes = edges.max() + 1
    g = sp.coo_matrix(
        (
            np.ones(
                len(edges),
            ),
            (edges[:, 0], edges[:, 1]),
        ),
        shape=(num_nodes, num_nodes),
    )

    def dfs_paths(g, root):
        o, preds = csgraph.depth_first_order(
            g, root, directed=False, return_predecessors=True
        )
        return reconstruct_all_paths(preds)

    if root is None:
        init_paths = dfs_paths(g, edges[0, 0])
        root_path = np.argmax([len(p) for p in init_paths])
        root = init_paths[root_path][0]

    tree_paths = dfs_paths(g, root)
    path_vertices = [tree.vertices[path] for path in tree_paths]

    if return_inds:
        # Have to do this since the consolidated tree inds =/= global inds
        vertex_lookup = {tuple(v): i for (i, v) in enumerate(skel.vertices)}
        return [
            np.array([vertex_lookup[tuple(v)] for v in single_path_vertices])
            for single_path_vertices in path_vertices
        ]

    else:
        return path_vertices


def path_distance_transform(path_inds, all_labels):

    distances = np.ones(all_labels.shape) * np.inf
    # init seeds to identity except non-zero labels
    seed_inds = np.arange(len(all_labels))
    seed_inds[all_labels != 0] = 0

    for path in path_inds:
        path_dists = all_labels[path]
        if np.all(path_dists == 1):
            continue
        dists, inds = distance_transform_edt(
            path_dists, return_distances=True, return_indices=True
        )
        # inds returned as 2d for some reason
        inds = inds[0, :]
        to_change = dists < distances[path]
        inds_to_change = path[to_change]

        distances[inds_to_change] = dists[to_change]
        seed_inds[inds_to_change] = path[inds[to_change]]

    return distances, seed_inds


def clean_prelim_labels(path_inds, prelim_labels, labels):
    for (i, p) in enumerate(path_inds):
        p_seeds = labels[p]
        p_expanded = prelim_labels[p]

        seed_inds = np.nonzero(p_seeds == 0)[0]
        if seed_inds.size == 0:
            continue

        if seed_inds[0] != 0:
            spines_before_seed = np.nonzero(p_expanded[: seed_inds[0]])[0]
            if spines_before_seed.size > 0:
                last_spine_before_seed = spines_before_seed[-1]
                prelim_labels[p[:last_spine_before_seed]] = 1

        if seed_inds[-1] != len(p) - 1:
            spines_after_seed = (
                np.nonzero(p_expanded[seed_inds[-1] :])[0] + seed_inds[-1]
            )
            if spines_after_seed.size > 0:
                first_spine_after_seed = spines_after_seed[0]
                prelim_labels[p[first_spine_after_seed:]] = 1

    return prelim_labels


def find_closest_node(skel, pt, kdtree=None):

    if kdtree is None:
        kdtree = KDTree(skel.vertices)

    return kdtree.query(pt)[1]


def find_containing_paths(
    path_inds, closest_node, single=False, max_len=False, min_len=False
):
    included_paths = [inds for inds in path_inds if closest_node in inds]

    if single:
        assert len(included_paths) == 1, (
            f"Node {closest_node} included" " in more than one path"
        )
        return included_paths[0]

    if max_len:
        return max(included_paths, key=len)

    if min_len:
        return min(included_paths, key=len)

    return included_paths


def extract_spine_from_path(path, labels, included_node):
    path_labels = labels[path]

    assert labels[included_node] == 1, f"Node {included_node} labeled" " as non-spine"
    assert included_node in path, f"Node {included_node} not in path"

    start = np.nonzero(path == included_node)[0][0]
    i = j = start
    while i > 0 and path_labels[i - 1] == 1:
        i -= 1
    while j < len(path_labels) and path_labels[j] == 1:
        j += 1

    if i > 0 and path_labels[i - 1] == 0:
        return path[i:j], path[i]
    if j < len(path_labels) and path_labels[j] == 0:
        return path[i:j], path[j - 1]
    # Entire path is included
    return path, path[0]


def write_skel(skel, filename):
    if os.path.isfile(filename):
        os.remove(filename)

    f = h5py.File(filename, "w")
    f.create_dataset("vertices", data=skel.vertices)
    f.create_dataset("edges", data=skel.edges)
    f.create_dataset("radii", data=skel.radii)
    f.create_dataset("vtypes", data=skel.vertex_types)

    return f


def read_skel(filename):
    assert os.path.isfile(filename)

    with h5py.File(filename) as f:
        vertices = f["vertices"][()]
        edges = f["edges"][()]
        radii = f["radii"][()]
        vtypes = f["vtypes"][()]

    return cv.PrecomputedSkeleton(vertices, edges, radii, vtypes, 99999)


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    if tprec + tsens == 0:
        return 0
    return 2*tprec*tsens/(tprec+tsens)


def Normalized_Line_Dice(pred_cl, label_cl):
    pred_cl_image = np.zeros([192, 192, 192])
    pred_cl_index = np.int16(np.round(pred_cl))
    for (x, y, z) in pred_cl_index:
        pred_cl_image[x, y, z] = 1
    pred_dilated_cl_image = binary_dilation(pred_cl_image, ball(3))

    label_cl_image = np.zeros([192, 192, 192])
    label_cl_index = np.int16(np.round(label_cl))
    for (x, y, z) in label_cl_index:
        label_cl_image[x, y, z] = 1
    label_dilated_cl_image = binary_dilation(label_cl_image, ball(3))

    top = np.sum(pred_dilated_cl_image * label_cl_image) + np.sum(label_dilated_cl_image * pred_cl_image)
    bottom = np.sum(label_cl_image) + np.sum(pred_cl_image)
    return top / bottom


def chamfer_distance(p_A, p_B):
    dist_a_to_b = np.linalg.norm(p_A[:, np.newaxis] - p_B, axis=2)  # (N, M)
    dist_b_to_a = np.linalg.norm(p_B[:, np.newaxis] - p_A, axis=2)  # (M, N)
    min_dist_a_to_b = np.min(dist_a_to_b, axis=1)  # (N,)
    min_dist_b_to_a = np.min(dist_b_to_a, axis=1)  # (M,)
    return np.mean(min_dist_a_to_b) + np.mean(min_dist_b_to_a)


def Line_Seg_Chamfer_DistanceError(pred_array, pred_cl, label_array, label_cl):
    pred_index = np.array(np.nonzero(pred_array)).T
    label_index = np.array(np.nonzero(label_array)).T
    pl = chamfer_distance(pred_cl, label_index)
    ll = chamfer_distance(label_cl, label_index)
    return abs(pl - ll) / ll