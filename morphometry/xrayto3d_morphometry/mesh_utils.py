from typing import Tuple

import numpy as np
import SimpleITK as sitk
import vedo
from collections.abc import Iterable
from .sitk_utils import make_isotropic
from .geom_ops import lerp


def get_principal_axis(mesh_obj: vedo.Mesh) -> Tuple[np.ndarray, vedo.Ellipsoid]:
    mesh_axes: vedo.Ellipsoid = vedo.pca_ellipsoid(mesh_obj.points())

    ax1 = vedo.versor(mesh_axes.axis1)
    ax2 = vedo.versor(mesh_axes.axis2)
    ax3 = vedo.versor(mesh_axes.axis3)
    T = np.array([ax1, ax2, ax3])
    return T, mesh_axes


def align_along_principal_axes(mesh_obj) -> Tuple[vedo.Mesh, np.ndarray]:
    T, mesh_axis = get_principal_axis(mesh_obj)
    aligned_mesh_obj = mesh_obj.clone().apply_transform(T)

    return aligned_mesh_obj, T


def get_mesh_from_segmentation(filename: str, label_idx=None, largest_component=False, flying_edges=True, decimate=False, decimation_ratio=1.0, isosurface_value=1.0, smooth=20, reorient=False, orientation='PIR') -> vedo.Mesh:
    np_volume = get_volume(filename, label_idx, largest_component, reorient=reorient, orientation=orientation)
    if np_volume == 0: return 0
    mesh_obj: vedo.Mesh = np_volume.isosurface(value=isosurface_value-0.1, flying_edges=flying_edges)
    mesh_obj = mesh_obj.fill_holes()
    mesh_obj.smooth(niter=smooth)
    if decimate:
        mesh_obj = mesh_obj.decimate(fraction=decimation_ratio)
    return mesh_obj.cap()


def get_array_and_mesh_from_segmentation(filename: str, label_idx=None, largest_component=False, flying_edges=True, decimate=False, decimation_ratio=1.0, isosurface_value=1.0, smooth=20, reorient=False, orientation='PIR') -> (np.array, vedo.Mesh):
    np_array, np_volume = get_array_and_volume(filename, label_idx, largest_component, reorient=reorient, orientation=orientation)
    if np.sum(np_array) == 0: return 0, 0
    mesh_obj: vedo.Mesh = np_volume.isosurface(value=isosurface_value-0.1, flying_edges=flying_edges)
    mesh_obj = mesh_obj.fill_holes()
    mesh_obj.smooth(niter=smooth)
    if decimate:
        mesh_obj = mesh_obj.decimate(fraction=decimation_ratio)
    return np_array, mesh_obj.cap()


def get_volume(filename, label_idx=None, largest_component=False, isotropic=False, reorient=False, orientation='PIR') -> vedo.Volume:
    sitk_volume = sitk.ReadImage(filename)
    volume_array = sitk.GetArrayFromImage(sitk_volume)
    if label_idx is None:
        volume_array = np.array(volume_array)
    elif label_idx == 0:
        volume_array = np.where(volume_array != 0, 1, 0)
    else:
        if label_idx is Iterable:
            new_volume_array = np.zeros_like(volume_array)
            for i in label_idx:
                new_volume_array += np.where(volume_array == i, 1, 0)
            volume_array = new_volume_array
        else:
            volume_array = np.where(volume_array == label_idx, 1, 0)

    modified_sitk_volume = sitk.GetImageFromArray(volume_array)
    modified_sitk_volume.SetSpacing(sitk_volume.GetSpacing())
    modified_sitk_volume.SetOrigin(sitk_volume.GetOrigin())
    modified_sitk_volume.SetDirection(sitk_volume.GetDirection())

    if reorient:
        modified_sitk_volume = sitk.DICOMOrient(modified_sitk_volume, orientation)
    if largest_component:
        # get largest connected component
        modified_sitk_volume = sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.Cast(modified_sitk_volume, sitk.sitkUInt8),
        ), sortByObjectSize=True) == 1
    if isotropic:
        modified_sitk_volume = make_isotropic(modified_sitk_volume, 1.0)

    '''np_volume = vedo.Volume(sitk.GetArrayFromImage(modified_sitk_volume),
                            origin=sitk_volume.GetOrigin(),
                            spacing=sitk_volume.GetSpacing())'''
    np_volume = vedo.Volume(sitk.GetArrayFromImage(modified_sitk_volume))
    return np_volume


def get_array_and_volume(filename, label_idx=None, largest_component=False, isotropic=False, reorient=False, orientation='PIR') -> (np.array, vedo.Volume):
    sitk_volume = sitk.ReadImage(filename)
    volume_array = sitk.GetArrayFromImage(sitk_volume)
    if label_idx is None:
        volume_array = np.array(volume_array)
    elif label_idx == 0:
        volume_array = np.where(volume_array != 0, 1, 0)
    else:
        if isinstance(label_idx, Iterable):
            new_volume_array = np.zeros_like(volume_array, dtype=np.int32)
            for i in label_idx:
                new_volume_array += np.where(volume_array == i, 1, 0)
            volume_array = new_volume_array
        else:
            volume_array = np.where(volume_array == label_idx, 1, 0)

    modified_sitk_volume = sitk.GetImageFromArray(volume_array)
    modified_sitk_volume.SetSpacing(sitk_volume.GetSpacing())
    modified_sitk_volume.SetOrigin(sitk_volume.GetOrigin())
    modified_sitk_volume.SetDirection(sitk_volume.GetDirection())

    if reorient:
        modified_sitk_volume = sitk.DICOMOrient(modified_sitk_volume, orientation)
    if largest_component:
        # get largest connected component
        modified_sitk_volume = sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.Cast(modified_sitk_volume, sitk.sitkUInt8),
        ), sortByObjectSize=True) == 1
    if isotropic:
        modified_sitk_volume = make_isotropic(modified_sitk_volume, 1.0)

    np_array = sitk.GetArrayFromImage(modified_sitk_volume)
    '''np_volume = vedo.Volume(np_array,
                            origin=sitk_volume.GetOrigin(),
                            spacing=sitk_volume.GetSpacing())'''
    np_volume = vedo.Volume(np_array,)
    return volume_array, np_volume


def move_to(mesh_obj: vedo.Mesh, shift):
    """changes the original mesh so that its center of mass lies at (0,0,0)"""
    return mesh_obj.shift(*-shift)


def move_to_origin(mesh_obj: vedo.Mesh):
    """changes the original mesh so that its center of mass lies at (0,0,0)"""
    shift = mesh_obj.center_of_mass()
    return mesh_obj.shift(*-shift), shift


def get_pointcloud_from_mesh(mesh_obj: vedo.Mesh, label, label_name='scalars'):
    """get mesh vertices with specific labels"""
    point_cloud: np.ndarray = mesh_obj.clone(transformed=True).points()
    point_labels: np.ndarray = mesh_obj.pointdata[label_name]
    return vedo.Points(point_cloud[point_labels == label])


def get_symmetry_plane(mesh_obj, sampled_points=100):
    '''
    get symmetry plane by mirroring the mesh obj and then registering
    the vertices between the original and mirrored mesh obj.
    Take average direction between 100 random registered points by
    fitting a plane through these sampled points
    '''
    mirrored_vert_mesh = mesh_obj.clone(deep=True, transformed=True).mirror("x")
    mirrored_vert_points = vedo.Points(mirrored_vert_mesh.points())
    vert_mesh_points = vedo.Points(
        mesh_obj.clone(deep=True, transformed=True).points()
    )
    aligned_pts1 = mirrored_vert_points.clone().align_to(vert_mesh_points, invert=False)

    # draw arrows to see where points end up
    rand_idx = np.random.randint(0, len(mesh_obj.points()), sampled_points)
    sampled_vmp = mesh_obj.points()[rand_idx]
    sampled_apts1 = aligned_pts1.points()[rand_idx]
    avg_points = [lerp(a, b, 0.5) for a, b in zip(sampled_vmp, sampled_apts1)]
    sym_plane = vedo.fit_plane(avg_points, signed=True)
    return sym_plane
