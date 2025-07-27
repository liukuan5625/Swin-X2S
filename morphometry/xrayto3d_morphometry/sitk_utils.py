from typing import Optional

import SimpleITK as sitk
import numpy as np
import vedo


def get_segmentation_labels(segmentation: sitk.Image):
    """return label indexes"""
    fltr = get_segmentation_stats(segmentation)
    return fltr.GetLabels()


def get_segmentation_stats(
    segmentation: sitk.Image,
) -> sitk.LabelShapeStatisticsImageFilter:
    """return filter obj containing segmentation metadata"""
    fltr: sitk.LabelShapeStatisticsImageFilter = sitk.LabelShapeStatisticsImageFilter()
    fltr.ComputeOrientedBoundingBoxOn()
    fltr.ComputeFeretDiameterOn()
    fltr.Execute(sitk.Cast(segmentation, sitk.sitkUInt8))
    return fltr


def change_label(img: sitk.Image, mapping_dict) -> sitk.Image:
    """use SimplITK AggregateLabelMapFilter to merge all segmentation labels to first label. This is used to obtain the bounding box of all the labels"""
    fltr = sitk.ChangeLabelImageFilter()
    fltr.SetChangeMap(mapping_dict)
    return fltr.Execute(sitk.Cast(img, sitk.sitkUInt8))


def get_segmentation_volume(
    filename, label_id, largest_component=False, isotropic=False
):
    sitk_volume = sitk.ReadImage(filename)
    numpy_array = sitk.GetArrayFromImage(sitk_volume)
    if label_id not in np.unique(numpy_array):
        return
    if isotropic:
        sitk_volume = make_isotropic(sitk_volume, spacing=1.0)
    labels = get_segmentation_labels(sitk_volume)

    # get segmentation volume with specific label id
    change_map_dict = {}
    for l in labels:
        if l == label_id:
            change_map_dict[l] = 1
        else:
            change_map_dict[l] = 0
    sitk_volume = change_label(sitk_volume, change_map_dict)

    if largest_component:
        # get largest connected component
        sitk_volume = (
            sitk.RelabelComponent(
                sitk.ConnectedComponent(
                    sitk.Cast(sitk_volume, sitk.sitkUInt8),
                ),
                sortByObjectSize=True,
            )
            == 1
        )
    '''return vedo.Volume(sitk.GetArrayFromImage(sitk_volume),
                       spacing=sitk_volume.GetSpacing(),
                       origin=sitk_volume.GetOrigin())'''
    return vedo.Volume(sitk.GetArrayFromImage(sitk_volume))


def make_isotropic(
    img: sitk.Image,
    spacing: Optional[float] = None,
    interpolator=sitk.sitkNearestNeighbor,
):
    """
    Resample `img` so that the voxel is isotropic with given physical spacing
    The image volume is shrunk or expanded as necessary to represent the same physical space.
    Use sitk.sitkNearestNeighbour while resampling Label images,
    when spacing is not supplied by the user, the highest resolution axis spacing is used
    """
    # keep the same physical space, size may shrink or expand

    if spacing is None:
        spacing = min(list(img.GetSpacing()))

    resampler = sitk.ResampleImageFilter()
    new_size = [
        round(old_size * old_spacing / spacing)
        for old_size, old_spacing in zip(img.GetSize(), img.GetSpacing())
    ]
    output_spacing = [spacing] * len(img.GetSpacing())

    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    # resampler.SetDefaultPixelValue(img.GetPixelIDValue())
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(img)
