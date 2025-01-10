import os
import time
import numpy as np
import vtk
import argparse

parser = argparse.ArgumentParser(description="Swin-X2S reconstruction pipeline")
parser.add_argument("--file_path", default="./visualization/vis_result/models/CTPelvic1K", type=str,
                    help="directory to save the tensorboard logs")

colors_pal = [(0.4980392156862745, 0.23529411764705882, 0.5529411764705883),
              (0.06666666666666667, 0.6470588235294118, 0.4745098039215686),
              (0.2235294117647059, 0.4117647058823529, 0.6745098039215687),
              (0.9490196078431372, 0.7176470588235294, 0.00392156862745098),
              (0.9058823529411765, 0.24705882352941178, 0.4549019607843137),
              (0.5019607843137255, 0.7294117647058823, 0.35294117647058826),
              (0.9019607843137255, 0.5137254901960784, 0.06274509803921569),
              (0.0, 0.5254901960784314, 0.5843137254901961),
              (0.8117647058823529, 0.10980392156862745, 0.5647058823529412),
              (0.9764705882352941, 0.4823529411764706, 0.4470588235294118)]
colors_pal = colors_pal * 6
color_dict = {}
for i, c in enumerate(colors_pal):
    color_dict["Segment_" + str(i + 1)] = c


def segment_generator():
    args = parser.parse_args()
    files = os.listdir(args.file_path)

    for f in files:
        if f.find("nii.gz") != -1:
            raw_file = os.path.join(args.file_path, f)
            volume_node = slicer.util.loadVolume(raw_file)
            slicer.mrmlScene.RemoveNode(volume_node)

    for f in files:
        if f.find("nii.gz") == -1:
            nii_file = os.path.join(args.file_path, f)
            for n in os.listdir(nii_file):
                if n.find("label.nii.gz") != -1:
                    volume_node = slicer.util.loadVolume(raw_file)
                    volume_origin = volume_node.GetOrigin()
                    volume_node.SetOrigin(volume_origin[0] + 100, volume_origin[1], volume_origin[2])
                    yellow_slice_logic = slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic()
                    yellow_slice_logic.GetSliceNode().SetSliceVisible(True)
                    yellow_slice_logic.SetSliceOffset(-volume_origin[0] + 150)

                    seg_path = os.path.join(nii_file, n)
                    segmentation_node = slicer.util.loadSegmentation(seg_path)
                    segmentation_node.CreateClosedSurfaceRepresentation()
                    segmentation_node.GetSegmentation().SetConversionParameter("Smoothing factor", "0.5")
                    segmentation_display_node = segmentation_node.GetDisplayNode()
                    segmentation_display_node.SetOpacity3D(1)
                    segmentation_display_node.SetVisibility(1)
                    segmentation_display_node.SetVisibility3D(True)

                    for k in segmentation_node.GetSegmentation().GetSegmentIDs():
                        segmentation_node.GetSegmentation().GetSegment(k).SetColor(*color_dict[k])

                    view3D = slicer.app.layoutManager().threeDWidget(0).threeDView()
                    view3D.resetFocalPoint()
                    view3D.resetCamera()
                    view3D.rotateToViewAxis(0)
                    view3D.mrmlViewNode().SetBackgroundColor(0, 0, 0)
                    view3D.mrmlViewNode().SetBackgroundColor2(0, 0, 0)
                    view3D.mrmlViewNode().SetBoxVisible(False)
                    view3D.mrmlViewNode().SetAxisLabelsVisible(False)
                    view3D.forceRender()

                    renderWindow = view3D.renderWindow()
                    renderWindow.SetSize(1500, 1000)
                    renderWindow.SetAlphaBitPlanes(1)
                    wti = vtk.vtkWindowToImageFilter()
                    wti.SetInputBufferTypeToRGBA()
                    wti.SetInput(renderWindow)
                    writer = vtk.vtkPNGWriter()
                    writer.SetFileName(os.path.join(nii_file, "label_sag.png"))
                    writer.SetInputConnection(wti.GetOutputPort())
                    writer.Write()
                    volume_node.SetOrigin(volume_origin[0], volume_origin[1], volume_origin[2])
                    yellow_slice_logic.GetSliceNode().SetSliceVisible(False)
                    segmentation_node.GetDisplayNode().SetVisibility3D(False)
                    segmentation_node.GetDisplayNode().SetVisibility(0)
                    slicer.mrmlScene.RemoveNode(segmentation_node)
                    slicer.mrmlScene.RemoveNode(volume_node)


                    volume_node = slicer.util.loadVolume(raw_file)
                    volume_origin = volume_node.GetOrigin()
                    volume_node.SetOrigin(volume_origin[0], volume_origin[1] - 100, volume_origin[2])
                    green_slice_logic = slicer.app.layoutManager().sliceWidget('Green').sliceLogic()
                    green_slice_logic.GetSliceNode().SetSliceVisible(True)
                    green_slice_logic.SetSliceOffset(volume_origin[1] - 250)

                    seg_path = os.path.join(nii_file, n)
                    segmentation_node = slicer.util.loadSegmentation(seg_path)
                    segmentation_node.CreateClosedSurfaceRepresentation()
                    segmentation_node.GetSegmentation().SetConversionParameter("Smoothing factor", "0.5")
                    segmentation_display_node = segmentation_node.GetDisplayNode()
                    segmentation_display_node.SetOpacity3D(1)
                    segmentation_display_node.SetVisibility(1)
                    segmentation_display_node.SetVisibility3D(True)
                    for k in segmentation_node.GetSegmentation().GetSegmentIDs():
                        segmentation_node.GetSegmentation().GetSegment(k).SetColor(*color_dict[k])
                    view3D = slicer.app.layoutManager().threeDWidget(0).threeDView()
                    view3D.resetFocalPoint()
                    view3D.resetCamera()
                    view3D.rotateToViewAxis(3)
                    view3D.mrmlViewNode().SetBackgroundColor(0, 0, 0)
                    view3D.mrmlViewNode().SetBackgroundColor2(0, 0, 0)
                    view3D.mrmlViewNode().SetBoxVisible(False)
                    view3D.mrmlViewNode().SetAxisLabelsVisible(False)
                    view3D.forceRender()

                    renderWindow = view3D.renderWindow()
                    renderWindow.SetSize(1500, 1000)
                    renderWindow.SetAlphaBitPlanes(1)
                    wti = vtk.vtkWindowToImageFilter()
                    wti.SetInputBufferTypeToRGBA()
                    wti.SetInput(renderWindow)
                    writer = vtk.vtkPNGWriter()
                    writer.SetFileName(os.path.join(nii_file, "label_cor.png"))
                    writer.SetInputConnection(wti.GetOutputPort())
                    writer.Write()
                    volume_node.SetOrigin(volume_origin[0], volume_origin[1], volume_origin[2])
                    green_slice_logic.GetSliceNode().SetSliceVisible(False)
                    segmentation_node.GetDisplayNode().SetVisibility3D(False)
                    segmentation_node.GetDisplayNode().SetVisibility(0)
                    slicer.mrmlScene.RemoveNode(segmentation_node)
                    slicer.mrmlScene.RemoveNode(volume_node)

                if n.find("pred.nii.gz") != -1:
                    sag_volume_node_name = n.split("pred.nii.gz")[0] + "input_sag.nii.gz"
                    sag_volume_node = slicer.util.loadVolume(os.path.join(nii_file, sag_volume_node_name))
                    volume_origin = sag_volume_node.GetOrigin()
                    sag_volume_node.SetOrigin(volume_origin[0] + 100, volume_origin[1], volume_origin[2])
                    yellow_slice_logic = slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic()
                    yellow_slice_logic.GetSliceNode().SetSliceVisible(True)
                    yellow_slice_logic.SetSliceOffset(-volume_origin[0] + 150)

                    seg_path = os.path.join(nii_file, n)
                    segmentation_node = slicer.util.loadSegmentation(seg_path)
                    segmentation_node.CreateClosedSurfaceRepresentation()
                    segmentation_node.GetSegmentation().SetConversionParameter("Smoothing factor", "0.5")
                    segmentation_display_node = segmentation_node.GetDisplayNode()
                    segmentation_display_node.SetOpacity3D(1)
                    segmentation_display_node.SetVisibility(1)
                    segmentation_display_node.SetVisibility3D(True)
                    for k in segmentation_node.GetSegmentation().GetSegmentIDs():
                        segmentation_node.GetSegmentation().GetSegment(k).SetColor(*color_dict[k])
                    view3D = slicer.app.layoutManager().threeDWidget(0).threeDView()
                    view3D.resetFocalPoint()
                    view3D.resetCamera()
                    view3D.rotateToViewAxis(0)
                    view3D.mrmlViewNode().SetBackgroundColor(0, 0, 0)
                    view3D.mrmlViewNode().SetBackgroundColor2(0, 0, 0)
                    view3D.mrmlViewNode().SetBoxVisible(False)
                    view3D.mrmlViewNode().SetAxisLabelsVisible(False)
                    view3D.forceRender()

                    renderWindow = view3D.renderWindow()
                    renderWindow.SetSize(1500, 1000)
                    renderWindow.SetAlphaBitPlanes(1)
                    wti = vtk.vtkWindowToImageFilter()
                    wti.SetInputBufferTypeToRGBA()
                    wti.SetInput(renderWindow)
                    writer = vtk.vtkPNGWriter()
                    writer.SetFileName(os.path.join(nii_file, "pred_sag.png"))
                    writer.SetInputConnection(wti.GetOutputPort())
                    writer.Write()
                    sag_volume_node.SetOrigin(volume_origin[0], volume_origin[1], volume_origin[2])
                    yellow_slice_logic.GetSliceNode().SetSliceVisible(False)
                    segmentation_node.GetDisplayNode().SetVisibility3D(False)
                    segmentation_node.GetDisplayNode().SetVisibility(0)
                    slicer.mrmlScene.RemoveNode(segmentation_node)
                    slicer.mrmlScene.RemoveNode(sag_volume_node)


                    cor_volume_node_name = n.split("pred.nii.gz")[0] + "input_cor.nii.gz"
                    cor_volume_node = slicer.util.loadVolume(os.path.join(nii_file, cor_volume_node_name))
                    volume_origin = cor_volume_node.GetOrigin()
                    cor_volume_node.SetOrigin(volume_origin[0], volume_origin[1] - 100, volume_origin[2])
                    green_slice_logic = slicer.app.layoutManager().sliceWidget('Green').sliceLogic()
                    green_slice_logic.GetSliceNode().SetSliceVisible(True)
                    green_slice_logic.SetSliceOffset(volume_origin[1] - 250)

                    seg_path = os.path.join(nii_file, n)
                    segmentation_node = slicer.util.loadSegmentation(seg_path)
                    segmentation_node.CreateClosedSurfaceRepresentation()
                    segmentation_node.GetSegmentation().SetConversionParameter("Smoothing factor", "0.5")
                    segmentation_display_node = segmentation_node.GetDisplayNode()
                    segmentation_display_node.SetOpacity3D(1)
                    segmentation_display_node.SetVisibility(1)
                    segmentation_display_node.SetVisibility3D(True)
                    for k in segmentation_node.GetSegmentation().GetSegmentIDs():
                        segmentation_node.GetSegmentation().GetSegment(k).SetColor(*color_dict[k])
                    view3D = slicer.app.layoutManager().threeDWidget(0).threeDView()
                    view3D.resetFocalPoint()
                    view3D.resetCamera()
                    view3D.rotateToViewAxis(3)
                    view3D.mrmlViewNode().SetBackgroundColor(0, 0, 0)
                    view3D.mrmlViewNode().SetBackgroundColor2(0, 0, 0)
                    view3D.mrmlViewNode().SetBoxVisible(False)
                    view3D.mrmlViewNode().SetAxisLabelsVisible(False)
                    view3D.forceRender()

                    renderWindow = view3D.renderWindow()
                    renderWindow.SetSize(1500, 1000)
                    renderWindow.SetAlphaBitPlanes(1)
                    wti = vtk.vtkWindowToImageFilter()
                    wti.SetInputBufferTypeToRGBA()
                    wti.SetInput(renderWindow)
                    writer = vtk.vtkPNGWriter()
                    writer.SetFileName(os.path.join(nii_file, "pred_cor.png"))
                    writer.SetInputConnection(wti.GetOutputPort())
                    writer.Write()
                    cor_volume_node.SetOrigin(volume_origin[0], volume_origin[1], volume_origin[2])
                    green_slice_logic.GetSliceNode().SetSliceVisible(False)
                    segmentation_node.GetDisplayNode().SetVisibility3D(False)
                    segmentation_node.GetDisplayNode().SetVisibility(0)
                    slicer.mrmlScene.RemoveNode(segmentation_node)
                    slicer.mrmlScene.RemoveNode(cor_volume_node)


if __name__ == "__main__":
    segment_generator()
    # exec(open('./visualization/slicer_visualization_3d.py').read())
