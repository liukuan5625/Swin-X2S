/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCenteredEuler3DTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkThresholdImageFilter.h"
#include "itkRayCastInterpolateImageFunction.h"
#include "itkMinimumMaximumImageCalculator.h"


void
usage()
{
    std::cerr << "\n";
    std::cerr << "Usage: DRR <options> [input]\n";
    std::cerr << "  calculates the Digitally Reconstructed Radiograph from a "
        "volume. \n\n";
    std::cerr << " where <options> is one or more of the following:\n\n";
    std::cerr << "  <-h>                    Display (this) usage information\n";
    std::cerr << "  <-res float float>      Pixel spacing of the output image "
        "[default: "
        "1x1mm]  \n";
    std::cerr << "  <-size int int>         Dimension of the output image "
        "[default: 501x501]  \n";
    std::cerr << "  <-sid float>            Distance of ray source (focal point) "
        "[default: 2000mm]\n";
    std::cerr << "  <-rx float>             Rotation around x,y,z axis in degrees \n";
    std::cerr << "  <-ry float>\n";
    std::cerr << "  <-rz float>\n";
    std::cerr << "  <-normal float float>   The 2D projection normal position "
        "[default: 0x0mm]\n";
    std::cerr << "  <-threshold float>      Threshold [default: 0]\n";
    std::cerr << "  <-o file>               Output image filename\n\n";

    std::cerr << "  <-ct_res float float float>      Pixel spacing of the output image "
        "[default: "
        "1x1x1mm]  \n";
    std::cerr << "  <-ct_size int int int>         Dimension of the output ct image "
        "[default: 501x501x501]  \n";
    std::cerr << "  <-ct_o file>               Output image filename  \n";
    std::cerr << "  <-is_seg_ct file>          Is segment CT ? 0 means is raw CT , else "
        "is segment CT  \n\n";

    std::cerr << "                          by  thomas@hartkens.de\n";
    std::cerr << "                          and john.hipwell@kcl.ac.uk (CISG "
        "London)\n\n";
    exit(1);
}

int
main(int argc, char* argv[])
{
    char* input_name = nullptr;
    char* output_name = nullptr;
    char* ct_output_name = nullptr;

    bool ok;
    bool is_seg_ct = 0;

    float rx = 0.;
    float ry = 0.;
    float rz = 0.;

    float sid = 2000.;

    float sxyz = 1.;

    float ct_sxyz = 1.;

    int dx = 501;
    int dy = 501;

    int ct_dx = 501;
    int ct_dy = 501;
    int ct_dz = 501;

    float o2Dx = 0;
    float o2Dy = 0;

    double threshold = 0;

    // Parse command line parameters

    while (argc > 1)
    {
        ok = false;

        if ((ok == false) && (strcmp(argv[1], "-h") == 0))
        {
            argc--;
            argv++;
            ok = true;
            usage();
        }

        if ((ok == false) && (strcmp(argv[1], "-rx") == 0))
        {
            argc--;
            argv++;
            ok = true;
            rx = std::stod(argv[1]);
            argc--;
            argv++;
        }

        if ((ok == false) && (strcmp(argv[1], "-ry") == 0))
        {
            argc--;
            argv++;
            ok = true;
            ry = std::stod(argv[1]);
            argc--;
            argv++;
        }

        if ((ok == false) && (strcmp(argv[1], "-rz") == 0))
        {
            argc--;
            argv++;
            ok = true;
            rz = std::stod(argv[1]);
            argc--;
            argv++;
        }

        if ((ok == false) && (strcmp(argv[1], "-threshold") == 0))
        {
            argc--;
            argv++;
            ok = true;
            threshold = std::stod(argv[1]);
            argc--;
            argv++;
        }

        if ((ok == false) && (strcmp(argv[1], "-res") == 0))
        {
            argc--;
            argv++;
            ok = true;
            sxyz = std::stod(argv[1]);
            argc--;
            argv++;
        }

        if ((ok == false) && (strcmp(argv[1], "-size") == 0))
        {
            argc--;
            argv++;
            ok = true;
            dx = std::stoi(argv[1]);
            argc--;
            argv++;
            dy = std::stoi(argv[1]);
            argc--;
            argv++;
        }

        if ((ok == false) && (strcmp(argv[1], "-sid") == 0))
        {
            argc--;
            argv++;
            ok = true;
            sid = std::stod(argv[1]);
            argc--;
            argv++;
        }

        if ((ok == false) && (strcmp(argv[1], "-normal") == 0))
        {
            argc--;
            argv++;
            ok = true;
            o2Dx = std::stod(argv[1]);
            argc--;
            argv++;
            o2Dy = std::stod(argv[1]);
            argc--;
            argv++;
        }


        if ((ok == false) && (strcmp(argv[1], "-o") == 0))
        {
            argc--;
            argv++;
            ok = true;
            output_name = argv[1];
            argc--;
            argv++;
        }

        if ((ok == false) && (strcmp(argv[1], "-ct_size") == 0))
        {
            argc--;
            argv++;
            ok = true;
            ct_dx = std::stoi(argv[1]);
            argc--;
            argv++;
            ct_dy = std::stoi(argv[1]);
            argc--;
            argv++;
            ct_dz = std::stoi(argv[1]);
            argc--;
            argv++;
        }

        if ((ok == false) && (strcmp(argv[1], "-ct_res") == 0))
        {
            argc--;
            argv++;
            ok = true;
            ct_sxyz = std::stod(argv[1]);
            argc--;
            argv++;
        }

        if ((ok == false) && (strcmp(argv[1], "-ct_o") == 0))
        {
            argc--;
            argv++;
            ok = true;
            ct_output_name = argv[1];
            argc--;
            argv++;
        }

        if ((ok == false) && (strcmp(argv[1], "-is_seg_ct") == 0))
        {
            argc--;
            argv++;
            ok = true;
            is_seg_ct = std::stoi(argv[1]);
            argc--;
            argv++;
        }

        if (ok == false)
        {

            if (input_name == nullptr)
            {
                input_name = argv[1];
                argc--;
                argv++;
            }

            else
            {
                std::cerr << "ERROR: Can not parse argument " << argv[1] << std::endl;
                usage();
            }
        }
    }


    sid = -1 * sid;
    constexpr unsigned int Dimension = 3;
    using InputPixelType = short;
    using OutputPixelType = unsigned char;

    using InputImageType = itk::Image<InputPixelType, Dimension>;
    using OutputImageType = itk::Image<OutputPixelType, Dimension>;

    InputImageType::Pointer image;
    InputImageType::Pointer ct_image;
    if (input_name)
    {
        using ReaderType = itk::ImageFileReader<InputImageType>;
        auto reader = ReaderType::New();
        reader->SetFileName(input_name);

        try
        {
            reader->Update();
        }
        catch (const itk::ExceptionObject& err)
        {
            std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
            return EXIT_FAILURE;
        }

        image = reader->GetOutput();
        ct_image = reader->GetOutput();
    }
    else
    {
        std::cout << "ERROR: No input given !" << std::endl;
        return EXIT_FAILURE;
    }


    using FilterType = itk::ResampleImageFilter<InputImageType, InputImageType>;

    auto filter = FilterType::New();
    filter->SetInput(image);
    filter->SetDefaultPixelValue(-1024);

    auto ct_filter = FilterType::New();
    ct_filter->SetInput(ct_image);
    ct_filter->SetDefaultPixelValue(-1024);

    if (is_seg_ct)
    {
        using ThresholdImageFilter = itk::ThresholdImageFilter<InputImageType>;
        auto thresholdfilter = ThresholdImageFilter::New();
        thresholdfilter->SetInput(image);
        thresholdfilter->SetUpper(0.1);
        thresholdfilter->SetOutsideValue(255);
        thresholdfilter->Update();

        image = thresholdfilter->GetOutput();
        filter->SetInput(image);
        filter->SetDefaultPixelValue(0);


        using ThresholdImageFilter = itk::ThresholdImageFilter<InputImageType>;
        auto ct_thresholdfilter = ThresholdImageFilter::New();
        ct_thresholdfilter->SetInput(ct_image);
        ct_thresholdfilter->ThresholdOutside(0, 55);
        ct_thresholdfilter->SetOutsideValue(0);
        ct_thresholdfilter->Update();

        ct_image = ct_thresholdfilter->GetOutput();
        ct_filter->SetInput(ct_image);
        ct_filter->SetDefaultPixelValue(0);
    }

    using ImageCalculatorFilterType = itk::MinimumMaximumImageCalculator<InputImageType>;
    auto imageCalculatorFilter = ImageCalculatorFilterType::New();
    imageCalculatorFilter->SetImage(ct_image);
    imageCalculatorFilter->Compute();
    short min_Val = imageCalculatorFilter->GetMinimum();
    short max_Val = imageCalculatorFilter->GetMaximum();


    // ct rotation transform
    using TransformType = itk::CenteredEuler3DTransform<double>;

    auto transform = TransformType::New();
    auto transform_rotx = TransformType::New();
    auto transform_roty = TransformType::New();
    auto transform_rotz = TransformType::New();

    const double dtr = (std::atan(1.0) * 4.0) / 180.0;
    transform_rotx->SetRotation(dtr * rx, 0, 0);
    transform_roty->SetRotation(0, dtr * ry, 0);
    transform_rotz->SetRotation(0, 0, dtr * rz);

    TransformType::MatrixType final_mat;
    final_mat = transform_rotx->GetMatrix() * transform_roty->GetMatrix() * transform_rotz->GetMatrix();

    InputImageType::PointType   imOrigin = image->GetOrigin();
    InputImageType::SpacingType imRes = image->GetSpacing();

    using InputImageRegionType = InputImageType::RegionType;
    using InputImageSizeType = InputImageRegionType::SizeType;

    InputImageRegionType imRegion = image->GetBufferedRegion();
    InputImageSizeType   imSize = imRegion.GetSize();

    imOrigin[0] += imRes[0] * static_cast<double>(imSize[0]) / 2.0;
    imOrigin[1] += imRes[1] * static_cast<double>(imSize[1]) / 2.0;
    imOrigin[2] += imRes[2] * static_cast<double>(imSize[2]) / 2.0;


    TransformType::InputPointType center;
    center[0] = imOrigin[0];
    center[1] = imOrigin[1];
    center[2] = imOrigin[2];

    transform->SetMatrix(final_mat);
    transform->SetCenter(center);


    std::vector<float> xyz(3);
    xyz[0] = imRes[0] * static_cast<float>(imSize[0]);
    xyz[1] = imRes[1] * static_cast<float>(imSize[1]);
    xyz[2] = imRes[2] * static_cast<float>(imSize[2]);
    std::sort(xyz.begin(), xyz.end());

    int ratio = xyz[2] / xyz[1] / 0.5;
    dy = dy * 0.5 * ratio;
    ct_dz = ct_dz * 0.5 * ratio;
    sxyz = 2 * xyz[2] / static_cast<float>(dy);
    sxyz = xyz[2] / static_cast<float>(dy);
    ct_sxyz = xyz[2] / static_cast<float>(ct_dz);

    InputImageRegionType      ct_imRegion = image->GetLargestPossibleRegion();
    InputImageSizeType        ct_imSize = ct_imRegion.GetSize();
    InputImageType::IndexType ct_centerIndex;
    ct_centerIndex[0] = ct_imSize[0] / 2;
    ct_centerIndex[1] = ct_imSize[1] / 2;
    ct_centerIndex[2] = ct_imSize[2] / 2;
    InputImageType::PointType ct_center;
    image->TransformIndexToPhysicalPoint(ct_centerIndex, ct_center);


    using InterpolatorType = itk::RayCastInterpolateImageFunction<InputImageType, double>;

    auto interpolator = InterpolatorType::New();
    interpolator->SetTransform(transform);
    // interpolator->SetThreshold(threshold);
    InterpolatorType::InputPointType focalpoint;
    focalpoint[0] = imOrigin[0];
    focalpoint[1] = imOrigin[1];
    focalpoint[2] = imOrigin[2] - sid * 9 / 10.;
    interpolator->SetFocalPoint(focalpoint);
    filter->SetInterpolator(interpolator);
    filter->SetTransform(transform);


    if (is_seg_ct)
    {
        using ct_InterpolatorType = itk::NearestNeighborInterpolateImageFunction<InputImageType, double>;

        auto ct_interpolator = ct_InterpolatorType::New();
        ct_filter->SetInterpolator(ct_interpolator);
    }

    InputImageType::SizeType size;
    size[0] = dx; // number of pixels along X of the 2D DRR image
    size[1] = dy; // number of pixels along Y of the 2D DRR image
    size[2] = 1;  // only one slice
    filter->SetSize(size);

    InputImageType::SizeType ct_size;
    ct_size[0] = ct_dx; // number of pixels along X of the 2D DRR image
    ct_size[1] = ct_dy; // number of pixels along Y of the 2D DRR image
    ct_size[2] = ct_dz; // only one slice
    ct_filter->SetSize(ct_size);

    InputImageType::SpacingType spacing;
    spacing[0] = sxyz; // pixel spacing along X of the 2D DRR image [mm]
    spacing[1] = sxyz; // pixel spacing along Y of the 2D DRR image [mm]
    spacing[2] = 1.0;  // slice thickness of the 2D DRR image [mm]
    filter->SetOutputSpacing(spacing);

    InputImageType::SpacingType ct_spacing;
    ct_spacing[0] = ct_sxyz; // pixel spacing along X of the 2D DRR image [mm]
    ct_spacing[1] = ct_sxyz; // pixel spacing along Y of the 2D DRR image [mm]
    ct_spacing[2] = ct_sxyz; // slice thickness of the 2D DRR image [mm]
    ct_filter->SetOutputSpacing(ct_spacing);


    double origin[Dimension];
    origin[0] = center[0] + o2Dx - sxyz * (static_cast<double>(dx) - 1.) / 2.;
    origin[1] = center[1] + o2Dy - sxyz * (static_cast<double>(dy) - 1.) / 2.;
    origin[2] = center[2] + sid / 10.;
    filter->SetOutputOrigin(origin);
    filter->Update();

    double ct_origin[Dimension];
    ct_origin[0] = ct_center[0] - ct_sxyz * (static_cast<double>(ct_dx) - 1.) / 2.;
    ct_origin[1] = ct_center[1] - ct_sxyz * (static_cast<double>(ct_dy) - 1.) / 2.;
    ct_origin[2] = ct_center[2] - ct_sxyz * (static_cast<double>(ct_dz) - 1.) / 2.;
    ct_filter->SetOutputOrigin(ct_origin);
    ct_filter->Update();


    // create writer
    using RescaleFilterType = itk::RescaleIntensityImageFilter<InputImageType, OutputImageType>;
    using WriterType = itk::ImageFileWriter<OutputImageType>;

    if (output_name)
    {
        auto rescaler = RescaleFilterType::New();
        rescaler->SetOutputMinimum(0);
        rescaler->SetOutputMaximum(255);
        /*if (is_seg_ct)
        {
          rescaler->SetOutputMinimum(min_Val);
          rescaler->SetOutputMaximum(max_Val);
        }*/
        rescaler->SetInput(filter->GetOutput());
        rescaler->Update();

        auto writer = WriterType::New();
        writer->SetFileName(output_name);
        writer->SetInput(rescaler->GetOutput());

        try
        {
            std::cout << "Writing image: " << output_name << std::endl;
            writer->Update();
        }
        catch (const itk::ExceptionObject& err)
        {
            std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
        }
    }
    else
    {
        filter->Update();
    }


    if (ct_output_name)
    {
        auto ct_rescaler = RescaleFilterType::New();
        ct_rescaler->SetOutputMinimum(0);
        ct_rescaler->SetOutputMaximum(255);
        if (is_seg_ct)
        {
            ct_rescaler->SetOutputMinimum(min_Val);
            ct_rescaler->SetOutputMaximum(max_Val);
        }
        ct_rescaler->SetInput(ct_filter->GetOutput());
        ct_rescaler->Update();

        auto ct_writer = WriterType::New();
        ct_writer->SetFileName(ct_output_name);
        ct_writer->SetInput(ct_rescaler->GetOutput());

        try
        {
            std::cout << "Writing CT image: " << ct_output_name << std::endl;
            ct_writer->Update();
        }
        catch (const itk::ExceptionObject& err)
        {
            std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
            std::cerr << err << std::endl;
        }
    }
    else
    {
        ct_filter->Update();
    }

    return EXIT_SUCCESS;
}
