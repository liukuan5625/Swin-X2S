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
#include "itkPermuteAxesImageFilter.h"


int main(int argc, char * argv[])
{
  char * input_name = nullptr;

  // Parse command line parameters
  while (argc > 1)
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
    }
  }

  constexpr unsigned int Dimension = 3;
  using InputPixelType = short;
  using InputImageType = itk::Image<InputPixelType, Dimension>;
  using ReaderType = itk::ImageFileReader<InputImageType>;
  auto                    reader = ReaderType::New();
  InputImageType::Pointer ct_image;
  if (input_name)
  {
    reader->SetFileName(input_name);

    try
    {
      reader->Update();
    }
    catch (const itk::ExceptionObject & err)
    {
      std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }

    ct_image = reader->GetOutput();
  }
  else
  {
    std::cout << "ERROR: No input given !" << std::endl;
    return EXIT_FAILURE;
  }


  InputImageType::DirectionType raw_direction;
  raw_direction = ct_image->GetDirection();
  using PermuteFilterType = itk::PermuteAxesImageFilter<InputImageType>;
  PermuteFilterType::Pointer permuteFilter = PermuteFilterType::New();
  permuteFilter->SetInput(ct_image);
  itk::FixedArray<unsigned int, Dimension> order;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      if (raw_direction[i][j] > 0.5 || raw_direction[i][j] < -0.5)
      {
        order[i] = j;
      }
    }
  }
  permuteFilter->SetOrder(order);
  ct_image = permuteFilter->GetOutput();
  ct_image->Update();


  InputImageType::DirectionType id_direction;
  id_direction.SetIdentity();
  ct_image->SetDirection(id_direction);
  ct_image->Update();


  using WriterType = itk::ImageFileWriter<InputImageType>;
  auto ct_writer = WriterType::New();
  ct_writer->SetFileName(input_name);
  ct_writer->SetInput(ct_image);
  try
  {
    std::cout << "Writing CT image: " << input_name << std::endl;
    ct_writer->Update();
  }
  catch (const itk::ExceptionObject & err)
  {
    std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
  }


  return EXIT_SUCCESS;
}
