# Set the minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Project name
project(IdentityDirection)

# Find ITK package
find_package(ITK REQUIRED)

INCLUDE(${ITK_USE_FILE})
# Include directories for ITK
include_directories(${ITK_INCLUDE_DIRS})

# Add the source file
add_executable(IdentityDirection IdentityDirection.cpp)

# Link ITK libraries
target_link_libraries(IdentityDirection ${ITK_LIBRARIES})

# Set the output directories for build
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

