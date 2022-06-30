cmake_minimum_required(VERSION 3.20)

project(image_stitching CUDA)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_C_STANDARD 11)

add_definitions(-D CMAKE_CXX_FLAGS="/O3")

find_package(OpenCV REQUIRED PATHS "C:/opencv")
find_package(OpenCV REQUIRED PATHS "C:/opencvBuild/build/install")

include_directories(${OpenCV_INCLUDE_DIR})

add_executable(image_stitching
        main.cpp
        MyBlend.cpp
        multiband_blend.cu
)

target_link_libraries(image_stitching ${OpenCV_LIBS})

target_compile_options(
        image_stitching PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info>)

set_target_properties(
        image_stitching
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)