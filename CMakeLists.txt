cmake_minimum_required(VERSION 3.20)

project(image_stitching)

add_definitions(-D CMAKE_CXX_FLAGS="/O3")

find_package(OpenCV REQUIRED PATHS "C:/opencvBuild/build/install")

include_directories(${OpenCV_INCLUDE_DIRF})

add_executable(image_stitching
main.cpp MyBlend.cpp
)

target_link_libraries(image_stitching ${OpenCV_LIBS})

