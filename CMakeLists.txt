cmake_minimum_required(VERSION 3.15)
project(orbslam3)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../../lib")
list(APPEND CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")

#list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../Pangolin/build")
#list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/lib")
set(Pangolin_DIR "$ENV{CONDA_PREFIX}/lib/cmake/Pangolin" CACHE PATH "Path to PangolinConfig.cmake" FORCE)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(third_party/ORB_SLAM3)

pybind11_add_module(orbslam3 
    src/ORBSLAM3Wrapper.cpp
    src/NDArrayConverter.cpp
)

# Python module goes to extdir (for setuptools)
set_target_properties(orbslam3 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYTHON_MODULE_OUTPUT_DIR}"
)

target_link_libraries(orbslam3
PRIVATE
    ORB_SLAM3
)

target_include_directories(orbslam3
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)
