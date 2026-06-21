cmake_minimum_required(VERSION 3.14)
project(edge_fusion_centerpoint LANGUAGES CXX CUDA)

# ── C++ / CUDA standards ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 87)    # Ampere / Jetson Orin
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ── ament auto-find ────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(autoware_perception_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# ── CUDA ───────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── TensorRT ──────────────────────────────────────────────────────────────────
# JetPack / standard Linux install locations.
find_path(TRT_INCLUDE_DIR NvInfer.h
    PATHS
        /usr/include/aarch64-linux-gnu
        /usr/include
        /usr/local/include
    REQUIRED)

foreach(lib nvinfer nvinfer_plugin)
    find_library(TRT_LIB_${lib} ${lib}
        PATHS
            /usr/lib/aarch64-linux-gnu
            /usr/lib
            /usr/local/lib
        REQUIRED)
endforeach()

message(STATUS "TRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "nvinfer:     ${TRT_LIB_nvinfer}")

# ── CUDA postprocessor .so (built in deployment/plugins/) ─────────────────────
# The path is resolved at runtime via EF_POSTPROC_SO env var OR the default
# install path set here.  We only need the header at compile time.
set(POSTPROC_HEADER_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../deployment/plugins"
    CACHE PATH "Path to center_head_postprocess.h")

# ── Linker flags ─────────────────────────────────────────────────────────────
# TRT on Jetson links against optional DLA libraries (libnvdla_compiler.so,
# libnvcudla.so) that may not be present inside the build container.
# We never use DLA (GPU-only engines), so these symbols are safe to leave
# unresolved at build time — they will never be called at runtime.
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS}    -Wl,--allow-shlib-undefined")

# ── Core inference library ────────────────────────────────────────────────────
add_library(centerpoint_inference SHARED
    src/voxelizer.cpp
    src/centerpoint_trt.cpp)

target_include_directories(centerpoint_inference PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${POSTPROC_HEADER_DIR})

target_link_libraries(centerpoint_inference
    PUBLIC
        ${TRT_LIB_nvinfer}
        ${TRT_LIB_nvinfer_plugin}
        CUDA::cudart
        CUDA::cuda_driver
        rclcpp::rclcpp        # use imported target; avoids mixing plain/keyword sigs
    PRIVATE
        ${CMAKE_DL_LIBS})     # dlopen / dlsym for .so loading

# ── ROS 2 node ────────────────────────────────────────────────────────────────
add_executable(lidar_centerpoint_node
    src/lidar_centerpoint_node.cpp)

target_include_directories(lidar_centerpoint_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(lidar_centerpoint_node
    centerpoint_inference)

ament_target_dependencies(lidar_centerpoint_node
    rclcpp
    rclcpp_components
    sensor_msgs
    geometry_msgs
    autoware_perception_msgs
    tf2_ros
    tf2_geometry_msgs)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS centerpoint_inference lidar_centerpoint_node
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
    DESTINATION include)

install(DIRECTORY launch config
    DESTINATION share/${PROJECT_NAME})

ament_package()