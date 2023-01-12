cmake_minimum_required(VERSION 3.10)

project(opencv_demo)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_PREFIX_PATH /home/ligirk/workspace/lib/libtorch)

find_package(Torch REQUIRED)

# add_executable(train_cnn train_cnn.cpp)
# add_executable(train_fcn train_fcn.cpp)
# add_executable(test test.cpp)
# add_executable(inference inference.cpp)
add_executable(transform transform.cpp)

set(OPENCV_INCLUDE_DIR /usr/local/include/opencv4/)
set(OPENCV_LIB_DIR /usr/local/lib/)
include_directories(./)

# set library directory
# target_link_directories(train_cnn PUBLIC "${OPENCV_LIB_DIR}")
# target_link_directories(train_fcn PUBLIC "${OPENCV_LIB_DIR}")
# target_link_directories(test PUBLIC "${OPENCV_LIB_DIR}")
# target_link_directories(inference PUBLIC "${OPENCV_LIB_DIR}")
target_link_directories(transform PUBLIC "${OPENCV_LIB_DIR}")

# link specific object files we need
# target_link_libraries(train_cnn PUBLIC "${TORCH_LIBRARIES}" 
#                                     opencv_imgcodecs opencv_core opencv_imgproc opencv_highgui)
# target_include_directories(train_cnn PUBLIC "${TORCH_INCLUDE_DIRS}" "${OPENCV_INCLUDE_DIR}")
# target_link_libraries(train_fcn PUBLIC "${TORCH_LIBRARIES}" 
#                                     opencv_imgcodecs opencv_core opencv_imgproc opencv_highgui)
# target_include_directories(train_fcn PUBLIC "${TORCH_INCLUDE_DIRS}" "${OPENCV_INCLUDE_DIR}")

# target_link_libraries(test PUBLIC "${TORCH_LIBRARIES}" 
#                                     opencv_imgcodecs opencv_core opencv_imgproc opencv_highgui)
# target_include_directories(test PUBLIC "${TORCH_INCLUDE_DIRS}" "${OPENCV_INCLUDE_DIR}")
# target_link_libraries(inference PUBLIC "${TORCH_LIBRARIES}" 
#                                     opencv_imgcodecs opencv_core opencv_imgproc opencv_highgui)
# target_include_directories(inference PUBLIC "${TORCH_INCLUDE_DIRS}" "${OPENCV_INCLUDE_DIR}")

target_link_libraries(transform PUBLIC opencv_imgcodecs opencv_core opencv_imgproc opencv_highgui opencv_videoio)
target_include_directories(transform PUBLIC "${TORCH_INCLUDE_DIRS}" "${OPENCV_INCLUDE_DIR}")