cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DeepStorm)

# set(CUDACXX "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Libraries/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(DeepStorm  example.cpp)

target_link_libraries(DeepStorm "${TORCH_LIBRARIES}")
target_link_libraries(DeepStorm "${OpenCV_LIBS}")

set_property(TARGET DeepStorm PROPERTY CXX_STANDARD 17)