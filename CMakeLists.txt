cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DeepStorm)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Libraries/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/Activations
    ${PROJECT_SOURCE_DIR}/include/BaseLayer
    ${PROJECT_SOURCE_DIR}/include/Datasets
    ${PROJECT_SOURCE_DIR}/include/Initializers
    ${PROJECT_SOURCE_DIR}/include/Layers
    ${PROJECT_SOURCE_DIR}/include/Losses
    ${PROJECT_SOURCE_DIR}/include/Model
    ${PROJECT_SOURCE_DIR}/include/Optimizers
    ${PROJECT_SOURCE_DIR}/include/Trainer
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SOURCES "src/*/*.cpp" example.cpp)

add_executable(DeepStorm ${SOURCES})

target_link_libraries(DeepStorm "${TORCH_LIBRARIES}")
target_link_libraries(DeepStorm "${OpenCV_LIBS}")

set_property(TARGET DeepStorm PROPERTY CXX_STANDARD 17)