cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DeepStorm)

# cmake -DCMAKE_PREFIX_PATH="/mnt/Work/Big Projects/DeepStorm/Libraries/libtorch" ..
# include_directories("${CMAKE_PREFIX_PATH}/Libraries/libtorch")
# set(DCMAKE_PREFIX_PATH "/mnt/Work/Big Projects/DeepStorm/Libraries/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(DeepStorm  example2.cpp)

target_link_libraries(DeepStorm "${TORCH_LIBRARIES}")
target_link_libraries(DeepStorm "${OpenCV_LIBS}")

set_property(TARGET DeepStorm PROPERTY CXX_STANDARD 17)