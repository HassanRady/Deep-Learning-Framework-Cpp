cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(CNN-IN-C++)

# cmake -DCMAKE_PREFIX_PATH="/mnt/Work/Big Projects/CNN-IN-C++/Libraries/libtorch" ..
# include_directories("${CMAKE_PREFIX_PATH}/Libraries/libtorch")
# set(DCMAKE_PREFIX_PATH "/mnt/Work/Big Projects/CNN-IN-C++/Libraries/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(CNN-IN-C++  example2.cpp)

target_link_libraries(CNN-IN-C++ "${TORCH_LIBRARIES}")
target_link_libraries(CNN-IN-C++ "${OpenCV_LIBS}")

set_property(TARGET CNN-IN-C++ PROPERTY CXX_STANDARD 17)