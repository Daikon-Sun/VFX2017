cmake_minimum_required(VERSION 3.0)

project(HDR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

aux_source_directory(./src SRC)
add_executable(main ${SRC})
target_link_libraries(main ${OpenCV_LIBS} ${Boost_LIBRARIES})
