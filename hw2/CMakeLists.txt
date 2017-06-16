cmake_minimum_required(VERSION 3.0)


project(PANORAMA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14")
#opencv
find_package(OpenCV REQUIRED)
#c++ boost
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
#openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
#ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
#eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
#message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
#linking
aux_source_directory(./src SRC)
add_executable(main ${SRC})
target_link_libraries(main ${OpenCV_LIBS} ${Boost_LIBRARIES} ${CERES_LIBRARIES})
