cmake_minimum_required(VERSION 3.0.2)
project(dmsa)

set(CMAKE_BUILD_TYPE release)

add_compile_options(-std=c++17)



find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  roscpp
  rospy
  std_msgs
  tf
  pcl_conversions)



find_package(Boost 1.40.0 COMPONENTS filesystem system iostreams REQUIRED)
#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/gnuplot-iostream")

#find_package(Gnuplot REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(PCL 1.3 REQUIRED COMPONENTS)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  message("OPENMP Found")
else()
  message("OPENMP NOT Found")
endif()

catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msgs tf pcl_conversions sensor_msgs
)

#add_compile_options(-pg)
#add_compile_options(-O3)
#set(catkin_LIBRARIES ${catkin_LIBRARIES} -pg)



include_directories(
        include
        Thirdparty
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

add_executable(dmsa
               src/main.cpp
               src/dmsa.cpp
              )


target_link_libraries(dmsa ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Boost_LIBRARIES})
