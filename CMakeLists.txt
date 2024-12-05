cmake_minimum_required(VERSION 3.10)
project(hybrid_mpc_mhe)

# Set build type to Release for optimization
set(CMAKE_BUILD_TYPE Release)

# Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

# Additional optimization flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O3)  # Enable high-level optimization
endif()

# Find catkin macros and libraries
set(${PROJECT_NAME}_CATKIN_DEPS
    roslib
    rospy
    std_msgs
    mav_msgs
    sensor_msgs
    geometry_msgs
    quadrotor_msgs
    nav_msgs
)
find_package(catkin REQUIRED COMPONENTS
    ${${PROJECT_NAME}_CATKIN_DEPS}
)

# Declare ROS messages, services, and actions
add_message_files(
  FILES
  ReferenceTrajectory.msg
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES 
  std_msgs
)

catkin_package(
  # INCLUDE_DIRS include
  CATKIN_DEPENDS ${${PROJECT_NAME}_CATKIN_DEPS}
)

# Retrieve the ACADOS source directory from the environment variable
if(DEFINED ENV{ACADOS_SOURCE_DIR})
  set(ACADOS_SOURCE_DIR $ENV{ACADOS_SOURCE_DIR})
else()
  message(FATAL_ERROR "Environment variable ACADOS_SOURCE_DIR is not set.")
endif()

# Build
include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
)

# Set C++ flags for LibTorch compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")