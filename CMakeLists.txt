cmake_minimum_required(VERSION 3.12)

option(EXTERNAL_PANGOLIN ON "Use external Pangolin (with -DCMAKE_MODULE_PATH)")

# Pangolin uses Hunter, so we'll require it here
include("cmake/HunterGate.cmake")

# LOCAL specifies that cmake/Hunter/config.cmake controls our pagacke versions and sources
# That file specifies that pangolin should be loaded from the git submodule in this repository
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.115.tar.gz"
    SHA1 "1b3f3addc801405769edbb7ebbe701223db3efa6"
    LOCAL
)

project(PangolinExample LANGUAGES CXX CUDA VERSION 0.1.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set a default build type if none was specified
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type "Debug")
endif()

 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    message(STATUS "${CMAKE_CUDA_FLAGS}")
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()

# add pangolin targets
if (EXTERNAL_PANGOLIN)
    find_package(pangolin CONFIG REQUIRED)
else()
    add_subdirectory("thirdparty/pangolin")
endif()

add_subdirectory(src)


