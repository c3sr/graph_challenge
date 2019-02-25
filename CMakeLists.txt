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

# add our example as an executable
add_executable(gc src/main.cpp)

target_include_directories(gc PRIVATE thirdparty)

# add pangolin targets
if (EXTERNAL_PANGOLIN)
    find_package(pangolin CONFIG REQUIRED)
else()
    add_subdirectory("thirdparty/pangolin")
endif()

# link against the 32-bit pangolin library
target_link_libraries(gc pangolin::pangolin32)