
function(add_exe)
    cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of boolean arguments
    "NAME" # list of names of mono-valued arguments
    "SRCS" # list of names of multi-valued arguments
    ${ARGN} # arguments of the function to parse
    )
    add_executable(${PARSED_ARGS_NAME} ${PARSED_ARGS_SRCS})
    # set_target_properties(${PARSED_ARGS_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${PARSED_ARGS_NAME} SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty)
    target_compile_options(${PARSED_ARGS_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall;
    -Xcompiler=-Wextra;
    -Xcompiler=-Wcast-align;
    -Xcompiler=-Wstrict-aliasing;
    -Xcompiler=-Wpointer-arith;
    -Xcompiler=-Winit-self;
    -Xcompiler=-Wswitch-enum;
    -Xcompiler=-Wredundant-decls;
    -Xcompiler=-Wfloat-equal;
    -Xcompiler=-Wundef;
    -Xcompiler=-Wvla;
    -Xcompiler=-Wshadow;>
    )
    target_link_libraries(${PARSED_ARGS_NAME} PUBLIC pangolin::pangolin)
    target_link_libraries(${PARSED_ARGS_NAME} PUBLIC nvToolsExt)

    if(OpenMP_CXX_FOUND)
      target_link_libraries(${PARSED_ARGS_NAME} PUBLIC ${OpenMP_CXX_LIBRARIES})
    endif()
endfunction()


find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
endif()

add_exe(NAME 2dhist SRCS 2dhist.cu)
add_exe(NAME benchmark-b-vertex-t-edge-binary SRCS benchmark_b_vertex_t_edge_binary.cu)
add_exe(NAME benchmark-bs-vertex-t-edge-binary SRCS benchmark_bs_vertex_t_edge_binary.cu)
add_exe(NAME benchmark-vertex-blocks-cache-block-binary SRCS benchmark_vertex_blocks_cache_block_binary.cu)
add_exe(NAME benchmark-binary SRCS benchmark_binary.cu)
add_exe(NAME benchmark-bitvector SRCS benchmark_bitvector.cu)
add_exe(NAME benchmark-dyn SRCS benchmark_dyn.cu)
add_exe(NAME benchmark-linear-io-queue SRCS benchmark_linear_io_queue.cu)
# add_exe(NAME benchmark-linear-io-vec-queue SRCS benchmark_linear_io_vec_queue.cu)
add_exe(NAME benchmark-linear-mp SRCS benchmark_linear_mp.cu)
# add_exe(NAME benchmark-linear3 SRCS benchmark_linear3.cu)
add_exe(NAME loads SRCS loads.cu)
add_exe(NAME pangolin-topo SRCS pangolin_topo.cu)
add_exe(NAME pangolin-version SRCS pangolin_version.cu)
add_exe(NAME mpi-test SRCS mpi_test.cu)