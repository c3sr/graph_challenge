


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
    -Xcompiler=-Wfloat-equal;
    -Xcompiler=-Wvla;
    -Xcompiler=-Wshadow;>
    )
    target_link_libraries(${PARSED_ARGS_NAME} PUBLIC pangolin::pangolin)
    target_link_libraries(${PARSED_ARGS_NAME} PUBLIC nvToolsExt)

    if(OpenMP_CXX_FOUND)
      target_link_libraries(${PARSED_ARGS_NAME} PUBLIC ${OpenMP_CXX_LIBRARIES})
    endif()
endfunction()


function(add_mpi_exe)
    cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of boolean arguments
    "NAME" # list of names of mono-valued arguments
    "SRCS" # list of names of multi-valued arguments
    ${ARGN} # arguments of the function to parse
    )

    add_exe(NAME ${PARSED_ARGS_NAME} SRCS ${PARSED_ARGS_SRCS})

  ## TODO
  ## just using target_link_libraries(pangolin INTERFACE MPI::MPI_CXX)
  ## causes device linking with -pthread, even as of 3.15.0-rc1
  ## https://gitlab.kitware.com/cmake/cmake/issues/18897
  ## Add MPI flags
  if(MPI_FOUND)
    target_include_directories(${PARSED_ARGS_NAME} SYSTEM PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${PARSED_ARGS_NAME} PUBLIC ${MPI_CXX_LIBRARIES})
  endif()
endfunction()

find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI)
if(MPI_FOUND)
  message(STATUS "MPI found")
  message(STATUS "MPI_CXX_INCLUDE_DIRS=" ${MPI_CXX_INCLUDE_DIRS})
  message(STATUS "MPI_CXX_LIBRARIES=" ${MPI_CXX_LIBRARIES})
  message(STATUS "MPIEXEC_EXECUTABLE=" ${MPIEXEC_EXECUTABLE})
endif()


add_mpi_exe(NAME mpi-test SRCS mpi_test.cu)

add_exe(NAME benchmark-ktruss_linear SRCS benchmark_ktruss.cu)
add_exe(NAME benchmark-binary-ktruss SRCS benchmark_binary_ktruss.cu)
add_exe(NAME benchmark-multiGPU-incremental-ktruss SRCS benchmark_multiGPU_incremental_HSC_ktruss.cu)
add_exe(NAME benchmark-multiGPU-incremental-ktruss-Twitter SRCS benchmark_multiGPU_binary_HSC_ktruss_TW.cu)

add_exe(NAME 2dhist SRCS 2dhist.cu)
add_exe(NAME benchmark-b-vertex-t-edge-binary SRCS benchmark_b_vertex_t_edge_binary.cu)
add_exe(NAME benchmark-bs-vertex-t-edge-binary SRCS benchmark_bs_vertex_t_edge_binary.cu)
add_exe(NAME benchmark-bitvector SRCS benchmark_bitvector.cu)
add_exe(NAME benchmark-binary SRCS benchmark_binary.cu)
add_exe(NAME benchmark-bissonfatica18 SRCS benchmark_bissonfatica18.cu)
add_exe(NAME benchmark-bmtx-cpu SRCS benchmark_bmtx_cpu.cu)
add_exe(NAME benchmark-bmtx-gpu SRCS benchmark_bmtx_gpu.cu)
add_exe(NAME benchmark-cusparse SRCS benchmark_cusparse.cu)
add_exe(NAME benchmark-dyn SRCS benchmark_dyn.cu)
add_exe(NAME benchmark-dysel SRCS benchmark_dysel.cu)
add_exe(NAME benchmark-io-overlap SRCS benchmark_io_overlap.cu)
add_exe(NAME benchmark-linear-io-queue SRCS benchmark_linear_io_queue.cu)
add_exe(NAME benchmark-linear-mp SRCS benchmark_linear_mp.cu)
add_exe(NAME benchmark-nvgraph SRCS benchmark_nvgraph.cu)
add_exe(NAME benchmark-vertex-binary-cpu SRCS benchmark_vertex_binary_cpu.cu)
add_exe(NAME benchmark-vertex-blocks-cache-block-binary SRCS benchmark_vertex_blocks_cache_block_binary.cu)
add_exe(NAME loads SRCS loads.cu)
add_exe(NAME pangolin-topo SRCS pangolin_topo.cu)
add_exe(NAME pangolin-version SRCS pangolin_version.cu)
