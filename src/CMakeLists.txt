
function(add_exe)
    cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of boolean arguments
    "NAME" # list of names of mono-valued arguments
    "SRCS" # list of names of nuluti-valued arguments
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



add_exe(NAME benchmark-ktruss_linear SRCS benchmark_ktruss.cu)
add_exe(NAME benchmark-binary-ktruss SRCS benchmark_binary_ktruss.cu)
add_exe(NAME benchmark-multiGPU-incremental-ktruss SRCS benchmark_multiGPU_incremental_HSC_ktruss.cu)
add_exe(NAME benchmark-multiGPU-incremental-ktruss-Twitter SRCS benchmark_multiGPU_binary_HSC_ktruss_TW.cu)


add_exe(NAME mpi-test SRCS mpi_test.cu)