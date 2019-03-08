
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
    target_include_directories(${PARSED_ARGS_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty SYSTEM)
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
    -Xcompiler=-Wshadow;
    -lineinfo>
    )
    target_link_libraries(${PARSED_ARGS_NAME} PUBLIC pangolin::pangolin)
endfunction()

add_exe(NAME gc SRCS main.cpp)
add_exe(NAME benchmark-linear SRCS benchmark_linear.cu)
add_exe(NAME benchmark-binary SRCS benchmark_binary.cu)