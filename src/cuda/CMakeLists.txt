find_package(CUDA REQUIRED)

add_library(cuda STATIC
        computeCUDA.cu
        cudaCSR.cu
        cudaHLL.cu
        cudaUtils.cu
)

target_link_libraries(cuda PUBLIC utils)

target_include_directories(cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Abilita la compilazione con CUDA
set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DSM_${CC};-arch=sm_${CC};-lineinfo;-Xcompiler=${CMAKE_CXX_FLAGS};-Xptxas=-v)