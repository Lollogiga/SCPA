# Trova CUDA e fornisci la versione minima
find_package(CUDA REQUIRED)

# Imposta l'architettura target per la tua GPU (sm_75 per la tua Quadro RTX 5000)
set(CC 75)

add_library(cuda STATIC
        computeCUDA.cu
        Utils.cu
        product/CSR.cu
        product/HLL.cu
)

# Linka la libreria 'utils'
target_link_libraries(cuda PUBLIC utils)

# Aggiungi le directory di include
target_include_directories(cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Abilita la compilazione con CUDA
set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DSM_${CC};-arch=sm_${CC};-lineinfo;-Xcompiler=${CMAKE_CXX_FLAGS};-Xptxas=-v)

# Se vuoi abilitare il debug per CUDA
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo")