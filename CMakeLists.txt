cmake_minimum_required(VERSION 3.16)
project(SCPA C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")

add_definitions(-DTEST)

# Opzione per abilitare/disabilitare CUDA
option(ENABLE_CUDA "Enable CUDA support" ON)

# Includi sottodirectory
add_subdirectory(src/utils)
add_subdirectory(src/preprocessing)
add_subdirectory(src/openmp)

if (ENABLE_CUDA)
    enable_language(CUDA)
    add_subdirectory(src/cuda)
endif()

# Crea eseguibile principale
add_executable(SCPA src/main.c
        src/utils/checkResultVector.c
        src/include/checkResultVector.h
)

# Linka le librerie
target_link_libraries(SCPA PRIVATE utils preprocessing openmp)

if (ENABLE_CUDA)
    target_link_libraries(SCPA PRIVATE cuda)
endif()

# Aggiungi include dir
target_include_directories(SCPA PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Stampa un messaggio di configurazione
if (ENABLE_CUDA)
    message(STATUS "CUDA support enabled")
else()
    message(STATUS "CUDA support disabled")
endif()
