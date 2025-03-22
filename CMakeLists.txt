cmake_minimum_required(VERSION 3.16)
project(SCPA C CUDA)

set(CMAKE_C_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")

add_definitions(-DTEST)

# Includi sottodirectory
add_subdirectory(src/utils)
add_subdirectory(src/preprocessing)
add_subdirectory(src/openmp)
add_subdirectory(src/cuda)

# Crea eseguibile principale
add_executable(SCPA src/main.c)

# Linka le librerie
target_link_libraries(SCPA PRIVATE utils preprocessing openmp cuda)

# Aggiungi include dir
target_include_directories(SCPA PRIVATE ${CMAKE_SOURCE_DIR}/include)
