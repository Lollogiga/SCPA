cmake_minimum_required(VERSION 3.16)
project(SCPA C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")

add_executable(
        SCPA src/main.c

        src/preprocessing/mmio.c
        src/include/mmio.h

        src/utils/matricDealloc.c
        src/include/matricDealloc.h

        src/utils/matrixPrint.c
        src/include/matrixPrint.h

        src/preprocessing/matrixPreProcessing.c
        src/include/matrixPreProcessing.h
)
