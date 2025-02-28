cmake_minimum_required(VERSION 3.16)
project(SCPA C)

set(CMAKE_C_STANDARD 11)

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
