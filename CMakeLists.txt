cmake_minimum_required(VERSION 3.16)
project(SCPA C)

set(CMAKE_C_STANDARD 11)

add_executable(
        SCPA src/main.c

            src/preprocessing/matrixPreProcessing.c
            src/include/matrixPreProcessing.h

            src/preprocessing/mmio.c
            src/include/mmio.h
)
