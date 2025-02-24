cmake_minimum_required(VERSION 3.31)
project(SCPA C)

set(CMAKE_C_STANDARD 11)

add_executable(SCPA main.c
        mmio.c
        header/mmio.h
        matrixPreProcessing.c
        header/matrixPreProcessing.h)
