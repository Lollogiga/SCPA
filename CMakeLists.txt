cmake_minimum_required(VERSION 3.16)
project(SCPA C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")

add_definitions(-DTEST)
add_definitions(-DTEST_SINGLE_FILE)

add_executable(
        SCPA src/main.c

        src/preprocessing/mmio.c
        src/include/mmio.h

        src/utils/matricFree.c
        src/include/matricFree.h

        src/utils/matrixPrint.c
        src/include/matrixPrint.h

        src/preprocessing/matrixPreProcessing.c
        src/include/matrixPreProcessing.h
        src/product/serialProduct.c
        src/include/serialProduct.h
        src/product/utilsProduct.c
        src/include/utilsProduct.h
        src/utils/fileUtils.c
        src/include/fileUtils.h
)

# Abilita OpenMP
find_package(OpenMP REQUIRED)

if(OpenMP_C_FOUND)
    target_link_libraries(SCPA PUBLIC OpenMP::OpenMP_C)
    target_compile_options(SCPA PRIVATE ${OpenMP_C_FLAGS})
endif()