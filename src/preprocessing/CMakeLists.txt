add_library(preprocessing STATIC
        matrixPreProcessing.c
        mmio.c
)

target_include_directories(preprocessing PRIVATE ${CMAKE_SOURCE_DIR}/include)
