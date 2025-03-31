add_library(preprocessing STATIC
        preprocessing.c
        mmio.c
)

target_include_directories(preprocessing PRIVATE ${CMAKE_SOURCE_DIR}/include)
