add_library(utils STATIC
        fileUtils.c
        flops.c
        matrixBalance.c
        matrixFree.c
)

target_link_libraries(utils PRIVATE m)

target_include_directories(utils PRIVATE ${CMAKE_SOURCE_DIR}/include)
