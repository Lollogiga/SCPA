add_library(utils STATIC
        checkResultVector.c
        createVector.c
        fileUtils.c
        flops.c
        mtxBalance.c
        mtxFree.c
        performance.c
)

target_link_libraries(utils PRIVATE m)

target_include_directories(utils PRIVATE ${CMAKE_SOURCE_DIR}/include)
