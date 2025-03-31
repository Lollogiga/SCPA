find_package(OpenMP REQUIRED)

add_library(openmp STATIC
        computeOpenMP.c
        product/CSR.c
        product/HLL.c
        product/Serial.c
)

target_link_libraries(openmp PUBLIC utils)

target_include_directories(openmp PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(OpenMP_C_FOUND)
    target_link_libraries(openmp PUBLIC OpenMP::OpenMP_C)
    target_compile_options(openmp PRIVATE ${OpenMP_C_FLAGS})
endif()
