#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <stdio.h>
#include <string.h>

typedef struct {
    char implementation[7]; // "OpenMP", "CUDA" (type of implementation used)
    char format[9];         // "CSR", "HLL", "HLLAlign" (matrix storage format)
    char matrix_name[32];   // e.g., "adder_dcop_32.mtx" (name of the matrix being tested)
    char curr_func[32];     // e.g., "csr_openmpProduct_sol1" (name of the function being called)
    int NZ;                 // Number of non-zero in the matrix

    int repetitions;        // Number of repetitions for the test to average results
    double avg_time_ms;     // Average time in milliseconds to perform the computation
    double gflops;          // Calculated performance in GFLOPS (Giga Floating Point Operations per Second)

    // OpenMP
    int threads;            // 0 for CUDA, number of threads used in OpenMP (number of threads for parallel execution)

    // CUDA
    int block_size;         // Number of threads in a block in CUDA
    int warp_size;          // Warp size (typically 32 on most GPUs)
    int blocks_per_grid;    // Number of blocks per grid in CUDA (defines grid configuration)

    // Error statistics if the matrix calculation has an error
    int has_error;          // 1 if there was an error, 0 otherwise
    double max_abs_error;   // Maximum absolute error
    double max_rel_error;   // Maximum relative error
    double avg_abs_error;   // Average absolute error
    double avg_rel_error;   // Average relative error
    double error_L2;        // L2 error (Euclidean norm) indicating the overall error magnitude
} PerformanceResult;

#ifdef __cplusplus
extern "C" {
#endif

    FILE* csv_logger_init(const char* filename);
    void csv_logger_write(const PerformanceResult* result);
    void csv_logger_close();

#ifdef __cplusplus
}
#endif

// OpenMP defines
#define INIT_BENCHMARK_OPENMP(start, end, cumulative) \
    double start = 0, end = 0;                 \
    double cumulative = 0;                     \

#define BEGIN_BENCHMARK_OPENMP(perf_ptr, func_name_str)                 \
    strcpy((perf_ptr)->curr_func, func_name_str);                       \
    cumulative = 0;                                                     \
    for (int iBenchOMP = 0; iBenchOMP < MAX_REPETITIONS; iBenchOMP++) {

#define END_BENCHMARK_OPENMP(perf_ptr, start, end, cumulative)                \
    cumulative += end - start;                                                \
    (perf_ptr)->avg_time_ms = cumulative / (iBenchOMP + 1);                   \
    (perf_ptr)->repetitions = iBenchOMP + 1;                                  \
    (perf_ptr)->gflops = computeFlops(perf_ptr->NZ, (perf_ptr)->avg_time_ms); \
    csv_logger_write(perf_ptr);                                               \
}

// CUDA defines
#define INIT_BENCHMARK_CUDA(cumulative) \
    double cumulative = 0;

#define BEGIN_BENCHMARK_CUDA(perf_ptr, func_name_str)                      \
    strcpy((perf_ptr)->curr_func, func_name_str);                          \
    cumulative = 0;                                                        \
    for (int iBenchCUDA = 0; iBenchCUDA < MAX_REPETITIONS; iBenchCUDA++) {

#define END_BENCHMARK_CUDA(perf_ptr, elapsed)                                 \
    cumulative += elapsed;                                                    \
    (perf_ptr)->avg_time_ms = cumulative / (iBenchCUDA + 1);                   \
    (perf_ptr)->repetitions = iBenchCUDA + 1;                                  \
    (perf_ptr)->gflops = computeFlops(perf_ptr->NZ, (perf_ptr)->avg_time_ms); \
    csv_logger_write(perf_ptr);                                               \
}

#endif // PERFORMANCE_H
