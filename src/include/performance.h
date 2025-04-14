#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <stdio.h>

typedef struct {
    char implementation[7]; // "OpenMP", "CUDA" (type of implementation used)
    char format[9];         // "CSR", "HLL", "HLLAlign" (matrix storage format)
    char matrix_name[32];   // e.g., "adder_dcop_32.mtx" (name of the matrix being tested)
    int NZ;                 // Number of non-zero in the matrix

    // Common fields
    int repetitions;        // Number of repetitions for the test to average results
    double avg_time_ms;     // Average time in milliseconds to perform the computation
    double gflops;          // Calculated performance in GFLOPS (Giga Floating Point Operations per Second)

    // OpenMP
    int threads;            // 0 for CUDA, number of threads used in OpenMP (number of threads for parallel execution)

    // CUDA
    int blockSize;          // Number of threads in a block in CUDA
    int blocksPerGrid;      // Number of blocks per grid in CUDA (defines grid configuration)

    // Error statistics if the matrix calculation has an error
    int has_error;          // 1 if there was an error, 0 otherwise
    double max_abs_error;   // Maximum absolute error
    double max_rel_error;   // Maximum relative error
    double avg_abs_error;   // Average absolute error
    double avg_rel_error;   // Average relative error
    double error_L2;        // L2 error (Euclidean norm) indicating the overall error magnitude
} PerformanceResult;

FILE* csv_logger_init(const char* filename);
void csv_logger_write(FILE* fp, const PerformanceResult* result);
void csv_logger_close(FILE* fp);

#endif // PERFORMANCE_H
