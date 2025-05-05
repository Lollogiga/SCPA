#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "../include/openmp/Serial.h"
#include "../include/cuda/Serial.cuh"
#include "../include/cuda/CSR.cuh"
#include "../include/cuda/HLL.cuh"
#include "../include/cuda/Utils.cuh"
#include "../include/checkResultVector.h"
#include "../include/mtxStructs.h"
#include "../include/flops.h"
#include "../include/performance.h"

/*
 * --- NOTE SULLA SCHEDA VIDEO SUL SERVER DI DIPARTIMENTO ---
 * Nome GPU: Quadro RTX 5000
 * Max threads per block: 1024
 * Warp size: 32
 * Max threads per multiprocessore: 1024
 * Numero di multiprocessori: 48
 * Max blocchi per griglia: 2147483647
 * Memoria condivisa per blocco: 49152 bytes
 */

int csr_product(CSRMatrix *h_csr, ResultVector *serial, int blockSize, int warpSize, PerformanceResult *performance) {
    cudaError_t cuda_error;
    double int_err;
    
    float elapsedTime;

    if (!h_csr || !serial) {
        perror("CSR_cuda_product failed: Input not initialized");
        return -1;
    }
    
    CSRMatrix *d_csr = uploadCSRToDevice(h_csr);
    if (d_csr == nullptr) {
        freeCSRDevice(d_csr);
        return -1;
    }

    int threadsPerBlock = blockSize;
    int warpsPerBlock = threadsPerBlock / warpSize;
    int blocksPerGrid = (h_csr->M + warpsPerBlock - 1) / warpsPerBlock;

    performance->blocks_per_grid = blocksPerGrid;

    MatVal *h_x = create_vector(h_csr->N);
    if (h_x == nullptr) {
        printf("\033[31mcsr_product - create_vector h_x->N failed\033[0m\n");

        freeCSRDevice(d_csr);
        return -1;
    }

    MatVal* d_x;
    cuda_error = cudaMallocManaged(&d_x, h_csr->N * sizeof(MatVal));
    if (cuda_error != cudaSuccess) {
        printf("\033[31mcsr_product - cudaMallocManaged h_csr->N failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeCSRDevice(d_csr);
        free_vector(h_x);
        return -1;
    }
    cuda_error = cudaMemcpy(d_x, h_x, h_csr->N * sizeof(MatVal), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        printf("\033[31mcsr_product - cudaMemcpy h_x failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        return -1;
    }
    cuda_error = cudaMemAdvise(d_x, h_csr->N * sizeof(MatVal), cudaMemAdviseSetReadMostly, 0);
    if (cuda_error != cudaSuccess) {
        printf("\033[31mcsr_product - cudaMemAdvise SetReadMostly failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        return -1;
    }
    cuda_error = cudaMemAdvise(d_x, h_csr->N * sizeof(MatVal), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    if (cuda_error != cudaSuccess) {
        printf("\033[31mcsr_product - cudaMemAdvise SetPreferredLocation failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        return -1;
    }

    ResultVector *h_result_vector = create_result_vector(h_csr->M);
    if (h_result_vector == nullptr) {
        printf("\033[31mcsr_product - create_result_vector h_csr->M failed\033[0m\n");

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        return -1;
    }
    ResultVector *d_result_vector = uploadResultVectorToDevice(h_result_vector);
    if (d_result_vector == nullptr) {
        printf("\033[31mcsr_product - uploadResultVectorToDevice h_result_vector failed\033[0m\n");

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        return -1;
    }

    CUDA_EVENT_CREATE(start, stop)
    INIT_BENCHMARK_CUDA(cumulative)

    // SOL SERIAL
    CUDA_EVENT_START(start)
    spmv_csr_serial<<<1, 1>>>(d_csr, d_x, d_result_vector);
    cuda_error = cudaGetLastError();
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    if (cuda_error) {
        printf("\033[31mcsr_product - spmv_csr_serial kernel failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    printf("csr_cuda_Serial: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    int_err = downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (int_err != 0) {
        printf("\033[31mcsr_product - downloadResultVectorToHost spmv_csr_serial failed\033[0m\n");

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = checkResultVector(serial, h_result_vector);
    if (int_err) {
        printf("\033[31mcsr_product - checkResultVector spmv_csr_serial failed\033[0m\n");

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    freeResultVectorFromDevice(d_result_vector);

    // SOL 1
    BEGIN_BENCHMARK_CUDA(performance, "csr_cudaProduct_sol1")
    for (int i = 0; i < h_csr->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);
    if (d_result_vector == nullptr) {
        printf("\033[31mcsr_product - uploadResultVectorToDevice h_result_vector failed\033[0m\n");

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    CUDA_EVENT_START(start)
    csr_cudaProduct_sol1<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    cuda_error = cudaGetLastError();
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    if (cuda_error) {
        printf("\033[31mcsr_product - csr_cudaProduct_sol1 kernel failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (int_err != 0) {
        printf("\033[31mcsr_product - downloadResultVectorToHost csr_cudaProduct_sol1 failed\033[0m\n");

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = checkResultVector(serial, h_result_vector);
    if (int_err) {
        printf("\033[31mcsr_product - checkResultVector csr_cudaProduct_sol1 failed\033[0m\n");

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    freeResultVectorFromDevice(d_result_vector);
    END_BENCHMARK_CUDA(performance, elapsedTime)
    printf("csr_cuda1: Flops: %f\n", computeFlops(h_csr->NZ, performance->avg_time_ms));

    // SOL 2
    BEGIN_BENCHMARK_CUDA(performance, "csr_cudaProduct_sol2")
    for (int i = 0; i < h_csr->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);
    if (d_result_vector == nullptr) {
        printf("\033[31mcsr_product - uploadResultVectorToDevice h_result_vector failed\033[0m\n");

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }

    blocksPerGrid = (h_csr->M + warpsPerBlock - 1) / warpsPerBlock;

    CUDA_EVENT_START(start)
    csr_cudaProduct_sol2<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    cuda_error = cudaGetLastError();
    if (cuda_error) {
        printf("\033[31mcsr_product - csr_cudaProduct_sol2 kernel failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    int_err = downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (int_err != 0) {
        printf("\033[31mcsr_product - downloadResultVectorToHost csr_cudaProduct_sol2 failed\033[0m\n");

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = checkResultVector(serial, h_result_vector);
    if (int_err) {
        printf("\033[31mcsr_product - checkResultVector csr_cudaProduct_sol2 failed\033[0m\n");

        analyzeErrorVector(serial, h_result_vector, performance);
    }
    freeResultVectorFromDevice(d_result_vector);
    END_BENCHMARK_CUDA(performance, elapsedTime)
    printf("csr_cuda2: Flops: %f\n", computeFlops(h_csr->NZ, performance->avg_time_ms));

    freeCSRDevice(d_csr);
    cudaFree(d_x);

    CUDA_EVENT_DESTROY(start, stop)

    free(h_x);

    return 0;
}

int hll_CUDA_product(HLLMatrix *h_hll, ResultVector *serial, int blockSize,  PerformanceResult *performance) {
    cudaError_t cuda_error;
    double int_err;

    float elapsedTime;

    if (!h_hll || !serial) {
        perror("HLL_cuda_product failed: Input not initialized");
        return -1;
    }

    // Upload hll to device:
    HLLMatrix *d_hll = uploadHLLToDevice(h_hll);
    if (d_hll == nullptr) {
        perror("Hll_cuda_product failed: uploadHLLToDevice failed");

        return -1;
    }

    int threadsPerBlock = blockSize;
    int blocksPerGrid = h_hll->numBlocks;

    dim3 grid(blocksPerGrid);
    dim3 block(threadsPerBlock);

    performance->blocks_per_grid = blocksPerGrid;

    //Create vector:
    MatVal *h_x = create_vector(h_hll->N);
    if (h_x == nullptr) {
        perror("Hll_cuda_product failed: create_vector failed");
        return -1;
    }

    //Upload vector on device:
    MatVal* d_x;
    cuda_error = cudaMallocManaged(&d_x, h_hll->N * sizeof(MatVal));
    if (cuda_error != cudaSuccess) {
        printf("\033[31mhll_product - cudaMallocManaged h_hll->N failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        free_vector(h_x);
        return -1;
    }
    cuda_error = cudaMemcpy(d_x, h_x, h_hll->N * sizeof(MatVal), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        printf("\033[31mhll_product - cudaMemcpy h_x failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        free_vector(h_x);
        cudaFree(d_x);
        return -1;
    }
    cuda_error = cudaMemAdvise(d_x, h_hll->N * sizeof(MatVal), cudaMemAdviseSetReadMostly, 0);
    if (cuda_error != cudaSuccess) {
        printf("\033[31mhll_product - cudaMemAdvise SetReadMostly failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        free_vector(h_x);
        cudaFree(d_x);
        return -1;
    }
    cuda_error = cudaMemAdvise(d_x, h_hll->N * sizeof(MatVal), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    if (cuda_error != cudaSuccess) {
        printf("\033[31mhll_product - cudaMemAdvise SetPreferredLocation failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        free_vector(h_x);
        cudaFree(d_x);
        return -1;
    }

    ResultVector *h_result_vector = create_result_vector(h_hll->M);
    if (h_result_vector == nullptr) {
        perror("Hll_cuda_product failed: create_result_vector failed");
        return -1;
    }
    ResultVector *d_result_vector = uploadResultVectorToDevice(h_result_vector);
    if (d_result_vector == nullptr) {
        printf("\033[31mhll_product - uploadResultVectorToDevice h_result_vector failed\033[0m\n");

        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        return -1;
    }

    CUDA_EVENT_CREATE(start, stop)
    INIT_BENCHMARK_CUDA(cumulative)

    // SOL SERIAL
    CUDA_EVENT_START(start)
    spmv_hll_serial<<<1, 1>>>(d_hll, d_x, d_result_vector);
    cuda_error = cudaGetLastError();
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    if (cuda_error) {
        printf("\033[31mhll_product - spmv_hll_serial kernel failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeHLLDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    printf("hll_cuda_Serial: Flops: %f\n", computeFlops(h_hll->NZ, elapsedTime));
    int_err = downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (int_err != 0) {
        printf("\033[31mhll_product - downloadResultVectorToHost spmv_hll_serial failed\033[0m\n");

        freeHLLDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = checkResultVector(serial, h_result_vector);
    if (int_err) {
        printf("\033[31mhll_product - checkResultVector spmv_hll_serial failed\033[0m\n");

        freeHLLDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    freeResultVectorFromDevice(d_result_vector);

    // SOL 1
    BEGIN_BENCHMARK_CUDA(performance, "hll_cudaProduct_sol1")
    for (int i = 0; i < h_hll->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);
    if (d_result_vector == nullptr) {
        printf("\033[31mcsr_product - uploadResultVectorToDevice h_result_vector failed\033[0m\n");

        freeHLLDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }

    CUDA_EVENT_START(start)
    spmv_hll_parallel<<<grid, block>>>(d_hll, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    cuda_error = cudaGetLastError();
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    if (cuda_error) {
        printf("\033[31mhll_product - hll_cudaProduct_sol1 kernel failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeHLLDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (int_err != 0) {
        printf("\033[31mcsr_product - downloadResultVectorToHost csr_cudaProduct_sol1 failed\033[0m\n");

        freeHLLDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = checkResultVector(serial, h_result_vector);
    if (int_err) {
        printf("\033[31hll_product - checkResultVector hll_cudaProduct_sol1 failed\033[0m\n");

        freeHLLDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    freeResultVectorFromDevice(d_result_vector);
    END_BENCHMARK_CUDA(performance, elapsedTime)
    printf("hll_cuda1: Flops: %f\n", computeFlops(h_hll->NZ, elapsedTime));


    // Solution 2:
    BEGIN_BENCHMARK_CUDA(performance, "hll_cudaProduct_sol2")
    for (int i = 0; i < h_hll->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);
    if (d_result_vector == nullptr) {
        printf("\033[31mhll_product - uploadResultVectorToDevice h_result_vector failed\033[0m\n");

        freeHLLDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }

    CUDA_EVENT_START(start)
    spmv_hll_coalesced<<<grid, block>>>(d_hll, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    cuda_error = cudaGetLastError();
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    if (cuda_error) {
        printf("\033[31mhll_product - hll_cudaProduct_sol2 kernel failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeHLLDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (int_err != 0) {
        printf("\033[31mhll_product - downloadResultVectorToHost hll_cudaProduct_sol2 failed\033[0m\n");

        freeHLLDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = checkResultVector(serial, h_result_vector);
    if (int_err) {
        printf("\033[31hll_product - checkResultVector hll_cudaProduct_sol2 failed\033[0m\n");

        freeHLLDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    END_BENCHMARK_CUDA(performance, elapsedTime)
    printf("hll_cuda2: Flops: %f\n", computeFlops(h_hll->NZ, elapsedTime));

    freeHLLDevice(d_hll);

    free_vector(h_x);
    cudaFree(d_x);

    free_result_vector(h_result_vector);
    freeResultVectorFromDevice(d_result_vector);
    return 0;

}

int hllAligned_CUDA_product(HLLMatrixAligned *h_hll, ResultVector *serial_result, int blockSize, PerformanceResult *performance){
    cudaError_t cuda_error;
    double int_err;

    float elapsedTime;

    if (!h_hll || !serial_result) {
       perror("HllAligned_cuda_product failed: Input not initialized");
        return -1;
    }

    // Upload hll to device:
    HLLMatrixAligned *d_hll = uploadHLLAlignedToDevice(h_hll);
    if (d_hll == nullptr) {
        perror("HllAligned_cuda_product failed: uploadHLLToDevice failed");

        return -1;
    }

    int threadsPerBlock = blockSize;
    int blocksPerGrid = h_hll->numBlocks;

    dim3 grid(blocksPerGrid);
    dim3 block(threadsPerBlock);

    //Create vector:
    MatVal *h_x = create_vector(h_hll->N);
    if (h_x == nullptr) {
        perror("HllAligned_cuda_product failed: create_vector failed");
        return -1;
    }

    //Upload vector on device:
    MatVal* d_x;
    cuda_error = cudaMallocManaged(&d_x, h_hll->N * sizeof(MatVal));
    if (cuda_error != cudaSuccess) {
        printf("\033[31mhllAligned_product - cudaMallocManaged h_hll->N failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        free_vector(h_x);
        return -1;
    }
    cuda_error = cudaMemcpy(d_x, h_x, h_hll->N * sizeof(MatVal), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        printf("\033[31mhllAligned_product - cudaMemcpy h_x failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        free_vector(h_x);
        cudaFree(d_x);
        return -1;
    }
    cuda_error = cudaMemAdvise(d_x, h_hll->N * sizeof(MatVal), cudaMemAdviseSetReadMostly, 0);
    if (cuda_error != cudaSuccess) {
        printf("\033[31mhllAligned_product - cudaMemAdvise SetReadMostly failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        free_vector(h_x);
        cudaFree(d_x);
        return -1;
    }
    cuda_error = cudaMemAdvise(d_x, h_hll->N * sizeof(MatVal), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    if (cuda_error != cudaSuccess) {
        printf("\033[31mhllAligned_product - cudaMemAdvise SetPreferredLocation failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        free_vector(h_x);
        cudaFree(d_x);
        return -1;
    }

    ResultVector *h_result_vector = create_result_vector(h_hll->M);
    if (h_result_vector == nullptr) {
        perror("HllAligned_cuda_product failed: create_result_vector failed");
        return -1;
    }
    ResultVector *d_result_vector = uploadResultVectorToDevice(h_result_vector);
    if (d_result_vector == nullptr) {
        printf("\033[31mhllAligned_product - uploadResultVectorToDevice h_result_vector failed\033[0m\n");

        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        return -1;
    }

    CUDA_EVENT_CREATE(start, stop)
    INIT_BENCHMARK_CUDA(cumulative)

    // Serial solution:
    CUDA_EVENT_START(start)
    spmv_hllAligned_serial<<<1, 1>>>(d_hll, d_x, d_result_vector);
    cuda_error = cudaGetLastError();
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    if (cuda_error) {
        printf("\033[31mhllAligned_product - spmv_hllAligned_serial kernel failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeHLLAlignedToDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    printf("hllAligned_cuda_Serial: Flops: %f\n", computeFlops(h_hll->NZ, elapsedTime));
    int_err = downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (int_err != 0) {
        printf("\033[31mhllAligned_product - downloadResultVectorToHost spmv_hll_serial failed\033[0m\n");

        freeHLLAlignedToDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = checkResultVector(serial_result, h_result_vector);
    if (int_err) {
        printf("\033[31mhllAligned_product - checkResultVector spmv_hll_serial failed\033[0m\n");

        freeHLLAlignedToDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }

    freeResultVectorFromDevice(d_result_vector);

    // SOL 1
    BEGIN_BENCHMARK_CUDA(performance, "hllAlign_cudaProduct_sol1")
    for (int i = 0; i < h_hll->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);
    if (d_result_vector == nullptr) {
        printf("\033[31mcsr_product - uploadResultVectorToDevice h_result_vector failed\033[0m\n");

        freeHLLAlignedToDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }

    CUDA_EVENT_START(start)
    spmv_hllAligned_parallel<<<grid, block>>>(d_hll, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    cuda_error = cudaGetLastError();
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    if (cuda_error) {
        printf("\033[31mhll_product - hll_cudaProduct_sol1 kernel failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeHLLAlignedToDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (int_err != 0) {
        printf("\033[31mcsr_product - downloadResultVectorToHost csr_cudaProduct_sol1 failed\033[0m\n");

        freeHLLAlignedToDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = checkResultVector(serial_result, h_result_vector);
    if (int_err) {
        printf("\033[31hll_product - checkResultVector hll_cudaProduct_sol1 failed\033[0m\n");

        freeHLLAlignedToDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    freeResultVectorFromDevice(d_result_vector);
    END_BENCHMARK_CUDA(performance, elapsedTime)
    printf("hllAligned_cuda1: Flops: %f\n", computeFlops(h_hll->NZ, elapsedTime));

    // SOL 2
    BEGIN_BENCHMARK_CUDA(performance, "hllAlign_cudaProduct_sol2")
    for (int i = 0; i < h_hll->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);
    if (d_result_vector == nullptr) {
        printf("\033[31mhll_product - uploadResultVectorToDevice h_result_vector failed\033[0m\n");

        freeHLLAlignedToDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }

    CUDA_EVENT_START(start)
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    spmv_hllAligned_coalesced<<<grid, block>>>(d_hll, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    cuda_error = cudaGetLastError();
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    if (cuda_error) {
        printf("\033[31mhll_product - hll_cudaProduct_sol2 kernel failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeHLLAlignedToDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (int_err != 0) {
        printf("\033[31mhll_product - downloadResultVectorToHost hll_cudaProduct_sol2 failed\033[0m\n");

        freeHLLAlignedToDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    int_err = checkResultVector(serial_result, h_result_vector);
    if (int_err) {
        printf("\033[31hll_product - checkResultVector hll_cudaProduct_sol2 failed\033[0m\n");

        freeHLLAlignedToDevice(d_hll);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    END_BENCHMARK_CUDA(performance, elapsedTime)
    printf("hllAligned_cuda2: Flops: %f\n", computeFlops(h_hll->NZ, elapsedTime));

    freeHLLAlignedToDevice(d_hll);

    free_vector(h_x);
    cudaFree(d_x);

    free_result_vector(h_result_vector);
    freeResultVectorFromDevice(d_result_vector);
    return 0;

}

extern "C" int computeCUDA(CSRMatrix *csr, HLLMatrix *hll, HLLMatrixAligned *hllAligned, int blockSize, int warpSize, PerformanceResult *performance) {
    MatVal *vector = create_vector(csr->N);
    if (vector == nullptr) return -1;

    ResultVector *serial = csr_serialProduct(csr, vector);

    // CSR format
    strcpy(performance->format, "CSR");

    printf("\033[1;31m---- csr ----:\033[0m\n");
    csr_product(csr, serial, blockSize, warpSize, performance);

    performance->warp_size = 0;

    // HLL format
    strcpy(performance->format, "HLL");

    printf("\033[1;32m---- hll ----:\033[0m\n");
    hll_CUDA_product(hll, serial, blockSize, performance);

    // HLLAlign format
    strcpy(performance->format, "HLLAlign");

    printf("\033[1;34m---- hll_aligned ----:\033[0m\n");
    hllAligned_CUDA_product(hllAligned, serial, blockSize, performance);

    return 0;
}
