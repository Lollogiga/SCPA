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

int csr_product(CSRMatrix *h_csr, ResultVector *serial) {
    cudaError_t cuda_error;
    int int_err;
    
    float elapsedTime;
    double check;
    
    CSRMatrix *d_csr = uploadCSRToDevice(h_csr);
    if (d_csr == nullptr) {
        freeCSRDevice(d_csr);
        return -1;
    }

    int threadsPerBlock = BLOCK_SIZE;
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    int blocksPerGrid = (h_csr->M + warpsPerBlock - 1) / warpsPerBlock;

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

    MatVal* d_y;
    cuda_error = cudaMalloc(&d_y, h_csr->M * sizeof(MatVal));
    if (cuda_error != cudaSuccess) {
        printf("\033[31mcsr_product - cudaMalloc d_y failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        return -1;
    }

    CUDA_EVENT_CREATE(start, stop)

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
    printf("CudaSerial: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
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
    printf("csr_cudaProduct_sol1: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
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

    // SOL 2
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

    MatVal* d_product;
    cuda_error = cudaMalloc(&d_product, h_csr->NZ * sizeof(MatVal));
    if (cuda_error != cudaSuccess) {
        printf("\033[31mcsr_product - cudaMalloc d_product failed: %s\033[0m\n", cudaGetErrorString(cuda_error));

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }

    blocksPerGrid = (h_csr->NZ + warpsPerBlock - 1) / warpsPerBlock;

    cuda_error = cudaGetLastError();
    if (cuda_error) {
        printf("\033[31mMy supposition: %s\033[0m\n", cudaGetErrorString(cuda_error));

        return -1;
    }

    CUDA_EVENT_START(start)
    csr_cudaProduct_sol2_product<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_product, d_x);
    cudaDeviceSynchronize();

    blocksPerGrid = (h_csr->M + warpsPerBlock - 1) / warpsPerBlock;
    csr_cudaProduct_sol2_reduce<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_product, d_result_vector);
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
    printf("csr_cudaProduct_sol2: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
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

        freeCSRDevice(d_csr);
        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        freeResultVectorFromDevice(d_result_vector);
        CUDA_EVENT_DESTROY(start, stop)
        return -1;
    }
    freeResultVectorFromDevice(d_result_vector);
    cudaFree(d_product);

    // SOL 3
    for (int i = 0; i < h_csr->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);

    CUDA_EVENT_START(start)
    csr_cudaProduct_sol3<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSol2: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    check = checkResultVector(serial, h_result_vector);
    if (check) {
        printf("check = %.0f\n", check);
        perror("\033[31mError checkResultVector in CudaSol2\033[0m\n");

        return -1;
    }
    freeResultVectorFromDevice(d_result_vector);

    freeCSRDevice(d_csr);
    cudaFree(d_x);
    cudaFree(d_y);

    CUDA_EVENT_DESTROY(start, stop)

    free(h_x);

    return 0;
}

extern "C" int computeCUDA(CSRMatrix *csr, HLLMatrix *hll, HLLMatrixAligned *hllAligned, int num_threads) {
    MatVal *vector = create_vector(csr->N);
    if (vector == nullptr) return -1;

    ResultVector *serial = csr_serialProduct(csr, vector);

    csr_product(csr, serial);
    // hll_CUDA_product(hll, serial);

    int sharedMemPerBlock;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);


    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);

    // printf("Nome GPU: %s\n", prop.name);
    // printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    // printf("Warp size: %d\n", prop.warpSize);
    // printf("Max threads per multiprocessore: %d\n", prop.maxThreadsPerMultiProcessor);
    // printf("Numero di multiprocessori: %d\n", prop.multiProcessorCount);
    // printf("Max blocchi per griglia: %d\n", prop.maxGridSize[0]);
    // printf("Memoria condivisa per blocco: %zu bytes\n", prop.sharedMemPerBlock);

    return 0;
}
