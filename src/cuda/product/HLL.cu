#include <cuda_runtime.h>
#include <stdio.h>

#include "../../include/cuda/Serial.cuh"
#include "../../include/cuda/HLL.cuh"
#include "../../include/createVector.h"
#include "../../include/cuda/Utils.cuh"
#include "../../include/flops.h"
#include "../../include/checkResultVector.h"

#define WARP_SIZE 32 // Impostato a 32 rispetto al server di dipartimento con sopra montata una Quadro RTX 5000
#define BLOCK_SIZE 1024 // Il server ha a disposizione al massimo 1024 thread attivi contemporaneamente
#define MAX_NZ_PER_BLOCK 1024


__global__ void spmv_hll_parallel(HLLMatrix *hll, const MatVal *vector, ResultVector *result) {
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;

    if(block_id >= hll->numBlocks) return;

    ELLPACKMatrix *blk = hll->blocks[block_id];

    if(thread_id >= blk->M) return;

    MatVal sum = 0.0;
#pragma unroll
    for(MatT j = 0; j < blk->MAXNZ; j++) {
        const MatT col = blk->JA[thread_id][j];
        sum += blk->AS[thread_id][j] * vector[col];
    }
    result->val[blk->startRow + thread_id] = sum;
}

__global__ void spmv_hll_coalesced(HLLMatrix *hll, const MatVal *vector, ResultVector *result) {
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;

    if(block_id >= hll->numBlocks) return;

    ELLPACKMatrix *blk = hll->blocks[block_id];

    if(thread_id >= blk->M) return;

    MatVal sum = 0.0;
#pragma unroll
    for(MatT j = 0; j < blk->MAXNZ; j++) {
        const MatT col = blk->JA[thread_id][j];
        sum += blk->AS[thread_id][j] * vector[col];
    }

    // Coalescenza: Accesso contiguo alla memoria
    result->val[blk->startRow + thread_id] = sum;
}


int hll_CUDA_product(HLLMatrix *h_hll, ResultVector *serial_result)
{
    cudaError_t cuda_error;
    int int_err;

    float elapsedTime;

    if (!h_hll || !serial_result)
    {
       perror("Hll_cuda_product failed: Input not initialized");
        return -1;
    }

    //Create vector:
    MatVal *h_x = create_vector(h_hll->N);
    if (h_x == nullptr)
    {
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

    //Create result_vector:
    ResultVector *h_result_vector = create_result_vector(h_hll->M);
    if (h_result_vector == nullptr)
    {
        perror("Hll_cuda_product failed: create_result_vector failed");
        return -1;
    }

    // Upload result_vector on device
    ResultVector *d_result_vector = uploadResultVectorToDevice(h_result_vector);
    if (d_result_vector == nullptr) {
        printf("\033[31mhll_product - uploadResultVectorToDevice h_result_vector failed\033[0m\n");

        free_vector(h_x);
        cudaFree(d_x);
        free_result_vector(h_result_vector);
        return -1;
    }

    // Upload hll to device:
    HLLMatrix *d_hll = uploadHLLToDevice(h_hll);
    if (d_hll == nullptr)
    {
        perror("Hll_cuda_product failed: uploadHLLToDevice failed");
        free_result_vector(d_result_vector);
        free_vector(h_x);
        cudaFree(d_x);
        freeHLLDevice(d_hll);

        return -1;
    }

    // Serial solution:
    CUDA_EVENT_CREATE(start, stop)

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
    printf("CudaSerial: Flops: %f\n", computeFlops(h_hll->NZ, elapsedTime));
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
    int_err = checkResultVector(serial_result, h_result_vector);
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


    // Solution 1:
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

    dim3 grid(h_hll->numBlocks);
    dim3 block(h_hll->hackSize);

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
    printf("CudaSol1: Flops: %f\n", computeFlops(h_hll->NZ, elapsedTime));
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
    int_err = checkResultVector(serial_result, h_result_vector);
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


    // Solution 1:
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
    spmv_hll_coalesced<<<grid, block>>>(d_hll, d_x, d_result_vector);
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
    printf("CudaSol1: Flops: %f\n", computeFlops(h_hll->NZ, elapsedTime));
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
    int_err = checkResultVector(serial_result, h_result_vector);
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



    return 0;
}


