#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "../include/openmp/Serial.h"
#include "../include/cuda/Serial.cuh"
#include "../include/cuda/Utils.cuh"
#include "../include/checkResultVector.h"
#include "../include/mtxStructs.h"
#include "../include/flops.h"

#include "../include/cuda/HLL.cuh"
/*
 * --- NOTE SULLA SCHEDA VIDEO SUL SERVER DI DIPARTIMENTO ---
 * Nome GPU: Quadro RTX 5000
 * Max threads per block: 1024#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/cuda/Serial.cuh"
#include "../../include/cuda/HLL.cuh"
#include "../../include/createVector.h"
#include "../../include/cuda/Utils.cuh"
 * Warp size: 32
 * Max threads per multiprocessore: 1024
 * Numero di multiprocessori: 48
 * Max blocchi per griglia: 2147483647
 * Memoria condivisa per blocco: 49152 bytes
 */

#define WARP_SIZE 32 // Impostato a 32 rispetto al server di dipartimento con sopra montata una Quadro RTX 5000
#define BLOCK_SIZE 1024 // Il server ha a disposizione al massimo 1024 thread attivi contemporaneamente
#define MAX_NZ_PER_BLOCK 1024

// Test 1
__global__ void spmv_csr_kernel_sol1(CSRMatrix *csr, MatVal *x, ResultVector *result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < csr->M) {
        MatVal sum = 0.0;
        for (int j = csr->IRP[row]; j < csr->IRP[row + 1]; j++) {
            sum += csr->AS[j] * x[csr->JA[j]];
        }

        result->val[row] = sum;
    }
}

// Test 2
__device__ inline MatVal warpReduceSum(MatVal val, unsigned int mask) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ inline float warpReduceSumKahan(MatVal val, unsigned int mask) {
    MatVal sum = val;
    float c = 0.0f;  // compensazione

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float y = __shfl_down_sync(mask, sum, offset) - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

__global__ void spmv_csr_warp(CSRMatrix *csr, const MatVal *x, ResultVector *result) {
    int warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (warp_id < csr->M) {
        int row_start = csr->IRP[warp_id];
        int row_end = csr->IRP[warp_id + 1];
        MatVal sum = 0.0;

        for (int i = row_start + lane; i < row_end; i += WARP_SIZE) {
            if (i < row_end) sum += csr->AS[i] * x[csr->JA[i]];
        }

        unsigned int mask = __ballot_sync(0xFFFFFFFF, (row_start + lane) < row_end);
        if (row_start + lane >= row_end) {
            mask = 0;
        }

        sum = warpReduceSum(sum, mask);
        // sum = warpReduceSumKahan(sum, mask);

        if (lane == 0) {
            result->val[warp_id] = sum;
        }
    }
}

// Test 3
__global__ void spmv_csr_shared(CSRMatrix *csr, MatVal *x, ResultVector *result) {
    __shared__ MatVal shared_AS[MAX_NZ_PER_BLOCK];  // Memoria condivisa per i valori della matrice
    __shared__ MatT shared_JA[MAX_NZ_PER_BLOCK];   // Memoria condivisa per gli indici di colonna

    int warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int local_tid = threadIdx.x;

    if (warp_id >= csr->M) return;

    // Lettura dell'intervallo della riga
    int row_start = csr->IRP[warp_id];
    int row_end = csr->IRP[warp_id + 1];
    int row_size = row_end - row_start;

    // Caricamento in memoria condivisa
    for (int i = local_tid; i < row_size; i += BLOCK_SIZE) {
        shared_AS[i] = csr->AS[row_start + i];
        shared_JA[i] = csr->JA[row_start + i];
    }
    __syncthreads();  // Sincronizziamo per garantire che i dati siano caricati

    // Calcolo del prodotto
    MatVal sum = 0.0;
    for (int i = lane; i < row_size; i += WARP_SIZE) {
        sum += shared_AS[i] * x[shared_JA[i]];
    }

    // Riduzione nella memoria condivisa (usiamo solo il primo warp del blocco)
    __shared__ MatVal warp_sums[WARP_SIZE];
    if (lane == 0) warp_sums[threadIdx.x / WARP_SIZE] = 0;
    __syncthreads();

    atomicAdd(&warp_sums[threadIdx.x / WARP_SIZE], sum);
    __syncthreads();

    // Solo il primo thread scrive il risultato finale
    if (lane == 0) {
        atomicAdd(&result->val[warp_id], warp_sums[threadIdx.x / WARP_SIZE]);
    }
}

// Test 5
__global__ void find_max_nnz_per_row(int M, CSRMatrix *csr, int *max_nnz) {
    __shared__ int local_max;
    if (threadIdx.x == 0) local_max = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        int nnz = csr->IRP[i + 1] - csr->IRP[i];
        atomicMax(&local_max, nnz);
    }

    __syncthreads();
    if (threadIdx.x == 0) atomicMax(max_nnz, local_max);
}

__global__ void spmv_csr_kernel_shared_memery_2(CSRMatrix *csr, MatVal *x, ResultVector *result, int max_nnz) {
    int row = blockIdx.x;  // Un blocco per riga
    int tid = threadIdx.x; // Thread all'interno del blocco
    int block_size = blockDim.x;

    int start = csr->IRP[row];
    int end = csr->IRP[row + 1];

    MatVal sum = 0.0f;

    // Calcolo del prodotto matrice-vettore per la riga "row"
    for (int j = start + tid; j < end; j += block_size) {
        sum += csr->AS[j] * x[csr->JA[j]];
    }

    // Memoria condivisa per la somma parziale
    __shared__ double shared_sum[1024];
    shared_sum[tid] = sum;
    __syncthreads();

    // Riduzione parallela usando la memoria condivisa
    for (int offset = block_size / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    // Scrivi il risultato finale se il thread è il thread 0
    if (tid == 0) {
        result->val[row] = shared_sum[0];
    }
}

// Test 6
__global__ void spmv_crs_kernel_fullvector(CSRMatrix *csr, MatVal *product, MatVal *x) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < csr->NZ) {
        product[index] = csr->AS[index] * x[csr->JA[index]];
    }
}

__global__ void reduce_global_product(CSRMatrix *csr, MatVal *product, ResultVector *result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < csr->M) {
        int row_start = csr->IRP[row];
        int row_end = csr->IRP[row + 1];

        for (int i = row_start; i < row_end; i++) {
            result->val[row] += product[i];
        }
    }
}

// Test 7
__global__ void spmv_csr_block(const CSRMatrix *csr, const MatVal *x, ResultVector *result) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= csr->M) return;

    int row_start = csr->IRP[row];
    int row_end   = csr->IRP[row + 1];

    extern __shared__ float sdata[];  // shared memory per le somme locali
    float sum = 0.0f;

    // Step 1: ogni thread lavora su una porzione della riga
    for (int i = row_start + tid; i < row_end; i += block_size) {
        int col = csr->JA[i];
        float val = csr->AS[i];
        sum += val * x[col];
    }

    // Step 2: riduzione in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Riduzione standard (oppure Kahan qui se vuoi più stabilità)
    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    // Step 3: scrittura del risultato
    if (tid == 0) {
        result->val[row] = sdata[0];
    }
}

int csr_product(CSRMatrix *h_csr, ResultVector *serial) {
    CUDA_EVENT_CREATE(start, stop)

    float elapsedTime;
    double check;

    CSRMatrix *d_csr = uploadCSRToDevice(h_csr);

    int threadsPerBlock = BLOCK_SIZE;
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    int blocksPerGrid = (h_csr->M + warpsPerBlock - 1) / warpsPerBlock;

    MatVal *h_x = create_vector(h_csr->N);

    MatVal* d_x;
    cudaMalloc(&d_x, h_csr->N * sizeof(MatVal));
    cudaMemcpy(d_x, h_x, h_csr->N * sizeof(MatVal), cudaMemcpyHostToDevice);
    cudaMemAdvise(d_x, h_csr->N * sizeof(MatVal), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(d_x, h_csr->N * sizeof(MatVal), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

    ResultVector *h_result_vector = create_result_vector(h_csr->M);
    ResultVector *d_result_vector = uploadResultVectorToDevice(h_result_vector);

    MatVal* d_y;
    cudaMalloc(&d_y, h_csr->M * sizeof(MatVal));

    // SOL SERIAL
    CUDA_EVENT_START(start)
    spmv_csr_serial<<<1, 1>>>(d_csr, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSerial: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    check = checkResultVector(serial, h_result_vector);
    if (check) {
        perror("\033[31mError checkResultVector in spmv_csr_serial \033[0m\n");

        // return -1;
    }
    freeResultVectorFromDevice(d_result_vector);

    // SOL 1
    for (int i = 0; i < h_csr->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);

    CUDA_EVENT_START(start)
    spmv_csr_kernel_sol1<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSol1: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    check = checkResultVector(serial, h_result_vector);
    if (check) {
        perror("\033[31mError checkResultVector in CudaSol1 \033[0m\n");

        // return -1;
    }
    freeResultVectorFromDevice(d_result_vector);

    // SOL 6
    for (int i = 0; i < h_csr->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);

    MatVal* d_product;
    cudaMalloc(&d_product, h_csr->NZ * sizeof(MatVal));

    blocksPerGrid = (h_csr->NZ + warpsPerBlock - 1) / warpsPerBlock;

    CUDA_EVENT_START(start)
    spmv_crs_kernel_fullvector<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_product, d_x);
    cudaDeviceSynchronize();

    blocksPerGrid = (h_csr->M + warpsPerBlock - 1) / warpsPerBlock;
    reduce_global_product<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_product, d_result_vector);
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSol6: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (checkResultVector(serial, h_result_vector)) {
        perror("\033[31mError checkResultVector in CudaSol6 \033[0m\n");

        // return -1;
    }
    freeResultVectorFromDevice(d_result_vector);

    // SOL 2
    for (int i = 0; i < h_csr->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);

    CUDA_EVENT_START(start)
    spmv_csr_warp<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSol2: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    check = checkResultVector(serial, h_result_vector);
    if (check) {
        printf("check = %.0f\n", check);
        perror("\033[31mError checkResultVector in CudaSol2\033[0m\n");

        // for (int i = 0; i < h_csr->M; i++) {
        //     if (serial->val[i] != h_result_vector->val[i])
        //         printf("serial [%f] - [%f] cuda_sol2\n", serial->val[i], h_result_vector->val[i]);
        // }

        // return -1;
    }
    freeResultVectorFromDevice(d_result_vector);

    // SOL 3
    for (int i = 0; i < h_csr->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);

    CUDA_EVENT_START(start)
    spmv_csr_shared<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSol3: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (checkResultVector(serial, h_result_vector)) {
        perror("\033[31mError checkResultVector in CudaSol3 \033[0m\n");

        // return -1;
    }
    freeResultVectorFromDevice(d_result_vector);

    // SOL 5
    for (int i = 0; i < h_csr->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);

    int *d_max_nnz;
    cudaMalloc(&d_max_nnz, sizeof(int));
    cudaMemset(d_max_nnz, 0, sizeof(int));
    find_max_nnz_per_row<<<blocksPerGrid, threadsPerBlock>>>(h_csr->M, d_csr, d_max_nnz);
    int max_nnz_per_row;
    cudaMemcpy(&max_nnz_per_row, d_max_nnz, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_max_nnz);
    CUDA_EVENT_START(start)
    spmv_csr_kernel_shared_memery_2<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_x, d_result_vector, max_nnz_per_row);
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSol5: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (checkResultVector(serial, h_result_vector)) {
        perror("\033[31mError checkResultVector in CudaSol5 \033[0m\n");

        // return -1;
    }
    freeResultVectorFromDevice(d_result_vector);

    // SOL 7
    int block_size = 64;
    int num_rows = h_csr->M;
    size_t shared_mem_bytes = block_size * sizeof(MatVal);
    for (int i = 0; i < h_csr->M; i++) h_result_vector->val[i] = 0;
    d_result_vector = uploadResultVectorToDevice(h_result_vector);

    CUDA_EVENT_START(start)
    spmv_csr_block<<<num_rows, block_size, shared_mem_bytes>>>(
        d_csr, d_x, d_result_vector
    );
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSol7: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (checkResultVector(serial, h_result_vector)) {
        perror("\033[31mError checkResultVector in CudaSol7 \033[0m\n");

        // return -1;
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
    ResultVector *serial = csr_serialProduct(csr, vector);

    csr_product(csr, serial);
    hll_CUDA_product(hll, serial);

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
