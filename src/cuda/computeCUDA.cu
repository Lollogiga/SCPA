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
 * Max threads per block: 1024
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
__device__ inline MatVal warpReduceSum(MatVal val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void spmv_csr_warp(CSRMatrix *csr, MatVal *x, ResultVector *result) {
    int warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;  // Warp globale
    int lane = threadIdx.x % WARP_SIZE;  // ID del thread nel warp

    if (warp_id < csr->M) {  // Ogni warp lavora su una riga
        MatVal sum = 0.0;
        int row_start = csr->IRP[warp_id];
        int row_end = csr->IRP[warp_id + 1];

        // Ogni thread nel warp processa un pezzo della riga
        for (int i = row_start + lane; i < row_end; i += WARP_SIZE) {
            sum += csr->AS[i] * x[csr->JA[i]];
        }

        // Riduzione della somma tra i thread del warp
        sum = warpReduceSum(sum);

        // Scriviamo il risultato solo dal primo thread del warp
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

__global__ void spmv_csr_kernel_shared_memery_2(CSRMatrix *csr, float *x, ResultVector *result, int max_nnz) {
    int row = blockIdx.x;  // Un blocco per riga
    int tid = threadIdx.x; // Thread all'interno del blocco

    int start = csr->IRP[row];
    int end = csr->IRP[row + 1];

    MatVal sum = 0.0f;
    for (int j = start + tid; j < end; j += blockDim.x) {
        sum += csr->AS[j] * x[csr->JA[j]];
    }

    // Riduzione parallela nella memoria condivisa
    __shared__ float shared_sum[1024];
    shared_sum[tid] = sum;
    __syncthreads();

    // Riduzione finale (warp shuffle)
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result->val[row] = shared_sum[0];
    }
}

int csr_product(CSRMatrix *h_csr, ResultVector *serial) {
    CUDA_EVENT_CREATE(start, stop)

    float elapsedTime;

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

    CUDA_EVENT_START(start)
    spmv_csr_serial<<<1, 1>>>(d_csr, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSerial: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (checkResultVector(serial, h_result_vector)) {
        perror("Error checkResultVector in spmv_csr_serial \n");

        // return -1;
    }

    CUDA_EVENT_START(start)
    spmv_csr_kernel_sol1<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSol1: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (checkResultVector(serial, h_result_vector)) {
        perror("Error checkResultVector in CudaSol1 \n");

        // return -1;
    }

    CUDA_EVENT_START(start)
    spmv_csr_warp<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSol2: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (checkResultVector(serial, h_result_vector)) {
        perror("Error checkResultVector in CudaSol2 \n");

        // return -1;
    }

    CUDA_EVENT_START(start)
    spmv_csr_shared<<<blocksPerGrid, threadsPerBlock>>>(d_csr, d_x, d_result_vector);
    CUDA_EVENT_STOP(stop)
    CUDA_EVENT_ELAPSED(start, stop, elapsedTime)
    printf("CudaSol3: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (checkResultVector(serial, h_result_vector)) {
        perror("Error checkResultVector in CudaSol3 \n");

        // return -1;
    }

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
    printf("CudaSol4: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime));
    downloadResultVectorToHost(h_result_vector, d_result_vector);
    if (checkResultVector(serial, h_result_vector)) {
        perror("Error checkResultVector in CudaSol4 \n");

        // return -1;
    }

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
