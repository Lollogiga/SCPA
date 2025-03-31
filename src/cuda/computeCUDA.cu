#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "../include/mtxStructs.h"
#include "../include/flops.h"

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_NZ_PER_BLOCK 1024

// CSR serial CUDA
__global__ void spmv_csr_serial(int M, MatT *IRP, MatT *JA, MatVal *AS, MatVal *x, MatVal *y) {
    // Ogni thread calcola il prodotto per una singola riga (seriale)
    int row = 0;  // Solo un thread, quindi lavoriamo su una riga alla volta

    if (row < M) {
        MatVal sum = 0.0f;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];

        // Calcola il prodotto matrice-vettore per la riga
        for (int i = row_start; i < row_end; i++) {
            MatT col = JA[i];  // Colonna dell'elemento non nullo
            MatVal val = AS[i];  // Valore non nullo
            sum += val * x[col];  // Prodotto scala matrice-vettore
        }

        y[row] = sum;  // Risultato del prodotto per la riga
    }
}

// Test 1
__global__ void spmv_csr_kernel_sol1(int M, MatT *IRP, MatT *JA, MatVal *AS, MatVal *x, MatVal *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        MatVal sum = 0.0;
        for (int j = IRP[row]; j < IRP[row + 1]; j++) {
            sum += AS[j] * x[JA[j]];
        }
        y[row] = sum;
    }
}

// Test 2
__device__ inline MatVal warpReduceSum(MatVal val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void spmv_csr_warp(int M, MatT *IRP, MatT *JA, MatVal *AS, MatVal *x, MatVal *y) {
    int warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;  // Warp globale
    int lane = threadIdx.x % WARP_SIZE;  // ID del thread nel warp

    if (warp_id < M) {  // Ogni warp lavora su una riga
        MatVal sum = 0.0;
        int row_start = IRP[warp_id];
        int row_end = IRP[warp_id + 1];

        // Ogni thread nel warp processa un pezzo della riga
        for (int i = row_start + lane; i < row_end; i += WARP_SIZE) {
            sum += AS[i] * x[JA[i]];
        }

        // Riduzione della somma tra i thread del warp
        sum = warpReduceSum(sum);

        // Scriviamo il risultato solo dal primo thread del warp
        if (lane == 0) {
            y[warp_id] = sum;
        }
    }
}

// Test 3
__global__ void spmv_csr_shared(int M, MatT *IRP, MatT *JA, MatVal *AS, MatVal *x, MatVal *y) {
    __shared__ MatVal shared_AS[MAX_NZ_PER_BLOCK];  // Memoria condivisa per i valori della matrice
    __shared__ MatT shared_JA[MAX_NZ_PER_BLOCK];   // Memoria condivisa per gli indici di colonna

    int warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int local_tid = threadIdx.x;

    if (warp_id >= M) return;

    // Lettura dell'intervallo della riga
    int row_start = IRP[warp_id];
    int row_end = IRP[warp_id + 1];
    int row_size = row_end - row_start;

    // Caricamento in memoria condivisa
    for (int i = local_tid; i < row_size; i += BLOCK_SIZE) {
        shared_AS[i] = AS[row_start + i];
        shared_JA[i] = JA[row_start + i];
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
        atomicAdd(&y[warp_id], warp_sums[threadIdx.x / WARP_SIZE]);
    }
}

int csr_product(CSRMatrix *h_csr) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsedTime;

    CSRMatrix d_csr;
    cudaMalloc((void**)&d_csr.IRP, (h_csr->M + 1) * sizeof(MatT));
    cudaMalloc((void**)&d_csr.JA, h_csr->NZ * sizeof(MatT));
    cudaMalloc((void**)&d_csr.AS, h_csr->NZ * sizeof(MatVal));

    cudaMemcpy(d_csr.IRP, h_csr->IRP, (h_csr->M + 1) * sizeof(MatT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.JA, h_csr->JA, h_csr->NZ * sizeof(MatT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.AS, h_csr->AS, h_csr->NZ * sizeof(MatVal), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    int blocksPerGrid = (h_csr->M + warpsPerBlock - 1) / warpsPerBlock;

    MatVal* h_x = (MatVal*) malloc(h_csr->N * sizeof(MatVal));
    if (h_x == NULL) {
        perror("malloc");
        return -1;
    }
    for (int i = 0; i < h_csr->N; i++) h_x[i] = 1.0;

    MatVal* d_x;
    cudaMalloc(&d_x, h_csr->N * sizeof(MatVal));
    cudaMemcpy(d_x, h_x, h_csr->N * sizeof(MatVal), cudaMemcpyHostToDevice);
    cudaMemAdvise(d_x, h_csr->N * sizeof(MatVal), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(d_x, h_csr->N * sizeof(MatVal), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

    MatVal* d_y;
    cudaMalloc(&d_y, h_csr->M * sizeof(MatVal));\

    cudaEventRecord(start);
    spmv_csr_serial<<<1, 1>>>(h_csr->M, d_csr.IRP, d_csr.JA, d_csr.AS, d_x, d_y);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CudaSerial: Elapsed time: %f s\n", elapsedTime);
    printf("CudaSerial: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime / 1000));

    cudaEventRecord(start);
    spmv_csr_kernel_sol1<<<blocksPerGrid, threadsPerBlock>>>(h_csr->M, d_csr.IRP, d_csr.JA, d_csr.AS, d_x, d_y);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CudaSol1: Elapsed time: %f s\n", elapsedTime);
    printf("CudaSol1: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime / 1000));

    cudaEventRecord(start);
    spmv_csr_warp<<<blocksPerGrid, threadsPerBlock>>>(h_csr->M, d_csr.IRP, d_csr.JA, d_csr.AS, d_x, d_y);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CudaSol2: Elapsed time: %f s\n", elapsedTime);
    printf("CudaSol2: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime / 1000));

    cudaEventRecord(start);
    spmv_csr_shared<<<blocksPerGrid, threadsPerBlock>>>(h_csr->M, d_csr.IRP, d_csr.JA, d_csr.AS, d_x, d_y);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CudaSol3: Elapsed time: %f s\n", elapsedTime);
    printf("CudaSol3: Flops: %f\n", computeFlops(h_csr->NZ, elapsedTime / 1000));

    cudaFree(d_csr.IRP);
    cudaFree(d_csr.JA);
    cudaFree(d_csr.AS);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_x);

    return 0;
}

extern "C" int computeCUDA(CSRMatrix *h_csr, HLLMatrix *h_hll, HLLMatrixAligned *h_hllAligned, int num_threads) {
    csr_product(h_csr);

    return 0;
}
