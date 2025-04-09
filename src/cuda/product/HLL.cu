#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/cuda/Serial.cuh"
#include "../../include/cuda/HLL.cuh"
#include "../../include/createVector.h"
#include "../../include/cuda/Utils.cuh"

#define WARP_SIZE 32 // Impostato a 32 rispetto al server di dipartimento con sopra montata una Quadro RTX 5000
#define BLOCK_SIZE 1024 // Il server ha a disposizione al massimo 1024 thread attivi contemporaneamente
#define MAX_NZ_PER_BLOCK 1024

// Funzione che carica la matrice HLL sulla GPU e la restituisce come puntatore
HLLMatrix* uploadHLLToDevice(HLLMatrix *hll) {
    HLLMatrix *d_hll;

    // Allocazione memoria per la struttura HLLMatrix sulla GPU
    cudaMalloc((void**)&d_hll, sizeof(HLLMatrix));

    // Copia della struttura HLLMatrix dalla memoria host alla memoria device
    cudaMemcpy(d_hll, hll, sizeof(HLLMatrix), cudaMemcpyHostToDevice);

    // Allocazione della memoria per i blocchi ELLPACKMatrix sulla GPU
    cudaMalloc((void**)&(d_hll->blocks), hll->numBlocks * sizeof(ELLPACKMatrix*));

    // Ciclo per copiare ogni blocco ELLPACKMatrix sulla GPU
    for (int blockIdx = 0; blockIdx < hll->numBlocks; blockIdx++) {
        ELLPACKMatrix *block = hll->blocks[blockIdx];
        ELLPACKMatrix *d_block;

        // Allocazione memoria per ogni blocco ELLPACKMatrix sulla GPU
        cudaMalloc((void**)&d_block, sizeof(ELLPACKMatrix));

        // Copia della struttura ELLPACKMatrix dalla memoria host alla memoria device
        cudaMemcpy(d_block, block, sizeof(ELLPACKMatrix), cudaMemcpyHostToDevice);

        // Allocazione memoria per i dati del blocco (JA e AS)
        cudaMalloc((void**)&(d_block->JA), block->M * block->MAXNZ * sizeof(MatT));
        cudaMalloc((void**)&(d_block->AS), block->M * block->MAXNZ * sizeof(MatVal));

        // Copia dei dati di ciascun blocco (JA e AS) dalla memoria host alla memoria device
        for (MatT row = 0; row < block->M; row++) {
            cudaMemcpy(d_block->JA[row], block->JA[row], block->MAXNZ * sizeof(MatT), cudaMemcpyHostToDevice);
            cudaMemcpy(d_block->AS[row], block->AS[row], block->MAXNZ * sizeof(MatVal), cudaMemcpyHostToDevice);
        }

        // Aggiornamento dell'array di blocchi nella memoria del dispositivo
        cudaMemcpy(&(d_hll->blocks[blockIdx]), &d_block, sizeof(ELLPACKMatrix*), cudaMemcpyHostToDevice);
    }

    // Restituzione del puntatore alla matrice HLLMatrix sulla GPU
    return d_hll;
}


__global__ void ellpack_CUDA_product_kernel(ELLPACKMatrix *d_block, MatVal *d_vector, MatVal *d_result) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < d_block->M) {
        MatT *JA = d_block->JA[row];
        MatVal *AS = d_block->AS[row];

        // Moltiplicazione della matrice per il vettore, con somma atomica
        for (int i = 0; i < d_block->MAXNZ; i++) {
            if (JA[i] >= 0) {
                atomicAdd(&d_result[row], AS[i] * d_vector[JA[i]]);
            }
        }
    }
}


ResultVector *hll_CUDA_product(HLLMatrix *h_hll, ResultVector *serial_vector) {
    if (!h_hll || !serial_vector) {
        perror("hll_CUDA_product: NULL pointer detected");
        return NULL;
    }

    // Creazione del vettore risultato sulla GPU
    ResultVector *h_result_vector = create_result_vector(h_hll->M);  // Crea il risultato host
    ResultVector *d_result_vector = uploadResultVectorToDevice(h_result_vector);  // Copia sulla GPU

    // Creazione del vettore di input sulla GPU
    MatVal *h_vector = create_vector(h_hll->N);

    MatVal* d_vector;
    cudaMalloc(&d_vector, h_hll->N * sizeof(MatVal));
    cudaMemcpy(d_vector, h_vector, h_hll->N * sizeof(MatVal), cudaMemcpyHostToDevice);
    cudaMemAdvise(d_vector, h_hll->N * sizeof(MatVal), cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(d_vector, h_hll->N * sizeof(MatVal), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

    // Ciclo sui blocchi HLL
    for (int blockIdx = 0; blockIdx < h_hll->numBlocks; blockIdx++) {
        ELLPACKMatrix *d_block;
        cudaMemcpy(&d_block, &h_hll->blocks[blockIdx], sizeof(ELLPACKMatrix*), cudaMemcpyHostToDevice);

        // Definizione della griglia di thread
        dim3 threadsPerBlock(256); // Numero di thread per blocco
        dim3 numBlocks((d_block->M + threadsPerBlock.x - 1) / threadsPerBlock.x);

        // Lancio del kernel CUDA
        ellpack_CUDA_product_kernel<<<numBlocks, threadsPerBlock>>>(d_block, d_vector, d_result_vector->val);

        // Controllo errori CUDA
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
    }

    // Copia dei risultati dalla GPU alla memoria host
    cudaMemcpy(h_result_vector->val, d_result_vector->val, h_hll->M * sizeof(MatVal), cudaMemcpyDeviceToHost);

    // Pulizia della memoria GPU
    cudaFree(d_vector);
    cudaFree(d_result_vector);

    // Restituisce il vettore risultato
    return h_result_vector;
}


