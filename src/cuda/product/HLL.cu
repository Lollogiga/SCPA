#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/cuda/Serial.cuh"
#include "../../include/cuda/HLL.cuh"
#include "../../include/createVector.h"
#include "../../include/cuda/Utils.cuh"
#include "../../include/checkResultVector.h"
#include "../../include/flops.h"

#define WARP_SIZE 32
#define BLOCK_SIZE 1024
#define MAX_NZ_PER_BLOCK 1024

// Kernel per l'elaborazione di un singolo blocco HLL
__global__ void hll_process_block_kernel(
    ELLPACKMatrix *block,
    MatVal *d_vector,
    MatVal *d_result,
    int blockGlobalOffset
) {
    int localRow = threadIdx.x + blockIdx.x * blockDim.x;
    if (localRow < block->M) {
        MatVal sum = 0.0;
        for (int j = 0; j < block->MAXNZ; ++j) {
            int col = block->JA[localRow][j];
            MatVal val = block->AS[localRow][j];
            sum += val * d_vector[col];
        }
        d_result[blockGlobalOffset + localRow] = sum;
    }
}

// Upload completo di HLL e blocchi sul device
HLLMatrix* uploadHLLToDevice(HLLMatrix *hll, ELLPACKMatrix ***d_blocks_ptr) {
    HLLMatrix *d_hll;
    cudaMalloc((void**)&d_hll, sizeof(HLLMatrix));
    cudaMemcpy(d_hll, hll, sizeof(HLLMatrix), cudaMemcpyHostToDevice);

    // Array device di puntatori ai blocchi
    ELLPACKMatrix **d_blocks;
    cudaMalloc((void**)&d_blocks, hll->numBlocks * sizeof(ELLPACKMatrix*));

    for (int b = 0; b < hll->numBlocks; b++) {
        ELLPACKMatrix *h_block = hll->blocks[b];
        ELLPACKMatrix *d_block;

        cudaMalloc((void**)&d_block, sizeof(ELLPACKMatrix));
        cudaMemcpy(d_block, h_block, sizeof(ELLPACKMatrix), cudaMemcpyHostToDevice);

        size_t size = h_block->M * h_block->MAXNZ;
        MatT *flat_JA;
        MatVal *flat_AS;

        cudaMalloc((void**)&flat_JA, size * sizeof(MatT));
        cudaMalloc((void**)&flat_AS, size * sizeof(MatVal));
        cudaMemcpy(flat_JA, h_block->JA[0], size * sizeof(MatT), cudaMemcpyHostToDevice);
        cudaMemcpy(flat_AS, h_block->AS[0], size * sizeof(MatVal), cudaMemcpyHostToDevice);

        MatT **d_JA_rows;
        MatVal **d_AS_rows;
        cudaMalloc(&d_JA_rows, h_block->M * sizeof(MatT*));
        cudaMalloc(&d_AS_rows, h_block->M * sizeof(MatVal*));

        MatT **h_JA_rows = (MatT**)malloc(h_block->M * sizeof(MatT*));
        MatVal **h_AS_rows = (MatVal**)malloc(h_block->M * sizeof(MatVal*));
        for (int i = 0; i < h_block->M; ++i) {
            h_JA_rows[i] = flat_JA + i * h_block->MAXNZ;
            h_AS_rows[i] = flat_AS + i * h_block->MAXNZ;
        }

        cudaMemcpy(d_JA_rows, h_JA_rows, h_block->M * sizeof(MatT*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_AS_rows, h_AS_rows, h_block->M * sizeof(MatVal*), cudaMemcpyHostToDevice);

        cudaMemcpy(&(d_block->JA), &d_JA_rows, sizeof(MatT**), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_block->AS), &d_AS_rows, sizeof(MatVal**), cudaMemcpyHostToDevice);

        cudaMemcpy(&(d_blocks[b]), &d_block, sizeof(ELLPACKMatrix*), cudaMemcpyHostToDevice);

        free(h_JA_rows);
        free(h_AS_rows);
    }

    cudaMemcpy(&(d_hll->blocks), &d_blocks, sizeof(ELLPACKMatrix**), cudaMemcpyHostToDevice);

    *d_blocks_ptr = d_blocks;
    return d_hll;
}

// Funzione principale per il prodotto parallelo con HLL
ResultVector* hll_CUDA_product(HLLMatrix *h_hll, ResultVector *serial_vector) {
    if (!h_hll || !serial_vector) {
        perror("hll_CUDA_product: NULL pointer detected");
        return NULL;
    }

    ResultVector *h_result_vector = create_result_vector(h_hll->M);
    MatVal *h_vector = create_vector(h_hll->N); // oppure copia da serial_vector->val

    // Allocazione e copia vettore input e risultato su device
    MatVal *d_vector, *d_result;
    cudaMalloc(&d_vector, h_hll->N * sizeof(MatVal));
    cudaMalloc(&d_result, h_hll->M * sizeof(MatVal));
    cudaMemcpy(d_vector, h_vector, h_hll->N * sizeof(MatVal), cudaMemcpyHostToDevice);

    // Upload HLL e blocchi
    ELLPACKMatrix **d_blocks;
    HLLMatrix *d_hll = uploadHLLToDevice(h_hll, &d_blocks);

    // Lancio di un kernel per ogni blocco
    for (int b = 0; b < h_hll->numBlocks; ++b) {
        ELLPACKMatrix *block = h_hll->blocks[b];
        int blockM = block->M;
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (blockM + threadsPerBlock - 1) / threadsPerBlock;

        hll_process_block_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_blocks[b],
            d_vector,
            d_result,
            block->startRow
        );
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_result_vector->val, d_result, h_hll->M * sizeof(MatVal), cudaMemcpyDeviceToHost);

    double check = checkResultVector(serial_vector, h_result_vector);
    if (check) {
        fprintf(stderr, "\033[1;31m[Error] checkResultVector failed in hll_CUDA_product: %f\033[0m\n", check);
    }

    cudaFree(d_vector);
    cudaFree(d_result);
    return h_result_vector;
}
