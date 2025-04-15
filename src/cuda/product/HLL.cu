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

    ELLPACKMatrix *block = hll->blocks[block_id];

    if(thread_id >= block->M) return;

    MatVal sum = 0.0;
#pragma unroll
    for(MatT j = 0; j < block->MAXNZ; j++) {
        const MatT col = block->JA[thread_id][j];
        sum += block->AS[thread_id][j] * vector[col];
    }
    result->val[block->startRow + thread_id] = sum;
}

__global__ void spmv_hll_coalesced(HLLMatrix *hll, const MatVal *vector, ResultVector *result) {
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;

    if(block_id >= hll->numBlocks) return;

    ELLPACKMatrix *block = hll->blocks[block_id];

    if(thread_id >= block->M) return;

    MatVal sum = 0.0;
#pragma unroll
    for(MatT j = 0; j < block->MAXNZ; j++) {
        const MatT col = block->JA[thread_id][j];
        sum += block->AS[thread_id][j] * vector[col];
    }

    // Coalescenza: Accesso contiguo alla memoria
    result->val[block->startRow + thread_id] = sum;
}

// Versione base parallela
__global__ void spmv_hllAligned_parallel(HLLMatrixAligned *hll, const MatVal *vector, ResultVector *result) {
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;

    if(block_id >= hll->numBlocks) return;

    ELLPACKMatrixAligned* block = hll->blocks[block_id];

    if(thread_id >= block->M) return;

    MatVal sum = 0.0;
#pragma unroll
    for(MatT j = 0; j < block->MAXNZ; j++) {
        const MatT col = block->JA[thread_id * block->MAXNZ + j];
        sum += block->AS[thread_id * block->MAXNZ + j] * vector[col];
    }
    result->val[block->startRow + thread_id] = sum;
}

__global__ void spmv_hllAligned_coalesced(HLLMatrixAligned* hll, const MatVal* __restrict__ vector, ResultVector* __restrict__ result) {
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int total_threads = blockDim.x;

    if(block_id >= hll->numBlocks) return;

    ELLPACKMatrixAligned* block = hll->blocks[block_id];
    const MatT rows_per_thread = (block->M + total_threads - 1) / total_threads;

    for(MatT r = 0; r < rows_per_thread; r++) {
        const MatT row = thread_id * rows_per_thread + r;
        if(row >= block->M) break;

        MatVal sum = 0.0;
#pragma unroll
        for(MatT j = 0; j < block->MAXNZ; j++) {
            const MatT idx = row * block->MAXNZ + j;
            sum += block->AS[idx] * vector[block->JA[idx]];
        }
        result->val[block->startRow + row] = sum;
    }
}



