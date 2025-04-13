#include "../../include/cuda/Serial.cuh"

#include <cstdio>

// CSR serial CUDA
__global__ void spmv_csr_serial(CSRMatrix *csr, MatVal *vector, ResultVector *result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int row = 0; row < csr->M; row++) {
            MatVal sum = 0.0f;
            int row_start = csr->IRP[row];
            int row_end = csr->IRP[row + 1];

            for (int i = row_start; i < row_end; i++) {
                MatT col = csr->JA[i];
                MatVal val = csr->AS[i];
                sum += val * vector[col];
            }

            result->val[row] = sum;
        }
    }
}

__device__ void *ellpack_serialProduct(ELLPACKMatrix *ell, const MatVal *vector, MatVal *result) {
    for (MatT i = 0; i < ell->M; i++) {
        for (MatT j = 0; j < ell->MAXNZ; j++) {
            MatT col_index = ell->JA[i][j];

            result[i] += ell->AS[i][j] * vector[col_index];
        }
    }

    return result;
}

__global__ void spmv_hll_serial(HLLMatrix *hll, MatVal *vector, ResultVector *result){
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (MatT i = 0; i < hll->numBlocks; i++)
        {
            ELLPACKMatrix *block = hll->blocks[i];
            ellpack_serialProduct(block, vector, result->val + block->startRow);
        }
    }
}
