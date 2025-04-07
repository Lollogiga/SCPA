#include "../../include/cuda/Serial.cuh"

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