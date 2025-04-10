#include "../../include/cuda/CSR.cuh"

// Sol 1
__global__ void csr_cudaProduct_sol1(CSRMatrix *csr, MatVal *v, ResultVector *result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < csr->M) {
        MatVal sum = 0.0;
        for (int j = csr->IRP[row]; j < csr->IRP[row + 1]; j++) {
            sum += csr->AS[j] * v[csr->JA[j]];
        }

        result->val[row] = sum;
    }
}

// Sol 2
__global__ void csr_cudaProduct_sol2_product(CSRMatrix *csr, MatVal *v_product, MatVal *x) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < csr->NZ) {
        v_product[index] = csr->AS[index] * x[csr->JA[index]];
    }
}

__global__ void csr_cudaProduct_sol2_reduce(CSRMatrix *csr, MatVal *v_product, ResultVector *result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < csr->M) {
        int row_start = csr->IRP[row];
        int row_end = csr->IRP[row + 1];

        for (int i = row_start; i < row_end; i++) {
            result->val[row] += v_product[i];
        }
    }
}

// Sol 3
__device__ inline MatVal warpReduceSum(MatVal val, const unsigned int mask) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void csr_cudaProduct_sol3(CSRMatrix *csr, MatVal *v, ResultVector *result) {
    int warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (warp_id < csr->M) {
        int row_start = csr->IRP[warp_id];
        int row_end = csr->IRP[warp_id + 1];
        MatVal sum = 0.0;

        for (int i = row_start + lane; i < row_end; i += WARP_SIZE) {
            if (i < row_end) sum += csr->AS[i] * v[csr->JA[i]];
        }

        unsigned int mask = __ballot_sync(0xFFFFFFFF, (row_start + lane) < row_end);
        if (row_start + lane >= row_end) {
            mask = 0;
        }

        sum = warpReduceSum(sum, mask);

        if (lane == 0) {
            result->val[warp_id] = sum;
        }
    }
}