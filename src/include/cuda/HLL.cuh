//
// Created by buniy on 20/03/2025.
//

#ifndef CUDAHLL_CUH
#define CUDAHLL_CUH

#include "../mtxStructs.h"

__global__ void spmv_hll_parallel(HLLMatrix *hll, const MatVal *vector, ResultVector *result);
__global__ void spmv_hll_coalesced(HLLMatrix *hll, const MatVal *vector, ResultVector *result);
__global__ void spmv_hllAligned_parallel(HLLMatrixAligned *hll, const MatVal *vector, ResultVector *result);
__global__ void spmv_hllAligned_coalesced(HLLMatrixAligned* hll, const MatVal* __restrict__ vector, ResultVector* __restrict__ result);
#endif //CUDAHLL_CUH
