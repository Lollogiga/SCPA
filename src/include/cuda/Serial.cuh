//
// Created by buniy on 01/04/2025.
//

#ifndef SERIAL_CUH
#define SERIAL_CUH

#include "../constants.h"
#include "../mtxStructs.h"
#include "../../include/result.h"

__global__ void spmv_csr_serial(CSRMatrix *csr, MatVal *vector, ResultVector *result);
__global__ void spmv_hll_serial(HLLMatrix *hll, MatVal *vector, ResultVector *result);
#endif //SERIAL_CUH
