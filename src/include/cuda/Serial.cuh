//
// Created by buniy on 01/04/2025.
//

#ifndef SERIAL_CUH
#define SERIAL_CUH

#include "../constants.h"
#include "../mtxStructs.h"

__global__ void spmv_csr_serial(CSRMatrix *csr, MatVal *vector, MatVal *result);

#endif //SERIAL_CUH
