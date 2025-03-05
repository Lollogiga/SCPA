//
// Created by buniy on 28/02/2025.
//

#ifndef MATRICDEALLOC_H
#define MATRICDEALLOC_H

#include "./matrixPreProcessing.h"
#include "./serialProduct.h"

// Function for de-allocation:
void free_MatrixData(MatrixData *matrix);
void free_CSRMatrix(CSRMatrix *csr);
void free_ELLPACKMatrix(ELLPACKMatrix *ell);
void free_HLLMatrix(HLLMatrix *hll);
void free_ResultVector(ResultVector *rv);
#endif //MATRICDEALLOC_H
