#ifndef MATRICDEALLOC_H
#define MATRICDEALLOC_H

#include "./mtxStructs.h"
#include "./result.h"

// Function for de-allocation:
void free_MatrixData(RawMatrix *matrix);
void free_CSRMatrix(CSRMatrix *csr);
void free_ELLPACKMatrix(ELLPACKMatrix *ell);
void free_HLLMatrix(HLLMatrix *hll);
void free_HLLMatrixAligned(HLLMatrixAligned *hll);
void free_ResultVector(ResultVector *rv);
#endif //MATRICDEALLOC_H
