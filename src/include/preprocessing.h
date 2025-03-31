#ifndef MATRIXPREPROCESSING_H
#define MATRIXPREPROCESSING_H

#include <stdio.h>

#include "./mtxStructs.h"

RawMatrix *read_matrix(FILE *f);
CSRMatrix *convert_to_CSR(RawMatrix *matrix);
ELLPACKMatrix *convert_to_ELLPACK(CSRMatrix *matrix);
ELLPACKMatrix *convert_to_ELLPACK_parametrized(CSRMatrix *csr, int iStart, int iEnd);
HLLMatrix *convert_to_HLL(CSRMatrix *matrix, int hackSize);
ELLPACKMatrixAligned *convert_to_ELLPACK_aligned(CSRMatrix *csr, int iStart, int iEnd);
HLLMatrixAligned *convert_to_HLL_aligned(CSRMatrix *matrix, int hackSize);

#endif //MATRIXPREPROCESSING_H
