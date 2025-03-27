#ifndef SERIALPRODUCT_H
#define SERIALPRODUCT_H

#include "../include/matrixPreProcessing.h"

typedef struct {
    MatT len_vector;
    MatVal *val;
} ResultVector;

ResultVector *csr_serialProduct(CSRMatrix *csr, const MatVal *vector);
ResultVector *hll_serialProduct(HLLMatrix *hll, MatVal *vector);
ResultVector *hllAligned_serialProduct(HLLMatrixAligned *hll, MatVal *vector);

#endif //SERIALPRODUCT_H
