#ifndef SERIALPRODUCT_H
#define SERIALPRODUCT_H

#include "../include/matrixPreProcessing.h"

typedef struct {
    MatT len_vector;
    MatVal *val;
}ResultVector;

ResultVector *csr_serialProduct(CSRMatrix *csr);
#endif //SERIALPRODUCT_H
