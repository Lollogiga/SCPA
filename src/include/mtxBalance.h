#ifndef MATRIXBALANCE_H
#define MATRIXBALANCE_H

#include "../include/preprocessing.h"

typedef struct {
    MatT start;
    MatT end;
} ThreadDataRange;

ThreadDataRange *matrixBalanceCSR(CSRMatrix *csr, int numThreads);
ThreadDataRange *matrixBalanceHLL(HLLMatrix *hll, int numThreads);
ThreadDataRange *matrixBalanceHLL_sol2(HLLMatrixAligned *hll, int numThreads);

#endif //MATRIXBALANCE_H
