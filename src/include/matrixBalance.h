//
// Created by buniy on 10/03/2025.
//

#ifndef MATRIXBALANCE_H
#define MATRIXBALANCE_H

#include "../include/matrixPreProcessing.h"

typedef struct {
    MatT start;
    MatT end;
} ThreadDataRange;

ThreadDataRange *matrixBalanceCSR(CSRMatrix *csr, int numThreads);
ThreadDataRange *matrixBalanceHLL(HLLMatrix *hll, int numThreads);
#endif //MATRIXBALANCE_H
