#ifndef MATRIXPRODUCT_H
#define MATRIXPRODUCT_H

#include "../mtxStructs.h"
#include "../performance.h"

int computeOpenMP(CSRMatrix *csrMatrix, HLLMatrix *hllMatrix, HLLMatrixAligned *hllMatrixAligned, int num_threads, PerformanceResult *performance);

#endif //MATRIXPRODUCT_H
