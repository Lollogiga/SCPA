#ifndef MATRIXPRODUCT_H
#define MATRIXPRODUCT_H

#include "../mtxStructs.h"

int computeOpenMP(CSRMatrix *csrMatrix, HLLMatrix *hllMatrix, HLLMatrixAligned *hllMatrixAligned, int num_threads);

#endif //MATRIXPRODUCT_H
