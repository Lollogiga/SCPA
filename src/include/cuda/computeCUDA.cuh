#ifndef COMPUTECUDA_CUH
#define COMPUTECUDA_CUH

#include "../mtxStructs.h"
#include "../performance.h"

#ifdef __cplusplus
extern "C" {
#endif

    int computeCUDA(CSRMatrix *csrMatrix, HLLMatrix *hllMatrix, HLLMatrixAligned *hllMatrixAligned, int blockSize, int warpSize, PerformanceResult *performance);

#ifdef __cplusplus
}
#endif

#endif //COMPUTECUDA_CUH
