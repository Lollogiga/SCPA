#ifndef COMPUTECUDA_CUH
#define COMPUTECUDA_CUH

#include "../mtxStructs.h"

#ifdef __cplusplus
extern "C" {
#endif

    int computeCUDA(CSRMatrix *csrMatrix, HLLMatrix *hllMatrix, HLLMatrixAligned *hllMatrixAligned, int num_threads);

#ifdef __cplusplus
}
#endif

#endif //COMPUTECUDA_CUH
