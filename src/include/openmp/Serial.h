#ifndef SERIALPRODUCT_H
#define SERIALPRODUCT_H

#include "../mtxStructs.h"
#include "../result.h"

#ifdef __cplusplus
extern "C" {
#endif

ResultVector *csr_serialProduct(CSRMatrix *csr, const MatVal *vector);
ResultVector *hll_serialProduct(HLLMatrix *hll, MatVal *vector);
ResultVector *hllAligned_serialProduct(HLLMatrixAligned *hll, MatVal *vector);

#ifdef __cplusplus
}
#endif

#endif //SERIALPRODUCT_H
