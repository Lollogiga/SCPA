//
// Created by buniy on 07/03/2025.
//

#ifndef OPENMPPRODUCT_H
#define OPENMPPRODUCT_H

#include "../include/createVectorUtil.h"

ResultVector *csr_openmpProduct_sol1(CSRMatrix *csr, MatVal *vector);
ResultVector *csr_openmpProduct_sol2(CSRMatrix *csr, MatVal *vector);
ResultVector *csr_openmpProduct_sol3(CSRMatrix *csr, MatVal *vector);
ResultVector *csr_openmpProduct_sol4(CSRMatrix *csr, MatVal *vector);

ResultVector *hll_openmpProduct_sol1(HLLMatrix *hll, MatVal *vector);
ResultVector *hll_openmpProduct_sol2(HLLMatrix *hll, MatVal *vector);
#endif //OPENMPPRODUCT_H
