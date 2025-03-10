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
ResultVector *csr_openmpProduct_sol5(CSRMatrix *csr, MatVal *vector, int num_threads, ThreadDataRange* tdr);

#endif //OPENMPPRODUCT_H
