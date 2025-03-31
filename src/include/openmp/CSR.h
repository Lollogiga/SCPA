#ifndef OPENMPPRODUCT_H
#define OPENMPPRODUCT_H

#include "../mtxBalance.h"
#include "../mtxStructs.h"
#include "../result.h"

ResultVector *csr_openmpProduct_sol1(CSRMatrix *csr, MatVal *vector, int num_threads);

ResultVector *csr_openmpProduct_sol2(CSRMatrix *csr, MatVal *vector, int num_threads);

ResultVector *csr_openmpProduct_sol3(CSRMatrix *csr, MatVal *vector, int num_threads);

ResultVector *csr_openmpProduct_sol4(CSRMatrix *csr, MatVal *vector, int num_threads);

ResultVector *csr_openmpProduct_sol5(CSRMatrix *csr, MatVal *vector, int num_threads, ThreadDataRange *tdr);

#endif //OPENMPPRODUCT_H
