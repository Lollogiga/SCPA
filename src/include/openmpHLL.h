//
// Created by buniy on 10/03/2025.
//

#ifndef OPENMPHLL_H
#define OPENMPHLL_H

#include "../include/createVectorUtil.h"

ResultVector *hll_openmpProduct_sol1(HLLMatrix *hll, MatVal *vector);
ResultVector *hll_openmpProduct_sol2(HLLMatrix *hll, MatVal *vector);
ResultVector *hll_openmpProduct_sol3(HLLMatrix *hll, MatVal *vector, int num_threads, ThreadDataRange* tdr);
ResultVector *hll_sol2_openmpProduct(HLLMatrix_sol2 *hll, MatVal *vector);
ResultVector *hll_sol2_openmpProduct_sol2(HLLMatrix_sol2 *hll, MatVal *vector);
ResultVector *hll_sol2_openmpProduct_sol3(HLLMatrix_sol2 *hll, MatVal *vector,int num_thread, ThreadDataRange* tdr);

#endif //OPENMPHLL_H
