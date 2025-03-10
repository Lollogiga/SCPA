//
// Created by buniy on 10/03/2025.
//

#ifndef OPENMPHLL_H
#define OPENMPHLL_H

#include "../include/createVectorUtil.h"

ResultVector *hll_openmpProduct_sol1(HLLMatrix *hll, MatVal *vector);
ResultVector *hll_openmpProduct_sol2(HLLMatrix *hll, MatVal *vector);
ResultVector *hll_openmpProduct_sol3(HLLMatrix *hll, MatVal *vector, int num_threads, ThreadDataRange* tdr);

#endif //OPENMPHLL_H
