#ifndef OPENMPHLL_H
#define OPENMPHLL_H

#include "../createVector.h"

ResultVector *hll_openmpProduct_sol1(HLLMatrix *hll, MatVal *vector, int num_threads);

ResultVector *hll_openmpProduct_sol2(HLLMatrix *hll, MatVal *vector, int num_threads);

ResultVector *hll_openmpProduct_sol3(HLLMatrix *hll, MatVal *vector, int num_threads, ThreadDataRange *tdr);

ResultVector *hllAligned_openmpProduct(HLLMatrixAligned *hll, MatVal *vector, int num_threads);

ResultVector *hllAligned_openmpProduct_sol2(HLLMatrixAligned *hll, MatVal *vector, int num_threads);

ResultVector *hllAligned_openmpProduct_sol3(HLLMatrixAligned *hll, MatVal *vector, int num_thread,
                                            ThreadDataRange *tdr);

#endif //OPENMPHLL_H
