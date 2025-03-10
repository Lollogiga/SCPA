//
// Created by buniy on 10/03/2025.
//

#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "../include/openmpHLL.h"
#include "../include/matrixPreProcessing.h"

void *ellpack_openmpProduct(ELLPACKMatrix *ell, const MatVal *vector, MatVal *result) {
    if (!ell) {
        perror("ellpack_serialProduct: ell is NULL");
        return NULL;
    }

    if (!result) {
        perror("ellpack_serialProduct: malloc");
        return NULL;
    }

    for (MatT i = 0; i < ell->M; i++) {
        for (MatT j = 0; j < ell->MAXNZ; j++) {
            MatT col_index = ell->JA[i][j];

            result[i] += ell->AS[i][j] * vector[col_index];
        }
    }

    return result;
}

ResultVector *hll_openmpProduct_sol1(HLLMatrix *hll, MatVal *vector) {
    if(!hll) {
        perror("hll_serialProduct: hll is NULL");
        return NULL;
    }

    if (!vector) {
        perror("hll_serialProduct: create_vector");
        return NULL;
    }

    //Create result vector
    ResultVector *result = create_result_vector(hll->M);
    if (!result) {
        perror("hll_serialProduct: create_result_vector");
        return NULL;
    }

    volatile bool error_flag = 0;

#pragma omp parallel for shared(error_flag)
    for (MatT i = 0; i < hll->numBlocks; i++) {
        if (error_flag) continue;

        ELLPACKMatrix *block = hll->blocks[i];
        void *res = ellpack_openmpProduct(block, vector, result->val + block->startRow);
        if (!res) {
            perror("hll_serialProduct: ellpack_serialProduct");

#pragma omp atomic write
            error_flag = 1;
        }
    }

    if (error_flag) {
        perror("hll_serialProduct: ellpack_serialProduct");
        free(result);
        return NULL;
    }

    return result;
}
