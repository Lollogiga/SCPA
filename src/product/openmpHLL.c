#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "../include/matrixBalance.h"
#include "../include/matrixPreProcessing.h"
#include "../include/openmpHLL.h"


void *ellpack_openmpProduct(ELLPACKMatrix *ell, const MatVal *vector, MatVal *result) {
    if (!ell) {
        perror("ellpack_openmpProduct: ell is NULL");
        return NULL;
    }

    if (!result) {
        perror("ellpack_openmpProduct: malloc");
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
        perror("hll_openmpProduct_sol1: hll is NULL");
        return NULL;
    }

    if (!vector) {
        perror("hll_openmpProduct_sol1: create_vector");
        return NULL;
    }

    //Create result vector
    ResultVector *result = create_result_vector(hll->M);
    if (!result) {
        perror("hll_openmpProduct_sol1: create_result_vector");
        return NULL;
    }

    volatile bool error_flag = 0;

#pragma omp parallel for shared(error_flag)
    for (MatT i = 0; i < hll->numBlocks; i++) {
        if (error_flag) continue;

        ELLPACKMatrix *block = hll->blocks[i];
        void *res = ellpack_openmpProduct(block, vector, result->val + block->startRow);
        if (!res) {
            perror("hll_openmpProduct_sol1: ellpack_serialProduct");

#pragma omp atomic write
            error_flag = 1;
        }
    }

    if (error_flag) {
        perror("hll_openmpProduct_sol1: ellpack_serialProduct");
        free(result);
        return NULL;
    }

    return result;
}

ResultVector *hll_openmpProduct_sol2(HLLMatrix *hll, MatVal *vector) {
    if(!hll) {
        perror("hll_openmpProduct_sol2: hll is NULL");
        return NULL;
    }

    if (!vector) {
        perror("hll_openmpProduct_sol2: create_vector");
        return NULL;
    }

    //Create result vector
    ResultVector *result = create_result_vector(hll->M);
    if (!result) {
        perror("hll_openmpProduct_sol2: create_result_vector");
        return NULL;
    }

    volatile bool error_flag = 0;

#pragma omp parallel for shared(error_flag) schedule(dynamic)
    for (MatT i = 0; i < hll->numBlocks; i++) {
        if (error_flag) continue;

        ELLPACKMatrix *block = hll->blocks[i];
        void *res = ellpack_openmpProduct(block, vector, result->val + block->startRow);
        if (!res) {
            perror("hll_openmpProduct_sol2: ellpack_serialProduct");

#pragma omp atomic write
            error_flag = 1;
        }
    }

    if (error_flag) {
        perror("hll_openmpProduct_sol2: ellpack_serialProduct");
        free(result);
        return NULL;
    }

    return result;
}

ResultVector *hll_openmpProduct_sol3(HLLMatrix *hll, MatVal *vector, int num_threads, ThreadDataRange* tdr) {
    if(!hll) {
        perror("hll_openmpProduct_sol3: hll is NULL");
        return NULL;
    }

    if (!vector) {
        perror("hll_openmpProduct_sol3: create_vector");
        return NULL;
    }

    if (!tdr) {
        perror("hll_openmpProduct_sol3: tdr is NULL");
        return NULL;
    }

    //Create result vector
    ResultVector *result = create_result_vector(hll->M);
    if (!result) {
        perror("hll_openmpProduct_sol3: create_result_vector");
        return NULL;
    }

    volatile bool error_flag = 0;

    omp_set_num_threads(num_threads);

#pragma omp parallel for shared(error_flag)
    for (int t = 0; t < num_threads; t++) {
        const MatT start = tdr[t].start;
        const MatT end = tdr[t].end;

        for (MatT i = start; i < end; i++) {
            if (error_flag) continue;

            ELLPACKMatrix *block = hll->blocks[i];
            void *res = ellpack_openmpProduct(block, vector, result->val + block->startRow);
            if (!res) {
                perror("hll_openmpProduct_sol3: ellpack_serialProduct");

#pragma omp atomic write
                error_flag = 1;
            }
        }
    }

    if (error_flag) {
        perror("hll_openmpProduct_sol3: ellpack_serialProduct");
        free(result);
        return NULL;
    }

    return result;
}

