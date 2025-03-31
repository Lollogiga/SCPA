#include <stdlib.h>

#include "../../include/constants.h"
#include "../../include/mtxStructs.h"
#include "../../include/preprocessing.h"
#include "../../include/openmp/Serial.h"
#include "../../include/createVector.h"

ResultVector *csr_serialProduct(CSRMatrix *csr, const MatVal *vector) {
    if (!csr) {
        perror("csr_serialProduct: csr is NULL");
        return NULL;
    }

    if (!vector) {
        perror("csr_serialProduct: vector is NULL");
        return NULL;
    }

    // Create result vector:
    MatT len_result_vector = csr->M;
    ResultVector *result = create_result_vector(len_result_vector);
    if (!result) {
        perror("csr_serialProduct: create_result_vector");
        return NULL;
    }

    for (MatT i = 0; i < csr->M; i++) {
        for (MatT j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
            MatT vector_index = csr->JA[j];
            result->val[i] += csr->AS[j] * vector[vector_index];
        }
    }

    return result;
}

void *ellpack_serialProduct(ELLPACKMatrix *ell, const MatVal *vector, MatVal *result) {
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

ResultVector *hll_serialProduct(HLLMatrix *hll, MatVal *vector) {
    if (!hll) {
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

    for (MatT i = 0; i < hll->numBlocks; i++) {
        ELLPACKMatrix *block = hll->blocks[i];
        void *res = ellpack_serialProduct(block, vector, result->val + block->startRow);
        if (!res) {
            perror("hll_serialProduct: ellpack_serialProduct");
            free(result);
            return NULL;
        }
    }

    return result;
}

void *ellpack_sol2_serialProduct(ELLPACKMatrixAligned *ell, const MatVal *vector, MatVal *result) {
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
            MatT col_index = ell->JA[i * ell->MAXNZ + j];

            result[i] += ell->AS[i * ell->MAXNZ + j] * vector[col_index];
        }
    }

    return result;
}

ResultVector *hllAligned_serialProduct(HLLMatrixAligned *hll, MatVal *vector) {
    if (!hll) {
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

    for (MatT i = 0; i < hll->numBlocks; i++) {
        ELLPACKMatrixAligned *block = hll->blocks[i];
        void *res = ellpack_sol2_serialProduct(block, vector, result->val + block->startRow);
        if (!res) {
            perror("hll_serialProduct: ellpack_serialProduct");
            free(result);
            return NULL;
        }
    }

    return result;
}
