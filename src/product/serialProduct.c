#include <stdlib.h>
#include <string.h>

#include "../include/constants.h"
#include "../include/matrixPreProcessing.h"
#include "../include/serialProduct.h"
#include "../include/createVectorUtil.h"

ResultVector *csr_serialProduct(CSRMatrix *csr, MatVal *vector) {
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
        for (MatT j = csr->IRP[i]; j < csr->IRP[i+1]; j++) {
            MatT vector_index = csr->JA[j];
            result->val[i] += csr->AS[j] * vector[vector_index];
        }
    }

    return result;
}

void *ellpack_serialProduct(ELLPACKMatrix *ell, const MatVal *vector, MatVal *partial_result) {
    if (!ell) {
        perror("ellpack_serialProduct: ell is NULL");
        return NULL;
    }

    if (!partial_result) {
        perror("ellpack_serialProduct: malloc");
        return NULL;
    }

    memset(partial_result, 0, ell->M * sizeof(MatVal));
    for (MatT i = 0; i < ell->M; i++) {
        for (MatT j = 0; j < ell->MAXNZ; j++) {
            MatT col_index = ell->JA[i][j];
            //Check correctness:
            partial_result[i] += ell->AS[i][j] * vector[col_index];
        }
    }
    return partial_result;
}

ResultVector *hll_serialProduct(HLLMatrix *hll, MatVal *vector) {
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

    MatT offset = 0;
    MatVal *partial_result = malloc(HACK_SIZE * sizeof(MatVal));
    if (!partial_result) {
        perror("hll_serialProduct: malloc");
        free(result);
        return NULL;
    }
    for (MatT i = 0; i < hll->numBlocks; i++) {
        ELLPACKMatrix *block = hll->blocks[i];
        void *res = ellpack_serialProduct(block, vector, partial_result);
        if (!res) {
            perror("hll_serialProduct: ellpack_serialProduct");
            free(result);
            free(partial_result);
            return NULL;
        }
        //Sum each blocks in the final vector:
        for (MatT j=0; j < block->M; j++) {
            result->val[offset+j] += partial_result[j];
        }
        offset += block->M;
    }
    free(partial_result);
    return result;
}

