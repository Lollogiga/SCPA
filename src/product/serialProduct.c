#include <stdlib.h>

#include "../include/constants.h"
#include "../include/matrixPreProcessing.h"
#include "../include/serialProduct.h"
#include "../include/utilsProduct.h"


ResultVector *csr_serialProduct(CSRMatrix *csr) {
    if (!csr) {
        perror("csr_serialProduct: csr is NULL");
        return NULL;
    }

    // Create vector for matrix vector multiply:
    MatT len_vector = csr->N;
    MatVal *vector = create_vector(len_vector);
    if (!vector) {
        perror("csr_serialProduct: create_vector");
        return NULL;
    }

    //Create result vector:
    MatT len_result_vector = csr->M;
    ResultVector *result = create_result_vector(len_result_vector);
    if (!result) {
        perror("csr_serialProduct: create_result_vector");
        free(vector);
        return NULL;
    }

    for (MatT i = 0; i <= csr->M; i++) {
        for (MatT j = csr->IRP[i]; j < csr->IRP[i+1]; j++) {
            MatT vector_index = csr->JA[j];
            result->val[i] += csr->AS[j] * vector[vector_index];
        }
    }

    free(vector);
    return result;
}



ResultVector *hll_serialProduct(HLLMatrix *hll) {
    if(!hll) {
        perror("hll_serialProduct: hll is NULL");
        return NULL;
    }

    //Create vector
    MatVal *vector = create_vector(hll->N);
    if (!vector) {
        perror("hll_serialProduct: create_vector");
        return NULL;
    }

    //Create result vector
    ResultVector *result = create_result_vector(hll->M);
    if (!result) {
        perror("hll_serialProduct: create_result_vector");
        free(vector);
        return NULL;
    }

    for (MatT i = 0; i < hll->numBlocks; i++) {

    }

    free(vector);
}