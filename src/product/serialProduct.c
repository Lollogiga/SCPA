#include <stdlib.h>

#include "../include/constants.h"
#include "../include/matrixPreProcessing.h"
#include "../include/serialProduct.h"

ResultVector *csr_serialProduct(CSRMatrix *csr) {
    if (!csr) {
        perror("csr_serialProduct: csr is NULL");
        return NULL;
    }

    MatT len_vector = csr->N;

    // Create vector for matrix vector multiply:
    MatVal *vector = calloc(len_vector , sizeof(MatVal));
    if (!vector) {
        perror("csr_serialProduct: vector malloc failed");
        return NULL;
    }

    // Fill vector
    for (int i = 0; i < len_vector; i++) {
        vector[i] = 1;
    }

    //Create result vector:
    ResultVector *result = malloc(sizeof(ResultVector));
    if (!result) {
        perror("csr_serialProduct: result malloc failed");
        free(vector);
        return NULL;
    }

    //Define dim of result vectors:
    result->len_vector = csr->M;
    result->val = calloc(result->len_vector , sizeof(MatVal));
    if (!result->val) {
        perror("csr_serialProduct: result val malloc failed");
        free(vector);
        free(result);
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