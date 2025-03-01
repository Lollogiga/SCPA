#include <stdlib.h>

#include "../include/constants.h"
#include "../include/matrixPreProcessing.h"
#include "../include/serialProduct.h"

MatVal *csr_serialProduct(CSRMatrix *csr) {
    if (!csr) {
        perror("csr_serialProduct: csr is NULL");
        return NULL;
    }

    MatT vector_col = csr->N;

    //Create vector for matrix vector multiply:
    MatVal *vector = calloc(vector_col , sizeof(MatVal));
    if (!vector) {
        perror("csr_serialProduct: vector malloc failed");
        return NULL;
    }
    for (int i = 0; i < vector_col; i++) {
        vector[i] = 1;
    }

    //Create result vector:
    MatVal *result = calloc(csr->M , sizeof(MatVal));
    if (!result) {
        perror("csr_serialProduct: result malloc failed");
        free(vector);
        return NULL;
    }


    for (MatT i = 0; i <= csr->M; i++) {
        for (MatT j = csr->IRP[i]; j < csr->IRP[i+1]; j++) {
            MatT vector_index = csr->JA[j];
            result[i] += csr->AS[j] * vector[vector_index];
        }
    }

    free(vector);
    return result;
}