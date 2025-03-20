#include "../include/createVectorUtil.h"
#include <stdlib.h>

MatVal *create_vector(MatT len_vector) {
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
    return vector;
}

ResultVector *create_result_vector(MatT len_vector) {
    ResultVector *result = malloc(sizeof(ResultVector));
    if (!result) {
        perror("csr_serialProduct: result malloc failed");
        return NULL;
    }

    //Define dim of result vectors:
    result->len_vector = len_vector;
    result->val = calloc(result->len_vector, sizeof(MatVal));
    if (!result->val) {
        perror("csr_serialProduct: result val malloc failed");
        free(result);
        return NULL;
    }

    return result;
}