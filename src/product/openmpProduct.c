//
// Created by buniy on 07/03/2025.
//

#include <omp.h>

#include "../include/constants.h"
#include "../include/matrixPreProcessing.h"
#include "../include/openmpProduct.h"
#include <string.h>
#include <stdlib.h>

ResultVector *csr_openmpProduct_sol1(CSRMatrix *csr, MatVal *vector) {
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

    // omp_set_num_threads(1);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        int row_per_thread = csr->M / num_threads;
        if (num_threads > csr->M && thread_id == 0) row_per_thread = 0;

        int exit_condition = (thread_id != num_threads-1) ? (thread_id + 1) * row_per_thread : csr->M;

        for (MatT i = thread_id * row_per_thread; i < exit_condition; i++) {
            MatVal temp = 0;
            for (MatT j = csr->IRP[i]; j < csr->IRP[i+1]; j++) {
                MatT vector_index = csr->JA[j];
                temp += csr->AS[j] * vector[vector_index];
            }

            result->val[i] = temp;
        }
    }

    return result;
}

ResultVector *csr_openmpProduct_sol2(CSRMatrix *csr, MatVal *vector) {
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

    // omp_set_num_threads(1);

#pragma omp parallel for
    for (MatT i = 0; i < csr->M; i++) {
        MatVal temp = 0;
        for (MatT j = csr->IRP[i]; j < csr->IRP[i+1]; j++) {
            MatT vector_index = csr->JA[j];
            temp += csr->AS[j] * vector[vector_index];
        }

        result->val[i] = temp;
    }

    return result;
}

ResultVector *csr_openmpProduct_sol3(CSRMatrix *csr, MatVal *vector) {
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

    // omp_set_num_threads(1);

#pragma omp parallel for
    for (MatT i = 0; i < csr->M; i++) {
        MatVal sum = 0.0;
#pragma omp simd reduction(+:sum)
        for (MatT j = csr->IRP[i]; j < csr->IRP[i+1]; j++) {
            sum += csr->AS[j] * vector[csr->JA[j]];
        }

        result->val[i] = sum;
    }

    return result;
}

ResultVector *csr_openmpProduct_sol4(CSRMatrix *csr, MatVal *vector) {
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

    // omp_set_num_threads(1);

#pragma omp parallel for schedule(auto)
    for (MatT i = 0; i < csr->M; i++) {
        MatVal sum = 0.0;
#pragma omp simd reduction(+:sum)
        for (MatT j = csr->IRP[i]; j < csr->IRP[i+1]; j++) {
            sum += csr->AS[j] * vector[csr->JA[j]];
        }

        result->val[i] = sum;
    }

    return result;
}

void *ellpack_openmpProduct(ELLPACKMatrix *ell, const MatVal *vector, MatVal *partial_result) {
    if (!ell) {
        perror("ellpack_serialProduct: ell is NULL");
        return NULL;
    }

    if (!partial_result) {
        perror("ellpack_serialProduct: malloc");
        return NULL;
    }

    memset(partial_result, 0, ell->M * sizeof(MatVal));
#pragma omp parallel for
    for (MatT i = 0; i < ell->M; i++) {
        MatVal sum = 0;
        for (MatT j = 0; j < ell->MAXNZ; j++) {
            MatT col_index = ell->JA[i][j];
            //Check correctness:
            sum += ell->AS[i][j] * vector[col_index];
        }
        partial_result[i] = sum;
    }
    return partial_result;
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
        return NULL;MatVal sum = 0;
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
        void *res = ellpack_openmpProduct(block, vector, partial_result);
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

/*
ResultVector *hll_openmpProduct_sol2(HLLMatrix *hll, MatVal *vector) {
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
        return NULL;MatVal sum = 0;
    }
    int error_flag = 0;
#pragma omp parallel for schedule(dynamic) shared(error_flag)
    for (MatT i = 0; i < hll->numBlocks; i++) {
        // Leggo error_flag in maniera atomica
#pragma omp atomic read
        if (error_flag) continue;

        MatVal *partial_result = malloc(HACK_SIZE * sizeof(MatVal));
        if (!partial_result) {
            perror("malloc");
#pragma omp atomic write
            error_flag = 1;
            continue;
        }
        ELLPACKMatrix *block = hll->blocks[i];
        void *res = ellpack_openmpProduct(block, vector, partial_result);
        if (!res) {
            perror("ellpack_openmpProduct_sol1");
            free(partial_result);
#pragma omp atomic write
            error_flag = 1;
            continue;
        }
        MatT offset = i * block->M;
        for (MatT j = 0; j < block->M; j++) {
            result->val[offset + j] += partial_result[j];
        }
        free(partial_result);
    }

    if (error_flag) return NULL;
    return result;

}
*/

