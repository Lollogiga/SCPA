//
// Created by buniy on 07/03/2025.
//

#include <omp.h>

#include "../../include/constants.h"
#include "../../include/matrixBalance.h"
#include "../../include/matrixPreProcessing.h"
#include "../../include/openmpCSR.h"

ResultVector *csr_openmpProduct_sol1(CSRMatrix *csr, MatVal *vector, int num_threads) {
    if (!csr) {
        perror("csr_openmpProduct_sol1: csr is NULL");
        return NULL;
    }

    if (!vector) {
        perror("csr_openmpProduct_sol1: vector is NULL");
        return NULL;
    }

    // Create result vector:
    MatT len_result_vector = csr->M;
    ResultVector *result = create_result_vector(len_result_vector);
    if (!result) {
        perror("csr_openmpProduct_sol1: create_result_vector");
        return NULL;
    }

    omp_set_num_threads(num_threads);

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

ResultVector *csr_openmpProduct_sol2(CSRMatrix *csr, MatVal *vector, int num_threads) {
    if (!csr) {
        perror("csr_openmpProduct_sol2: csr is NULL");
        return NULL;
    }

    if (!vector) {
        perror("csr_openmpProduct_sol2: vector is NULL");
        return NULL;
    }

    // Create result vector:
    MatT len_result_vector = csr->M;
    ResultVector *result = create_result_vector(len_result_vector);
    if (!result) {
        perror("csr_openmpProduct_sol2: create_result_vector");
        return NULL;
    }

    omp_set_num_threads(num_threads);

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

ResultVector *csr_openmpProduct_sol3(CSRMatrix *csr, MatVal *vector, int num_threads) {
    if (!csr) {
        perror("csr_openmpProduct_sol3: csr is NULL");
        return NULL;
    }

    if (!vector) {
        perror("csr_openmpProduct_sol3: vector is NULL");
        return NULL;
    }

    // Create result vector:
    MatT len_result_vector = csr->M;
    ResultVector *result = create_result_vector(len_result_vector);
    if (!result) {
        perror("csr_openmpProduct_sol3: create_result_vector");
        return NULL;
    }

    omp_set_num_threads(num_threads);

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

ResultVector *csr_openmpProduct_sol4(CSRMatrix *csr, MatVal *vector, int num_threads) {
    if (!csr) {
        perror("csr_openmpProduct_sol4: csr is NULL");
        return NULL;
    }

    if (!vector) {
        perror("csr_openmpProduct_sol4: vector is NULL");
        return NULL;
    }

    // Create result vector:
    MatT len_result_vector = csr->M;
    ResultVector *result = create_result_vector(len_result_vector);
    if (!result) {
        perror("csr_openmpProduct_sol4: create_result_vector");
        return NULL;
    }

    omp_set_num_threads(num_threads);

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

ResultVector *csr_openmpProduct_sol5(CSRMatrix *csr, MatVal *vector, int num_threads, ThreadDataRange* tdr) {
    if (!csr) {
        perror("csr_openmpProduct_sol5: csr is NULL");
        return NULL;
    }

    if (!vector) {
        perror("csr_openmpProduct_sol5: vector is NULL");
        return NULL;
    }

    if (!tdr) {
        perror("csr_openmpProduct_sol5: tdr is NULL");
        return NULL;
    }

    // Create result vector:
    MatT len_result_vector = csr->M;
    ResultVector *result = create_result_vector(len_result_vector);
    if (!result) {
        perror("csr_openmpProduct_sol5: create_result_vector");
        return NULL;
    }

    omp_set_num_threads(num_threads);

#pragma omp parallel for
    for (int t = 0; t < num_threads; t++) {
        const MatT start = tdr[t].start;
        const MatT end = tdr[t].end;

        for (MatT i = start; i < end; i++) {
            MatVal sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (MatT j = csr->IRP[i]; j < csr->IRP[i+1]; j++) {
                sum += csr->AS[j] * vector[csr->JA[j]];
            }

            result->val[i] = sum;
        }
    }

    return result;
}

