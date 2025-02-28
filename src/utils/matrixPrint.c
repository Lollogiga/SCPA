//
// Created by buniy on 28/02/2025.
//

#include "../include/matrixPrint.h"

//Function for debugging:
void print_matrix_data(MatrixData *matrix) {
    for (MatT i=0; i<matrix->NZ; i++) {
        printf("A[%d][%d] = %f\n", matrix->I[i], matrix->J[i], matrix->val[i]);
    }
}

void print_csr_matrix(CSRMatrix *csr) {
    for (MatT i = 0; i <= csr->M; i++) {
        printf("IRP[%d] = %d\n", i, csr->IRP[i]);
    }

    for (MatT i = 0; i < csr->NZ; i++) {
        printf("JA[%d] = %d\n", i, csr->JA[i]);
    }

    for (MatT i = 0; i < csr->NZ; i++) {
        printf("AS[%d] = %f\n", i, csr->AS[i]);
    }
}

void print_ellpack_matrix(ELLPACKMatrix *ell) {
    for (MatT i = 0; i < ell->N; i++) {
        for (MatT j = 0; j < ell->MAXNZ; j++) {
            printf("JA[%d][%d] = %d\n", i, j, ell->JA[i][j]);
        }
    }

    for (MatT i = 0; i < ell->N; i++) {
        for (MatT j = 0; j < ell->MAXNZ; j++) {
            printf("AS[%d][%d] = %f\n", i, j, ell->AS[i][j]);
        }
    }
}

