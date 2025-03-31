#include <stdio.h>

#include "../include/mtxPrint.h"
#include "../include/constants.h"

void print_matrix_data(RawMatrix *matrix) {
    if (!matrix) return;

    for (MatT i=0; i<matrix->NZ; i++) {
        printf("A[%d][%d] = %f\n", matrix->I[i], matrix->J[i], matrix->val[i]);
    }
}

void print_matrix_data_verbose(RawMatrix *matrix, bool verbose) {
    if (!matrix) return;

    printf("M = %d, N = %d, NZ = %d\n", matrix->M, matrix-> N, matrix-> NZ);

    const int max_print_nz = (matrix->NZ < MAX_PRINT_COLUMN) ? matrix->NZ : MAX_PRINT_COLUMN;

    printf("I = [");
    for (MatT i = 0; i < max_print_nz; i++) {
        printf("%d, ", matrix->I[i]);
    }
    printf("\b\b]\n");

    printf("J = [");
    for (MatT i = 0; i < max_print_nz; i++) {
        printf("%d, ", matrix->J[i]);
    }
    printf("\b\b]\n");

    printf("val = [");
    for (MatT i = 0; i < max_print_nz; i++) {
        printf("%f, ", matrix->val[i]);
    }
    printf("\b\b]\n");

    if (verbose) {
        print_matrix_data(matrix);
    }
}

void print_csr_matrix(CSRMatrix *csr) {
    if (!csr) return;

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

void print_csr_matrix_verbose(CSRMatrix *csr, bool verbose) {
    if (!csr) return;

    printf("M = %d, N = %d, NZ = %d\n", csr->M, csr-> N, csr-> NZ);

    const int max_print_cols_IRP = (csr->M < MAX_PRINT_ROW) ? csr->M : MAX_PRINT_ROW;
    const int max_print_nz = (csr->NZ < MAX_PRINT_COLUMN) ? csr->NZ : MAX_PRINT_COLUMN;

    printf("IRP = [");
    for (MatT i = 0; i < max_print_cols_IRP + 1; i++) {
        printf("%d, ", csr->IRP[i]);
    }
    printf("\b\b]\n");

    printf("JA = [");
    for (MatT i = 0; i < max_print_nz; i++) {
        printf("%d, ", csr->JA[i]);
    }
    printf("\b\b]\n");

    printf("AS = [");
    for (MatT i = 0; i < max_print_nz; i++) {
        printf("%f, ", csr->AS[i]);
    }
    printf("\b\b]\n");

    if (verbose) {
        print_csr_matrix(csr);
    }
}

void print_ellpack_matrix(ELLPACKMatrix *ell) {
    if (!ell) return;

    for (MatT i = 0; i < ell->M; i++) {
        for (MatT j = 0; j < ell->MAXNZ; j++) {
            printf("JA[%d][%d] = %d\n", i, j, ell->JA[i][j]);
        }
    }

    for (MatT i = 0; i < ell->M; i++) {
        for (MatT j = 0; j < ell->MAXNZ; j++) {
            printf("AS[%d][%d] = %f\n", i, j, ell->AS[i][j]);
        }
    }
}

void print_ellpack_matrix_verbose(ELLPACKMatrix *ell, bool verbose) {
    if (!ell) return;

    printf("M = %d, N = %d, MAXNZ = %d\n", ell->M, ell->N, ell->MAXNZ);

    const int max_print_rows = (ell->M < MAX_PRINT_ROW) ? ell->M : MAX_PRINT_ROW;
    const int max_print_nz = (ell->MAXNZ < MAX_PRINT_COLUMN) ? ell->MAXNZ : MAX_PRINT_COLUMN;

    printf("JA = [\n");
    for (MatT i = 0; i < max_print_rows; i++) {
        printf("  [");
        for (MatT j = 0; j < max_print_nz; j++) {
            printf("%d, ", ell->JA[i][j]);
        }
        printf(i != ell->N-1 ? "\b\b],\n" : "\b\b]\n");
    }
    printf("]\n");

    printf("AS = [\n");
    for (MatT i = 0; i < max_print_rows; i++) {
        printf("  [");
        for (MatT j = 0; j < max_print_nz; j++) {
            printf("%f, ", ell->AS[i][j]);
        }
        printf(i != ell->N-1 ? "\b\b],\n" : "\b\b]\n");
    }
    printf("\b\b]\n");

    if (verbose) {
        print_ellpack_matrix(ell);
    }
}

void print_result_vector(ResultVector *result) {
    if (!result) return;

    const int max_print_cols = (result->len_vector < MAX_PRINT_ROW) ? result->len_vector : MAX_PRINT_ROW;

    printf("[");
    for (int i = 0; i < max_print_cols; i++) {
        if (i > 0) {
            printf(", ");
        }
        printf("%f", result->val[i]);
    }
    printf("]\n");
}