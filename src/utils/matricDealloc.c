//
// Created by buniy on 28/02/2025.
//

#include <stdlib.h>

#include "../include/matricDealloc.h"

// Functions for deallocation:
void free_MatrixData(MatrixData *matrix) {
    if (!matrix) return;

    if (matrix->I) free(matrix->I);
    if (matrix->J) free(matrix->J);
    if (matrix->val) free(matrix->val);

    free(matrix);
}

void free_CSRMatrix(CSRMatrix *csr) {
    if (!csr) return;

    if (csr->IRP) free(csr->IRP);
    if (csr->JA) free(csr->JA);
    if (csr->AS) free(csr->AS);

    free(csr);
}

void free_ELLPACKMatrix(ELLPACKMatrix *ell) {
    if (!ell) return;

    if (ell->AS) {
        for (int i = 0; i < ell->M; i++) {
            free(ell->AS[i]);
        }
        free(ell->AS);
    }

    if (ell->JA) {
        for (int i = 0; i < ell->M; i++) {
            free(ell->JA[i]);
        }
        free(ell->JA);
    }

    free(ell);
}

void free_HLLMatrix(HLLMatrix *hll) {
    if (!hll) return;

    if (!hll->blocks) return;

    for (int i = 0; i < hll->numBlocks; i++)
        if (hll->blocks[i]) free_ELLPACKMatrix(hll->blocks[i]);

    free(hll->blocks);
    free(hll);
}
