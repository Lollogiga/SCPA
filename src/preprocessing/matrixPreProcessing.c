#include <stdio.h>
#include <stdlib.h>

#include "../include/constants.h"
#include "../include/matrixPreProcessing.h"
#include "../include/mmio.h"

MatrixData *read_matrix(FILE *f) {

    MatrixData *data = malloc(sizeof(MatrixData));
    MM_typecode matcode;

    // Read and check the presence of the banner in the Matrix Market file
    if (mm_read_banner(f, &matcode) != 0) {
        perror("Could not process Matrix Market banner.\n");
        free(data);
        return NULL;
    }

    /*
    * This block of code allows filtering matrix types if the application
    * only supports a subset of Matrix Market data types.
    */

    // Check if the matrix is complex, a matrix type (not a vector or something else), and sparse
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) ) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        free(data);
        return NULL;
    }

    // Find out the size of the sparse matrix (number of rows, columns, and non-zero entries)
    if (mm_read_mtx_crd_size(f, &data->M, &data->N, &data->NZ) != 0) {
        perror("Error reading matrix data\n");
        free(data);
        return NULL;
    }

    //Check if matrix is symmetric:
    const char sym = mm_is_symmetric(matcode);
    printf("Matrix is symmetric?: %d\n", sym);
    const MatT nz_alloc = sym ? 2 * data->NZ : data->NZ; //If matrix is sym, the value of NZ is at most double

    // Dynamically allocate memory for the sparse matrix's row indices, column indices, and values
    data->I = (MatT *) malloc(nz_alloc * sizeof(MatT));
    data->J = (MatT *) malloc(nz_alloc * sizeof(MatT));
    data->val = (MatVal *) malloc(nz_alloc * sizeof(MatVal));

    if (!data->I || !data->J || !data->val) {
        perror("Memory allocation error");
        free(data->I);
        free(data->J);
        free(data->val);
        free(data);

        return NULL;
    }

    /*
     * NOTE: When reading double values, ANSI C requires the use of the "l" specifier,
     * as in "%lg", "%lf", or "%le". Otherwise, errors can occur. This is part of ANSI C
     * (X3.159-1989, Section 4.9.6.2, pages 136 lines 13-15).
     *
     * fscanf(f, "%d %d %lg\n", &data.I[i], &data.J[i], &data.val[i]);
     */

    int index = 0;
    for (int i = 0; i < data->NZ; i++) {
        /* Retrieve row, col and val from file line */
        char line[MATRIX_FILE_MAX_ROW_LENGTH];  // assuming each line is small enough for this buffer
        if (fgets(line, sizeof(line), f)) {
            char *endptr;

            MatT row = strtol(line, &endptr, 10);  // Convert first integer
            if (*endptr != ' ' && *endptr != '\t') {
                return NULL;
            }

            MatT col = strtol(endptr, &endptr, 10);  // Convert second integer
            if (*endptr != ' ' && *endptr != '\t') {
                return NULL;
            }

            const MatVal val = (MatVal) strtod(endptr, &endptr);  // Convert non-zero value
            if (*endptr != '\n' && *endptr != '\0') {
                return NULL;
            }

            /* Adjust from 1-based to 0-based */
            row--;
            col--;

            data->I[index] = row;
            data->J[index] = col;
            data->val[index] = val;

            index++;

            //If the matrix is symmetric, and we aren't on diagonal:
            if (sym && row != col) {
                data->I[index] = col;
                data->J[index] = row;
                data->val[index] = val;
                index++;
            }
        }
    }

    data->NZ = index; //Update the effective number of NZ

    if (f !=stdin) fclose(f);

    return data;

    /*

    /************************/
    /* now write out matrix */
    /************************/

    /*mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
    for (i=0; i<nz; i++)
        fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);

    return 0;*/

}

CSRMatrix *convert_to_CSR(MatrixData *rawMatrixData) {
    CSRMatrix *csrMatrix = (CSRMatrix *) malloc(sizeof(CSRMatrix)) ;
    csrMatrix->M = rawMatrixData->M;
    csrMatrix->N = rawMatrixData->N;
    csrMatrix->NZ = rawMatrixData->NZ;

    csrMatrix->IRP = (MatT *) calloc(csrMatrix->M + 1, sizeof(MatT));
    csrMatrix->JA  = (MatT *) malloc(csrMatrix->NZ * sizeof(MatT));
    csrMatrix->AS  = (MatVal *) malloc(csrMatrix->NZ * sizeof(MatVal));

    if (!csrMatrix->IRP || !csrMatrix->JA || !csrMatrix->AS) {
        perror("Memory allocation error");
        free(csrMatrix->IRP);
        free(csrMatrix->JA);
        free(csrMatrix->AS);
        free(csrMatrix);

        return NULL;
    }

    //Count NZ for each row:
    for (int i = 0; i < rawMatrixData->NZ; i++) {
        csrMatrix->IRP[rawMatrixData->I[i] + 1]++;
    }

    //Calcutate IRP: A.IRP[i] = A.IRP[i-1] + #NZ_at_i-1:
    for (int i = 1; i <= csrMatrix->M; i++) {
        csrMatrix->IRP[i] += csrMatrix->IRP[i-1];
    }

    // Inserimento degli elementi in JA e AS
    int *row_offset = calloc(csrMatrix->M, sizeof(int));
    for (int i = 0; i < rawMatrixData->NZ; i++) {
        int row = rawMatrixData->I[i];
        int pos = csrMatrix->IRP[row] + row_offset[row];
        csrMatrix->JA[pos] = rawMatrixData->J[i];
        csrMatrix->AS[pos] = rawMatrixData->val[i];
        row_offset[row]++;
    }
    free(row_offset);

    return csrMatrix;
}

/*CSRMatrix *extractCSRBlock(CSRMatrix *csr, MatT start, MatT end) {

    CSRMatrix *csrBlock = malloc(sizeof(CSRMatrix));
    if (!csrBlock) {
        perror("Memory allocation error");
        return NULL;
    }

    csrBlock->M = end - start;
    csrBlock->N = csr->N;

    // Il numero di elementi non zero per il nuovo blocco
    csrBlock->NZ = csr->IRP[end] - csr->IRP[start];

    // Allocazione memoria
    csrBlock->IRP = (MatT *)malloc((csrBlock->M + 1) * sizeof(MatT));
    csrBlock->JA = (MatT *)malloc(csrBlock->NZ * sizeof(MatT));
    csrBlock->AS = (MatVal *)malloc(csrBlock->NZ * sizeof(MatVal));

    // Copia il vettore IRP, traslando gli indici
    MatT baseIndex = csr->IRP[start];
    for (MatT i = 0; i <= csrBlock->M; i++) {
        csrBlock->IRP[i] = csr->IRP[start + i] - baseIndex;
    }

    // Copia JA e AS
    for (MatT i = 0; i < csrBlock->NZ; i++) {
        csrBlock->JA[i] = csr->JA[baseIndex + i];
        csrBlock->AS[i] = csr->AS[baseIndex + i];
    }

    return csrBlock;
}*/

/*HLLMatrix *convert_to_HLL(CSRMatrix *csrMatrix, int hackSize) {
    HLLMatrix *hllMatrix = (HLLMatrix *) malloc(sizeof(HLLMatrix));
    if (!hllMatrix) {
        perror("Memory allocation error");
        return NULL;
    }

    hllMatrix->hackSize = hackSize;
    hllMatrix->N = csrMatrix->N;

    //Compute number of blocks:
    hllMatrix->numBlocks = (csrMatrix->M + hackSize - 1) / hackSize; //Rounding up

    //Allocate memory:
    hllMatrix->blocks = (ELLPACKMatrix **)malloc(hllMatrix->numBlocks * sizeof(ELLPACKMatrix*));

    //Manage each block:
    for (MatT i = 0; i < hllMatrix->numBlocks; i++) {
        //Define start and end point
        MatT row_start = i*hllMatrix->hackSize;
        MatT row_end = (row_start + hllMatrix->hackSize > csrMatrix->M) ? csrMatrix->M : row_start + hllMatrix->hackSize;

        CSRMatrix *csrBlock = extractCSRBlock(csrMatrix, row_start, row_end);
        if (!csrBlock) {
            perror("Memory allocation error");
            return NULL;
        }
        hllMatrix->blocks[i] = convert_to_ELLPACK(csrBlock);
        free_CSRMatrix(csrBlock);
    }

    return hllMatrix;
}*/

ELLPACKMatrix *convert_to_ELLPACK(CSRMatrix *csrMatrix) {

    ELLPACKMatrix *A = malloc(sizeof(ELLPACKMatrix));
    if(!A) {
        perror("Memory allocation error");
        return NULL;
    }

    A->M = csrMatrix->M;
    A->N = csrMatrix->N;

    // Find maxnz for row:
    A->MAXNZ = 0;
    MatT row_nnz;
    for (int i = 0; i < csrMatrix->M; i++) {
        //Derive from definition of IRP
        row_nnz = csrMatrix->IRP[i + 1] - csrMatrix->IRP[i];
        if (row_nnz > A->MAXNZ) {
            A->MAXNZ = row_nnz;
        }
    }

    // Allocate memory for JA and AS:
    A->JA = (MatT **) malloc(A->M * sizeof(MatT *));
    A->AS = (MatVal **) malloc(A->M * sizeof(MatVal *));

    for (int i = 0; i < A->M; i++) {
        A->JA[i] = (MatT *) calloc(A->MAXNZ, sizeof(MatT));
        A->AS[i] = (MatVal *) calloc(A->MAXNZ, sizeof(MatVal));

        if (!A->JA[i] || !A->AS[i]) {
            perror("Memory allocation error");
            // Deallocate all previous rows:
            for (int j = 0; j < i; j++) {
                free(A->JA[j]);
                free(A->AS[j]);
            }
            free(A->JA);
            free(A->AS);
            free(A);

            return NULL;
        }
    }

    // FILL A->JA and A->AS

    for (MatT i = 0; i < A->M; i++) {

        row_nnz = csrMatrix->IRP[i+1] - csrMatrix->IRP[i];
        MatT row_start = csrMatrix->IRP[i];
        MatT j;

        for (j = 0; j < row_nnz; j++) {
            A->JA[i][j] = csrMatrix->JA[row_start + j];
            A->AS[i][j] = csrMatrix->AS[row_start + j];

        }
        //Fill JA matrix with last valid index if row_nnz < maxnz
        for (; j < A->MAXNZ; j++) {
            A->JA[i][j] = (j>0) ? A->JA[i][j-1] : 0;
        }
    }

    return A;
}
