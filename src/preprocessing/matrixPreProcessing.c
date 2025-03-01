#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    //Check if matrix is symmetric:
    const char sym = mm_is_symmetric(matcode);

    // Check if matrix has a pattern
    const char pattern = mm_is_pattern(matcode);

    // Find out the size of the sparse matrix (number of rows, columns, and non-zero entries)
    if (mm_read_mtx_crd_size(f, &data->M, &data->N, &data->NZ) != 0) {
        perror("Error reading matrix data\n");
        free(data);
        return NULL;
    }

    const MatT nz_alloc = sym ? 2 * data->NZ : data->NZ; //If matrix is sym, the value of NZ is at most double

    // Dynamically allocate memory for the sparse matrix's row indices, column indices, and values
    data->I = (MatT *) malloc(nz_alloc * sizeof(MatT));
    data->J = (MatT *) malloc(nz_alloc * sizeof(MatT));
    data->val = (MatVal *) malloc(nz_alloc * sizeof(MatVal));

    if (!data->I || !data->J || !data->val) {
        perror("Memory allocation error");
        if (data->I) free(data->I);
        if (data->J) free(data->J);
        if (data->val) free(data->val);

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

        if (!fgets(line, sizeof(line), f)) {
            perror("Error reading matrix data\n");

            free(data->I);
            free(data->J);
            free(data->val);

            free(data);

            return NULL;
        }

        char *endptr;

        MatT row = strtol(line, &endptr, 10);  // Convert first integer
        if (*endptr != ' ' && *endptr != '\t') {
            return NULL;
        }

        MatT col = strtol(endptr, &endptr, 10);  // Convert second integer
        MatVal val = 1L;
        if (pattern) {
            if (*endptr != '\n' && *endptr != '\0') {
                return NULL;
            }
        } else {
            if (*endptr != ' ' && *endptr != '\t') {
                return NULL;
            }

            val = (MatVal) strtod(endptr, &endptr);  // Convert non-zero value
            if (*endptr != '\n' && *endptr != '\0') {
                return NULL;
            }
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

    data->NZ = index; //Update the effective number of NZ

    if (f && f != stdin) fclose(f);

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
        if (csrMatrix->IRP) free(csrMatrix->IRP);
        if (csrMatrix->JA) free(csrMatrix->JA);
        if (csrMatrix->AS) free(csrMatrix->AS);
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

ELLPACKMatrix *convert_to_ELLPACK(CSRMatrix *csr) {
    if (!csr) {
        errno = EINVAL;
        perror("convert_to_ELLPACK: CSR matrix not initialized");

        return NULL;
    }

    ELLPACKMatrix *ell = malloc(sizeof(ELLPACKMatrix));
    if (!ell) {
        errno = EINVAL;
        perror("convert_to_ELLPACK: Memory allocation error");

        return NULL;
    }

    ell->M = csr->M;
    ell->N = csr->N;

    // Find max nz for row:
    ell->MAXNZ = 0;
    MatT row_nnz;
    for (int i = 0; i < csr->M; i++) {
        //Derive from definition of IRP
        row_nnz = csr->IRP[i + 1] - csr->IRP[i];
        if (row_nnz > ell->MAXNZ) {
            ell->MAXNZ = row_nnz;
        }
    }

    /* Allocate memory for JA and AS: */
    ell->JA = (MatT **) malloc(ell->M * sizeof(MatT *));
    ell->AS = (MatVal **) malloc(ell->M * sizeof(MatVal *));

    for (int i = 0; i < ell->M; i++) {
        ell->JA[i] = (MatT *) calloc(ell->MAXNZ, sizeof(MatT));
        ell->AS[i] = (MatVal *) calloc(ell->MAXNZ, sizeof(MatVal));

        // Check if there is an error during allocation
        if (!ell->JA[i] || !ell->AS[i]) {
            perror("convert_to_ELLPACK: Memory allocation error");

            // Deallocate all previous rows:
            for (int j = 0; j < i; j++) {
                free(ell->JA[j]);
                free(ell->AS[j]);
            }

            free(ell->JA);
            free(ell->AS);
            free(ell);

            return NULL;
        }
    }

    // FILL ell->JA and ell->AS
    for (MatT i = 0; i < ell->M; i++) {

        row_nnz = csr->IRP[i+1] - csr->IRP[i];
        MatT row_start = csr->IRP[i];
        MatT j;

        for (j = 0; j < row_nnz; j++) {
            ell->JA[i][j] = csr->JA[row_start + j];
            ell->AS[i][j] = csr->AS[row_start + j];
        }

        // Fill JA matrix with last valid index if row_nnz < maxnz
        for (; j < ell->MAXNZ; j++) {
            ell->JA[i][j] = (j > 0) ? ell->JA[i][j-1] : 0;
        }
    }

    return ell;
}

ELLPACKMatrix *convert_to_ELLPACK_parametrized(CSRMatrix *csr, MatT iStart, MatT iEnd) {
    if (!csr) {
        errno = EINVAL;
        fprintf(stderr, "convert_to_ELLPACK_parametrized: Error: %s - CSR matrix not initialized\n", strerror(errno));

        return NULL;
    }

    if (iStart >= iEnd) {
        errno = EINVAL;
        fprintf(stderr, "convert_to_ELLPACK_parametrized: Error: %s - Invalid row index, iStart > iEnd\n", strerror(errno));

        return NULL;
    }

    if (iStart >= csr->M) {
        errno = EINVAL;
        fprintf(stderr, "convert_to_ELLPACK_parametrized: Error: %s - Invalid row index, iStart >= csr->M\n", strerror(errno));

        return NULL;
    }

    if (iEnd > csr->M)
        iEnd = csr->M;

    ELLPACKMatrix *ell = malloc(sizeof(ELLPACKMatrix));
    if(!ell) {
        errno = ENOMEM;
        fprintf(stderr, "convert_to_ELLPACK_parametrized: Error: %s - Problem allocating \"ELLPACKMatrix\"\n", strerror(errno));

        return NULL;
    }

    ell->M = iEnd - iStart;
    ell->N = csr->N;

    // Find max nz into rows
    ell->MAXNZ = 0;
    MatT row_nnz = 0;
    for (MatT i = iStart; i < iEnd; i++) {
        row_nnz = csr->IRP[i+1] - csr->IRP[i];
        if (row_nnz > ell->MAXNZ) {
            ell->MAXNZ = row_nnz;
        }
    }

    if (row_nnz == 0) {
        printf("Sub-matrix between indexes %d - %d is totally empty\n", iStart, iEnd);
    }

    /* Allocate memory for JA and AS: */
    ell->JA = (MatT **) malloc((iEnd - iStart) * sizeof(MatT *));
    ell->AS = (MatVal **) malloc((iEnd - iStart) * sizeof(MatVal *));
    if (!ell->JA || !ell->AS) {
        fprintf(stderr, "convert_to_ELLPACK_parametrized: Error: %s - Problem allocating arrays \"JA\" or \"AS\" for \"ELLPACKMatrix\"\n", strerror(errno));

        if (ell->JA) free(ell->JA);
        if (ell->AS) free(ell->AS);

        free(ell);

        return NULL;
    }

    for (MatT i = 0; i < ell->M; i++) {
        ell->JA[i] = (MatT *) calloc(ell->MAXNZ, sizeof(MatT));
        ell->AS[i] = (MatVal *) calloc(ell->MAXNZ, sizeof(MatVal));

        // Check if there is an error during allocation
        if (!ell->JA[i] || !ell->AS[i]) {
            fprintf(stderr, "convert_to_ELLPACK_parametrized: Error: %s - Problem allocating inner \"JA\" or \"AS\" for \"ELLPACKMatrix\"\n", strerror(errno));

            // Deallocate all previous rows:
            for (int j = 0; j < i; j++) {
                free(ell->JA[j]);
                free(ell->AS[j]);
            }

            if (ell->JA[i]) free(ell->JA[i]);
            if (ell->AS[i]) free(ell->AS[i]);

            free(ell->JA);
            free(ell->AS);

            free(ell);

            return NULL;
        }
    }

    // FILL ell->JA and ell->AS
    for (MatT i = 0; i < ell->M; i++) {

        row_nnz = csr->IRP[(i + iStart) + 1] - csr->IRP[i + iStart];
        MatT row_start = csr->IRP[i + iStart];
        MatT j;

        for (j = 0; j < row_nnz; j++) {
            ell->JA[i][j] = csr->JA[row_start + j];
            ell->AS[i][j] = csr->AS[row_start + j];
        }

        // Fill JA matrix with last valid index if row_nnz < maxnz
        for (; j < ell->MAXNZ; j++) {
            ell->JA[i][j] = (j > 0) ? ell->JA[i][j-1] : 0;
        }
    }

    return ell;
}

HLLMatrix *convert_to_HLL(CSRMatrix *csr, int hackSize) {
    if (!csr) {
        errno = EINVAL;
        fprintf(stderr, "convert_to_HLL: Error: %s - CSRMatrix input is invalid\n", strerror(errno));

        return NULL;
    }

    HLLMatrix *hllMatrix = malloc(sizeof(HLLMatrix));
    if (!hllMatrix) {
        perror("Memory allocation error");
        return NULL;
    }

    hllMatrix->hackSize = hackSize;
    hllMatrix->N = csr->N;

    //Compute number of blocks:
    hllMatrix->numBlocks = (csr->M + hackSize - 1) / hackSize; //Rounding up

    //Allocate memory:
    hllMatrix->blocks = (ELLPACKMatrix **)malloc(hllMatrix->numBlocks * sizeof(ELLPACKMatrix*));
    if (!hllMatrix->blocks) {
        perror("Memory allocation error");

        free(hllMatrix); // Free the allocated hllMatrix as well

        return NULL;
    }

    //Manage each block:
    for (MatT i = 0; i < hllMatrix->numBlocks; i++) {
        //Define start and end point
        MatT row_start = i * hllMatrix->hackSize;
        MatT row_end = (row_start + hllMatrix->hackSize > csr->M) ? csr->M : row_start + hllMatrix->hackSize;

        // Leak of memory allocated in function 'convert_to_ELLPACK_parametrized'
        hllMatrix->blocks[i] = convert_to_ELLPACK_parametrized(csr, row_start, row_end);
        if (!hllMatrix->blocks[i]) {
            perror("Memory allocation error");

            for (MatT j = 0; j < i; j++) free(hllMatrix->blocks[j]);

            free(hllMatrix->blocks);
            free(hllMatrix);

            return NULL;
        }
    }

    return hllMatrix;
}