#include <stdio.h>
#include <stdlib.h>

#include "../include/mmio.h"
#include "../include/matrixPreProcessing.h"

#include "../include/constants.h"

MatrixData read_matrix(FILE *f) {

    MatrixData data = {0};
    MM_typecode matcode;

    // Read and check the presence of the banner in the Matrix Market file
    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");

        data.val = NULL;
        return data;
    }

    /*
    * This block of code allows filtering matrix types if the application
    * only supports a subset of Matrix Market data types.
    */

    // Check if the matrix is complex, a matrix type (not a vector or something else), and sparse
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) ) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));

        data.val = NULL;
        return data;
    }

    // Find out the size of the sparse matrix (number of rows, columns, and non-zero entries)
    if (mm_read_mtx_crd_size(f, &data.M, &data.N, &data.NZ) != 0) {
        perror("Error reading matrix data\n");

        data.val = NULL;
        return data;
    }

    //Check if matrix is symmetric:
    const char sym = mm_is_symmetric(matcode);
    printf("Matrix is symmetric?: %d\n", sym);
    const MatT nz_alloc = sym ? 2 * data.NZ : data.NZ; //If matrix is sym, the value of NZ is at most double

    // Dynamically allocate memory for the sparse matrix's row indices, column indices, and values
    data.I = (MatT *) malloc(nz_alloc * sizeof(MatT));
    data.J = (MatT *) malloc(nz_alloc * sizeof(MatT));
    data.val = (MatVal *) malloc(nz_alloc * sizeof(MatVal));

    /*
     * NOTE: When reading double values, ANSI C requires the use of the "l" specifier,
     * as in "%lg", "%lf", or "%le". Otherwise, errors can occur. This is part of ANSI C
     * (X3.159-1989, Section 4.9.6.2, pages 136 lines 13-15).
     *
     * fscanf(f, "%d %d %lg\n", &data.I[i], &data.J[i], &data.val[i]);
     */

    int index = 0;
    for (int i = 0; i < data.NZ; i++) {
        /* Retrieve row, col and val from file line */
        char line[MATRIX_FILE_MAX_ROW_LENGTH];  // assuming each line is small enough for this buffer
        if (fgets(line, sizeof(line), f)) {
            char *endptr;

            MatT row = strtol(line, &endptr, 10);  // Convert first integer
            if (*endptr != ' ' && *endptr != '\t') {
                data.val = NULL;
                return data;
            }

            MatT col = strtol(endptr, &endptr, 10);  // Convert second integer
            if (*endptr != ' ' && *endptr != '\t') {
                data.val = NULL;
                return data;
            }

            const MatVal val = (MatVal) strtod(endptr, &endptr);  // Convert non-zero value
            if (*endptr != '\n' && *endptr != '\0') {
                data.val = NULL;
                return data;
            }

            /* Adjust from 1-based to 0-based */
            row--;
            col--;

            data.I[index] = row;
            data.J[index] = col;
            data.val[index] = val;

            index++;

            //If the matrix is symmetric, and we aren't on diagonal:
            if (sym && row != col) {
                data.I[index] = col;
                data.J[index] = row;
                data.val[index] = val;
                index++;
            }
        }
    }

    data.NZ = index; //Update the effective number of NZ

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

CSRMatrix convert_to_CSR(MatrixData rawMatrixData) {
    CSRMatrix csrMatrix;
    csrMatrix.M = rawMatrixData.M;
    csrMatrix.N = rawMatrixData.N;
    csrMatrix.NZ = rawMatrixData.NZ;

    csrMatrix.IRP = (MatT *) calloc(csrMatrix.M + 1, sizeof(MatT));
    csrMatrix.JA  = (MatT *) malloc(csrMatrix.NZ * sizeof(MatT));
    csrMatrix.AS  = (MatVal *) malloc(csrMatrix.NZ * sizeof(MatVal));

    //Count NZ for each row:
    for (int i = 0; i < rawMatrixData.NZ; i++) {
        csrMatrix.IRP[rawMatrixData.I[i] + 1]++;
    }

    //Calcutate IRP: A.IRP[i] = A.IRP[i-1] + #NZ_at_i-1:
    for (int i = 1; i <= csrMatrix.M; i++) {
        csrMatrix.IRP[i] += csrMatrix.IRP[i-1];
    }

    // Inserimento degli elementi in JA e AS
    for (int i = 0; i < rawMatrixData.NZ; i++) {
        csrMatrix.JA[i] = rawMatrixData.J[i];
        csrMatrix.AS[i] = rawMatrixData.val[i];
    }

    return csrMatrix;
}

ELLPACKMatrix convert_to_ELLPACK(MatrixData rawMatrixData) {
    ELLPACKMatrix A;

    //Calculate #NZ for each rows
    int *row_counts = calloc(rawMatrixData.M, sizeof(MatT));
    for (int i = 0; i < rawMatrixData.NZ; i++)
        row_counts[rawMatrixData.I[i]]++;

    //Search for max:
    int maxnz = 0;
    for (int i = 0; i < rawMatrixData.M; i++) {
        if (row_counts[i] > maxnz)
            maxnz = row_counts[i];
    }

    //Fill ELLPACK struct
    A.M = rawMatrixData.M;
    A.N = rawMatrixData.N;
    A.MAXNZ = maxnz;
    A.JA = (MatT **) malloc(rawMatrixData.M * sizeof(MatT));
    A.AS = (MatVal **) malloc(rawMatrixData.M * sizeof(MatVal));
    for (int i = 0; i < rawMatrixData.M; i++) {
        A.JA[i] = (MatT *) calloc(maxnz, sizeof(MatT));
        A.AS[i] = (MatVal *) calloc(maxnz, sizeof(MatVal));
    }

    MatT *row_offset = calloc(rawMatrixData.M, sizeof(MatT));
    for (int i = 0; i < rawMatrixData.NZ; i++) {
        MatT row = rawMatrixData.I[i];
        MatT pos = row_offset[row];  // Posizione corrente nella riga "row"

        A.JA[row][pos] = rawMatrixData.J[i];
        A.AS[row][pos] = rawMatrixData.val[i];

        row_offset[row]++;

        if (row_counts[row] == pos + 1) {
            for (MatT j = pos + 1; j < maxnz; j++)
                A.JA[row][j] = rawMatrixData.J[i];
        }
    }
    free(row_counts);
    free(row_offset);

    return A;
}

//Function for debugging:
void print_matrix_data(MatrixData data) {
    for (MatT i=0; i<data.NZ; i++) {
        printf("A[%d][%d] = %f\n", data.I[i], data.J[i], data.val[i]);
    }
}

void print_csr_matrix(CSRMatrix csrMatrix) {
    for (MatT i = 0; i <= csrMatrix.M; i++) {
        printf("IRP[%d] = %d\n", i, csrMatrix.IRP[i]);
    }

    for (MatT i = 0; i < csrMatrix.NZ; i++) {
        printf("JA[%d] = %d\n", i, csrMatrix.JA[i]);
    }

    for (MatT i = 0; i < csrMatrix.NZ; i++) {
        printf("AS[%d] = %f\n", i, csrMatrix.AS[i]);
    }
}

void print_ellpack_matrix(ELLPACKMatrix ellpackMatrix) {
    for (MatT i = 0; i < ellpackMatrix.N; i++) {
        for (MatT j = 0; j < ellpackMatrix.MAXNZ; j++) {
            printf("JA[%d][%d] = %d\n", i, j, ellpackMatrix.JA[i][j]);
        }
    }

    for (MatT i = 0; i < ellpackMatrix.N; i++) {
        for (MatT j = 0; j < ellpackMatrix.MAXNZ; j++) {
            printf("AS[%d][%d] = %f\n", i, j, ellpackMatrix.AS[i][j]);
        }
    }
}
