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
    int sym = mm_is_symmetric(matcode);
    printf("Matrix is symmetric?: %d\n", sym);
    int nz_alloc = sym ? 2 * data.NZ : data.NZ; //If matrix is sym, the value of Nz is at most double

    // Dynamically allocate memory for the sparse matrix's row indices, column indices, and values
    data.I = (int *) malloc(nz_alloc * sizeof(int));
    data.J = (int *) malloc(nz_alloc * sizeof(int));
    data.val = (double *) malloc(nz_alloc * sizeof(double));


    /*
     * NOTE: When reading double values, ANSI C requires the use of the "l" specifier,
     * as in "%lg", "%lf", or "%le". Otherwise, errors can occur. This is part of ANSI C
     * (X3.159-1989, Section 4.9.6.2, pages 136 lines 13-15).
     *
     * fscanf(f, "%d %d %lg\n", &data.I[i], &data.J[i], &data.val[i]);
     */

    int index = 0;
    for (int i=0; i<data.NZ; i++)
    {
        /* Retrieve row, col and val from file line */
        char line[MAX_LENGTH_MATRIX_FILE_ROW];  // assuming each line is small enough for this buffer
        if (fgets(line, sizeof(line), f)) {
            char *endptr;

            int row = strtol(line, &endptr, 10);  // Convert first integer
            if (*endptr != ' ' && *endptr != '\t') {
                data.val = NULL;
                return data;
            }

            int col = strtol(endptr, &endptr, 10);  // Convert second integer
            if (*endptr != ' ' && *endptr != '\t') {
                data.val = NULL;
                return data;
            }

            const double val = strtod(endptr, &endptr);  // Convert double value
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
            if (sym && data.I[index] != data.J[index]) {
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

    csrMatrix.IRP = (int *) calloc(csrMatrix.M + 1, sizeof(int));
    csrMatrix.JA  = (int *) malloc(csrMatrix.NZ * sizeof(int));
    csrMatrix.AS  = (double *) malloc(csrMatrix.NZ * sizeof(double));

    //Count NZ for each row:
    for (int i=0; i<rawMatrixData.NZ; i++) {
        csrMatrix.IRP[rawMatrixData.I[i] + 1]++;
    }

    //Calcutate IRP: A.IRP[i] = A.IRP[i-1] + #NZ_at_i-1:
    for (int i=1; i <= csrMatrix.M; i++) {
        csrMatrix.IRP[i] += csrMatrix.IRP[i-1];
    }

    // 4. Inserimento degli elementi in JA e AS
    int *row_offset = calloc(csrMatrix.M, sizeof(int));
    for (int i = 0; i < rawMatrixData.NZ; i++) {
        int row = rawMatrixData.I[i];
        int pos = csrMatrix.IRP[row] + row_offset[row];
        csrMatrix.JA[pos] = rawMatrixData.J[i];
        csrMatrix.AS[pos] = rawMatrixData.val[i];
        row_offset[row]++;
    }

    //free(row_offset);

    return csrMatrix;
}

//TODO check
ELLPACKMatrix convert_to_ELLPACK(MatrixData data) {
    ELLPACKMatrix A;

    //1. Calculate #NZ for each rows
    int *row_counts = (int *) calloc(data.M, sizeof(int));
    for (int i=0; i<data.NZ; i++) {
        row_counts[data.I[i]]++;
    }

    //2. Search for max:
    int maxnz = 0;
    for (int i=0; i<data.M; i++) {
        if (row_counts[i] > maxnz) {
            maxnz = row_counts[i];
        }
    }

    //3.Fill ELLPACK struct
    A.M = data.M;
    A.N = data.N;
    A.MAXNZ = maxnz;
    A.JA = (int **) malloc(data.M * sizeof(int));
    A.AS = (double **) malloc(data.M * sizeof(double));
    for (int i=0; i<data.M; i++) {
        A.JA[i] = (int *) malloc(maxnz * sizeof(int));
        A.AS[i] = (double *) malloc(maxnz * sizeof(double));
        //Init matrix
        for (int j=0; j<maxnz; j++) {
            A.JA[i][j] = 0;
            A.AS[i][j] = 0;
        }
    }

    int *row_offset = (int *) calloc(data.M, sizeof(int));

    for (int i = 0; i < data.NZ; i++) {
        int row = data.I[i];
        int pos = row_offset[row];  // Posizione corrente nella riga "row"
        A.JA[row][pos] = data.J[i];
        A.AS[row][pos] = data.val[i];
        row_offset[row]++;
    }

    //free(row_counts);
    //free(row_offset);
    return A;
}