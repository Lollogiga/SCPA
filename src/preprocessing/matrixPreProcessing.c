#include <stdio.h>
#include <stdlib.h>

#include "../include/mmio.h"
#include "../include/matrixPreProcessing.h"


MatrixData read_matrix(FILE * f) {

    MatrixData data;
    MM_typecode matcode;
    int i;
    double *val;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((mm_read_mtx_crd_size(f, &data.M, &data.N, &data.NZ)) !=0)
        exit(1);

    //Check if matrix is symmetric:
    int sym = mm_is_symmetric(matcode);
    printf("Matrix is symmetric?: %d\n", sym);
    int nz_alloc = sym ? 2 * data.NZ : data.NZ; //If matrix is sym, the value of Nz is at most double

    data.I = (int *) malloc(nz_alloc * sizeof(int));
    data.J = (int *) malloc(nz_alloc * sizeof(int));
    data.val = (double *) malloc(nz_alloc * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    int index = 0;
    for (i=0; i<data.NZ; i++)
    {
        int row, col;
        double val;
        fscanf(f, "%d %d %lg\n", &row, &col, &val);
        row--;  /* adjust from 1-based to 0-based */
        col--;  /* adjust from 1-based to 0-based */

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

CSRMatrix convert_to_CSR(MatrixData data) {
    CSRMatrix A;
    A.M = data.M;
    A.N = data.N;
    A.NZ = data.NZ;

    A.IRP = (int *) calloc(A.M + 1, sizeof(int));
    A.JA  = (int *) malloc(A.NZ * sizeof(int));
    A.AS  = (double *) malloc(A.NZ * sizeof(double));

    //Count NZ for each row:
    for (int i=0; i<data.NZ; i++) {
        A.IRP[data.I[i] + 1]++;
    }

    //Calcutate IRP: A.IRP[i] = A.IRP[i-1] + #NZ_at_i-1:
    for (int i=1; i<= A.M; i++) {
        A.IRP[i] += A.IRP[i-1];
    }

    // 4. Inserimento degli elementi in JA e AS
    int *row_offset = calloc(A.M, sizeof(int));
    for (int i = 0; i < data.NZ; i++) {
        int row = data.I[i];
        int pos = A.IRP[row] + row_offset[row];
        A.JA[pos] = data.J[i];
        A.AS[pos] = data.val[i];
        row_offset[row]++;
    }
    //free(row_offset);

    return A;
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