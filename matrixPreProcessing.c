#include <stdio.h>
#include <stdlib.h>

#include "header/mmio.h"
#include "header/matrixPreProcessing.h"


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

    data.I = (int *) malloc(data.NZ * sizeof(int));
    data.J = (int *) malloc(data.NZ * sizeof(int));
    data.val = (double *) malloc(data.NZ * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<data.NZ; i++)
    {
        fscanf(f, "%d %d %lg\n", &data.I[i], &data.J[i], &data.val[i]);
        printf("%d %d %lg\n", data.I[i], data.J[i], data.val[i]);
        data.I[i]--;  /* adjust from 1-based to 0-based */
        data.J[i]--;
    }
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
}