#include <stdio.h>
#include <stdlib.h>
#include "./include/matrixPreProcessing.h"
#include <unistd.h>
#include <limits.h>

int main(void) {
    FILE *f;
    //Open Matrix Market file
    //TODO: Gestire l'apertura di diversi file

    if ((f = fopen("../matrix/example.mtx", "r")) == NULL) {
        perror("Error opening file\n");
        exit(-1);
    }

    // Read file and save matrix into MatrixMarket format
    MatrixData rawMatrixData  = read_matrix(f);
    if (rawMatrixData .val == NULL) {
        perror("Error reading matrix data\n");
        exit(-1);
    }

    //Convert in CSR format:
    CSRMatrix csrMatrix = convert_to_CSR(rawMatrixData );

    ELLPACKMatrix ellpackMatrix = convert_to_ELLPACK(rawMatrixData );

    for (int i =0; i < ellpackMatrix.N; i++) {
        for (int j = 0; j < ellpackMatrix.MAXNZ; j++) {
            printf("JA[%d][%d] = %f\n", i, j, ellpackMatrix.AS[i][j]);
        }
    }
    //Free memory:
    free(rawMatrixData.val);
    free(rawMatrixData.I);
    free(rawMatrixData.J);

    free(csrMatrix.IRP);
    free(csrMatrix.JA);
    free(csrMatrix.AS);

    free(ellpackMatrix.JA);
    free(ellpackMatrix.AS);

    return 0;
}