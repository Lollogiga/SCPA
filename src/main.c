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

    //Read matrix and save into CSR rappresentation
    MatrixData data = read_matrix(f);


    //Convert in CSR format:
    CSRMatrix A = convert_to_CSR(data);


    ELLPACKMatrix E = convert_to_ELLPACK(data);

    for (int i =0; i < E.N; i++) {
        for (int j = 0; j < E.MAXNZ; j++) {
            printf("JA[%d][%d] = %f\n", i, j, E.AS[i][j]);
        }
    }
    //Free memory:
    free(data.val);
    free(data.I);
    free(data.J);

    free(A.IRP);
    free(A.JA);
    free(A.AS);

    free(*E.JA);
    free(*E.AS);

    return 0;
}