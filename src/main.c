#include <stdio.h>
#include <stdlib.h>
#include "./include/matrixPreProcessing.h"
#include <unistd.h>
#include <limits.h>



//TODO: Gestire l'apertura di diversi file
int main(void) {

    FILE *f;
    //Open Matrix Market file
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

    //Print for debugging:
    printf("Matrix: \n");
    print_matrix_data(rawMatrixData);
    printf("CSR Matrix: \n");
    print_csr_matrix(csrMatrix);
    printf("ELLPACK Matrix: \n");
    print_ellpack_matrix(ellpackMatrix);


    //Free memory:
    free(rawMatrixData.val);
    free(rawMatrixData.I);
    free(rawMatrixData.J);

    free(csrMatrix.IRP);
    free(csrMatrix.JA);
    free(csrMatrix.AS);

    free(*ellpackMatrix.JA);
    free(*ellpackMatrix.AS);

    return 0;
}