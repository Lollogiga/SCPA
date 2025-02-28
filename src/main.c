#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "./include/matrixPreProcessing.h"
#include "./include/matricDealloc.h"
#include "./include/matrixPrint.h"

// TODO: Gestire l'apertura di diversi file
int main(void) {

    FILE *f;
    // Open Matrix Market file
    if ((f = fopen("../matrixTest/symmetrical_example.mtx", "r")) == NULL) {
        perror("Error opening file\n");
        exit(-1);
    }

    // Read file and save matrix into MatrixMarket format
    MatrixData *rawMatrixData  = read_matrix(f);
    if (rawMatrixData == NULL) {
        perror("Error reading matrix data\n");
        exit(-1);
    }

    // Convert in CSR format:
    CSRMatrix *csrMatrix = convert_to_CSR(rawMatrixData );

    // Convert in ELLPACK format:
    ELLPACKMatrix *ellpackMatrix = convert_to_ELLPACK(csrMatrix);

    // Print for debugging:
    printf("\nMatrix: \n");
    print_matrix_data_verbose(rawMatrixData,false);

    printf("\nCSR Matrix: \n");
    print_csr_matrix_verbose(csrMatrix, false);

    printf("\nELLPACK Matrix: \n");
    print_ellpack_matrix_verbose(ellpackMatrix, false);

    //Free memory:
    free_MatrixData(rawMatrixData);
    free_CSRMatrix(csrMatrix);
    free_ELLPACKMatrix(ellpackMatrix);

    return 0;
}



