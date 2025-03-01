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
    CSRMatrix *csrMatrix = convert_to_CSR(rawMatrixData);
    if (csrMatrix == NULL) {
        perror("Error convert_to_CSR\n");
        exit(-1);
    }

    // Convert in ELLPACK format:
    ELLPACKMatrix *ellpackMatrix = convert_to_ELLPACK(csrMatrix);
    if (ellpackMatrix == NULL) {
        perror("Error convert_to_ELLPACK\n");
        exit(-1);
    }

    ELLPACKMatrix *subEllpackMatrix = convert_to_ELLPACK_parametrized(csrMatrix, 0, 2);
    if (subEllpackMatrix == NULL) {
        perror("Error convert_to_ELLPACK_parametrized\n");
        exit(-1);
    }

    HLLMatrix *hllMatrix = convert_to_HLL(csrMatrix, 2);
    if (hllMatrix == NULL) {
        perror("Error convert_to_HLL\n");
        exit(-1);
    }

    // Print for debugging:
    printf("\nMatrix: \n");
    print_matrix_data_verbose(rawMatrixData,false);

    printf("\nCSR Matrix: \n");
    print_csr_matrix_verbose(csrMatrix, false);

    printf("\nELLPACK Matrix: \n");
    print_ellpack_matrix_verbose(ellpackMatrix, false);

    // Convert in ELLPACK format:
    printf("\nELLPACK Matrix reduced: \n");
    print_ellpack_matrix_verbose(subEllpackMatrix, false);

    //Free memory:
    free_MatrixData(rawMatrixData);
    free_CSRMatrix(csrMatrix);
    free_ELLPACKMatrix(ellpackMatrix);
    free_ELLPACKMatrix(subEllpackMatrix);
    free_HLLMatrix(hllMatrix);

    return 0;
}



