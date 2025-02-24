#include <stdio.h>
#include <stdlib.h>
#include "header/matrixPreProcessing.h"
#include <unistd.h>
#include <limits.h>

int main(void) {
    FILE *f;
    //Open Matrix Market file
    //TODO: Gestire l'apertura di diversi file

    if ((f = fopen("../matrix/mhda416.mtx", "r")) == NULL) {
        perror("Error opening file\n");
        exit(-1);
    }
    //Read matrix and save into CSR rappresentation
    MatrixData data = read_matrix(f);
    printf("Data[1][1]: %d %d %f\n", data.I[0], data.J[0], data.val[0]);
    //convert_to_CSR(data);
}