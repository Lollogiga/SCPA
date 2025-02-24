//
// Created by lollogiga on 2/23/25.
//

#ifndef MATRIXPREPROCESSING_H
#define MATRIXPREPROCESSING_H

typedef struct {
    int M, N, NZ;
    int *I; //Row Index
    int *J; //Column Index
    double *val; //Value of element Matrix(i,j)
}MatrixData;

typedef struct{
    int M, N, NZ;
    int *IRP; //Puntatore a inizio riga
    int *JA; //Indici di colonna
    double *AS; //Valori
}CSRMatrix;

MatrixData read_matrix(FILE *f);
CSRMatrix convert_to_CSR(MatrixData matrix);
#endif //MATRIXPREPROCESSING_H
