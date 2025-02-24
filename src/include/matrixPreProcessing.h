//
// Created by lollogiga on 2/23/25.
//

#ifndef MATRIXPREPROCESSING_H
#define MATRIXPREPROCESSING_H

typedef struct {
    int M, N, NZ;
    int *I;         //Row Index
    int *J;         //Column Index
    double *val;    //Value of element Matrix(i,j)
} MatrixData;

typedef struct {
    int M, N, NZ;
    int *IRP;       //Puntatore a inizio riga
    int *JA;        //Indici di colonna
    double *AS;     //Valori
} CSRMatrix;

typedef struct {
    int M, N;
    int MAXNZ;      //Numero massimo di non-zero per riga
    int **JA;       //2D Array di indici di colonna
    double **AS;    //2D Array di coefficienti
} ELLPACKMatrix;

MatrixData read_matrix(FILE *f);
CSRMatrix convert_to_CSR(MatrixData matrix);
ELLPACKMatrix convert_to_ELLPACK(MatrixData matrix);
#endif //MATRIXPREPROCESSING_H
