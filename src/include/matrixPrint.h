//
// Created by buniy on 28/02/2025.
//

#ifndef MATRIXPRINT_H
#define MATRIXPRINT_H

#include "./matrixPreProcessing.h"

// Function for debugging:
void print_ellpack_matrix(ELLPACKMatrix *matrix);
void print_matrix_data(MatrixData *csr);
void print_csr_matrix(CSRMatrix *ell);

#endif //MATRIXPRINT_H
