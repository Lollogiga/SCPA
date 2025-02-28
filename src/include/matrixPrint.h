//
// Created by buniy on 28/02/2025.
//

#ifndef MATRIXPRINT_H
#define MATRIXPRINT_H

#include <stdbool.h>

#include "./matrixPreProcessing.h"

// Function for debugging:
void print_matrix_data(MatrixData *matrix);
void print_matrix_data_verbose(MatrixData *matrix, bool verbose);

void print_csr_matrix(CSRMatrix *csr);
void print_csr_matrix_verbose(CSRMatrix *csr, bool verbose);

void print_ellpack_matrix(ELLPACKMatrix *ell);
void print_ellpack_matrix_verbose(ELLPACKMatrix *ell, bool verbose);

#endif //MATRIXPRINT_H
