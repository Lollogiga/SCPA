//
// Created by lollogiga on 2/23/25.
//

#ifndef MATRIXPREPROCESSING_H
#define MATRIXPREPROCESSING_H

/**
 * @brief Data structure for storing a sparse matrix in MatrixMarket format.
 *
 * This structure is used to represent a sparse matrix by storing only
 * the nonzero values along with their corresponding row and column indices.
 *
 * @b Example: \n
 * Consider the following sparse matrix:
 * <pre>
 *     0   5   0
 *     7   0   0
 *     0   0   3
 * </pre>
 * It can be stored as:
 * <pre>
 *     M = 3, N = 3, NZ = 3
 *
 *     I   = { 0,  1,  2 }  // Row indices of nonzero elements
 *     J   = { 1,  0,  2 }  // Column indices of nonzero elements
 *     val = { 5,  7,  3 }  // Nonzero values
 * </pre>
 * Accessing matrix elements: \n
 * To iterate through the nonzero elements and print their values:
 * @code
 * for (int k = 0; k < matrix.NZ; k++) {
 *     printf("Value at (%d, %d): %f\n", matrix.I[k], matrix.J[k], matrix.val[k]);
 * }
 * @endcode
 */
typedef struct {
    int M;  /**< Total number of rows in the matrix */
    int N;  /**< Total number of columns in the matrix */
    int NZ; /**< Number of nonzero elements in the matrix */

    int *I;      /**< Array of row indices for nonzero elements (size: NZ) */
    int *J;      /**< Array of column indices for nonzero elements (size: NZ) */
    double *val; /**< Array of nonzero values in the matrix (size: NZ) */
} MatrixData;

/**
 * @brief Data structure for storing a sparse matrix in Compressed Sparse Row (CSR) format.
 *
 * The CSR format efficiently stores a sparse matrix by compressing row information,
 * allowing fast matrix-vector multiplications and reduced memory usage.
 *
 * @b Example: \n
 * Consider the following sparse matrix:
 * <pre>
 *     0   5   0
 *     7   0   0
 *     0   0   3
 * </pre>
 * It is stored in CSR format as:
 * <pre>
 *     M   = 3   // Number of rows
 *     N   = 3   // Number of columns
 *     NZ  = 3   // Number of nonzero elements
 *
 *     IRP = { 0, 1, 2, 3 }  // Row pointer array (size: M+1)
 *     JA  = { 1, 0, 2 }      // Column indices of nonzero elements (size: NZ)
 *     AS  = { 5, 7, 3 }      // Nonzero values (size: NZ)
 * </pre>
 * Accessing matrix elements: \n
 * To iterate through the nonzero elements of each row:
 * @code
 * for (int row = 0; row < matrix.M; row++) {
 *     for (int k = matrix.IRP[row]; k < matrix.IRP[row + 1]; k++) {
 *         printf("Value at (%d, %d): %f\n", row, matrix.JA[k], matrix.AS[k]);
 *     }
 * }
 * @endcode
 */
typedef struct {
    int M;  /**< Total number of rows in the matrix */
    int N;  /**< Total number of columns in the matrix */
    int NZ; /**< Number of nonzero elements in the matrix */

    int *IRP;  /**< Row pointer array (size: M+1). \n IRP[i] stores the index in JA/AS where row i starts. */
    int *JA;   /**< Column indices of nonzero elements (size: NZ). */
    double *AS; /**< Array of nonzero values in the matrix (size: NZ). */
} CSRMatrix;

/**
 * @brief Data structure for storing a sparse matrix in ELLPACK format.
 *
 * The ELLPACK format is optimized for vectorized operations and parallel computing.
 * It stores the matrix using a fixed number of nonzero elements per row (`MAXNZ`),
 * ensuring efficient memory access patterns.
 *
 * @b Example: \n
 * Consider the following sparse matrix:
 * <pre>
 *     0   5   0   0
 *     7   0   8   0
 *     0   0   3   6
 * </pre>
 * It is stored in ELLPACK format as:
 * <pre>
 *     M      = 3   // Number of rows
 *     N      = 4   // Number of columns
 *     MAXNZ  = 2   // Maximum number of nonzero elements per row
 *
 *     JA  = {
 *         { 1, -1 },  // Row 0: column indices of nonzero elements
 *         { 0,  2 },  // Row 1: column indices of nonzero elements
 *         { 2,  3 }   // Row 2: column indices of nonzero elements
 *     }
 *
 *     AS  = {
 *         { 5,  0 },  // Row 0: nonzero values (padded with 0)
 *         { 7,  8 },  // Row 1: nonzero values
 *         { 3,  6 }   // Row 2: nonzero values
 *     }
 * </pre>
 * Accessing matrix elements: \n
 * To iterate through the nonzero elements of each row:
 * @code
 * for (int row = 0; row < matrix.M; row++) {
 *     for (int k = 0; k < matrix.MAXNZ; k++) {
 *         if (matrix.JA[row][k] != -1) {  // Ignore padding values
 *             printf("Value at (%d, %d): %f\n", row, matrix.JA[row][k], matrix.AS[row][k]);
 *         }
 *     }
 * }
 * @endcode
 */
typedef struct {
    int M;      /**< Total number of rows in the matrix */
    int N;      /**< Total number of columns in the matrix */
    int MAXNZ;  /**< Maximum number of nonzero elements per row */

    int **JA;   /**< 2D array (size: M x MAXNZ) storing column indices of nonzero elements. */
    double **AS; /**< 2D array (size: M x MAXNZ) storing nonzero values. */
} ELLPACKMatrix;

MatrixData read_matrix(FILE *f);
CSRMatrix convert_to_CSR(MatrixData matrix);
ELLPACKMatrix convert_to_ELLPACK(MatrixData matrix);

//Function for debugging:
void print_ellpack_matrix(ELLPACKMatrix matrix);
void print_matrix_data(MatrixData matrix);
void print_csr_matrix(CSRMatrix matrix);
#endif //MATRIXPREPROCESSING_H
