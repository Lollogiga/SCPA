#ifndef CONSTANTS_H
#define CONSTANTS_H

// Constants
#define MATRIX_FILE_MAX_ROW_LENGTH 256 // Maximum length of a row in the matrix file
#define HACK_SIZE 32

#define MAX_PRINT_ROW 20
#define MAX_PRINT_COLUMN 20

static const char MATRIX_FOLDER_DEFAULT[] = "../matrix/";
static const char MATRIX_TEST_FOLDER_DEFAULT[] = "../matrixTest/";

// Typedefs for matrix storage
typedef int MatT; // Used for row/column indices, non-zero count, data info and others
typedef float MatVal; // Used for matrix values

#endif // CONSTANTS_H