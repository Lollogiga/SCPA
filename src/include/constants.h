#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdint.h>

// Constants
#define MATRIX_FILE_MAX_ROW_LENGTH 256 // Maximum length of a row in the matrix file

// Typedefs for matrix storage
typedef uint32_t MatT; // Used for row/column indices, non-zero count, data info and others
typedef float MatVal; // Used for matrix values

#endif // CONSTANTS_H