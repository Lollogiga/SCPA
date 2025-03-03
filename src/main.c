#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include "./include/matrixPreProcessing.h"
#include "./include/matricDealloc.h"
#include "./include/matrixPrint.h"
#include "./include/serialProduct.h"
#include "./include/utilsProduct.h"

int checkFolder(char *checkFolder, char **destFolder) {
    struct stat info;

    if (checkFolder && stat(checkFolder, &info) == 0 && S_ISDIR(info.st_mode)) {
        char *folder = malloc(strlen(checkFolder) + 1);
        if (!folder) {
            perror("checkFolder: error allocating space to forder path");
            return -1;
        }

        strcpy(folder, checkFolder);

        *destFolder = folder;

        return 0;
    }

#ifdef TEST
    if (stat(MATRIX_TEST_FOLDER_DEFAULT, &info) == 0 && (info.st_mode & S_IFDIR)) {
        char *folder = malloc(strlen(MATRIX_TEST_FOLDER_DEFAULT) + 1);
        if (!folder) {
            perror("checkFolder: error allocating space to forder path");
            return -1;
        }

        strcpy(folder, MATRIX_TEST_FOLDER_DEFAULT);

        *destFolder = folder;

        return 0;
    }
#endif

    if (stat(MATRIX_FOLDER_DEFAULT, &info) == 0 && (info.st_mode & S_IFDIR)) {
        char *folder = malloc(strlen(MATRIX_FOLDER_DEFAULT) + 1);
        if (!folder) {
            perror("checkFolder: error allocating space to forder path");
            return -1;
        }

        strcpy(folder, MATRIX_FOLDER_DEFAULT);

        *destFolder = folder;

        return 0;
    }

    return 1;
}

int computeMatrixFile(char *matrixFile) {
    FILE *f;
    // Open Matrix Market file
    if ((f = fopen(matrixFile, "r")) == NULL) {
        perror("Error opening file\n");
        return -1;
    }

    // Read file and save matrix into MatrixMarket format
    MatrixData *rawMatrixData  = read_matrix(f);
    if (rawMatrixData == NULL) {
        perror("Error reading matrix data\n");
    }

    // Convert in CSR format:
    CSRMatrix *csrMatrix = convert_to_CSR(rawMatrixData);
    if (csrMatrix == NULL) {
        perror("Error convert_to_CSR\n");
    }

    // Convert in ELLPACK format:
    ELLPACKMatrix *ellpackMatrix = convert_to_ELLPACK(csrMatrix);
    if (ellpackMatrix == NULL) {
        perror("Error convert_to_ELLPACK\n");
    }

    ELLPACKMatrix *subEllpackMatrix = convert_to_ELLPACK_parametrized(csrMatrix, 0, 2);
    if (subEllpackMatrix == NULL) {
        perror("Error convert_to_ELLPACK_parametrized\n");
    }

    HLLMatrix *hllMatrix = convert_to_HLL(csrMatrix, 2);
    if (hllMatrix == NULL) {
        perror("Error convert_to_HLL\n");
    }


    ResultVector *csr_product = csr_serialProduct(csrMatrix, create_vector(csrMatrix->N));
    if (csr_product == NULL) {
        perror("Error csr_SerialProduct\n");
    }

    // ResultVector *hll_product = hll_serialProduct(hllMatrix, create_vector(hllMatrix->N));

    // Print for debugging:
    printf("\nMatrix: \n");
    print_matrix_data_verbose(rawMatrixData,false);

    printf("\nCSR Matrix: \n");
    print_csr_matrix_verbose(csrMatrix, false);

    printf("\nELLPACK Matrix: \n");
    print_ellpack_matrix_verbose(ellpackMatrix, false);

    printf("\nELLPACK Matrix reduced: \n");
    print_ellpack_matrix_verbose(subEllpackMatrix, false);

    printf("\nResult csr vector: \n");
    print_result_vector(csr_product);

    // printf("\nResult hll vector: \n");
    // print_result_vector(hll_product);

    //Free memory:
    free_MatrixData(rawMatrixData);
    free_CSRMatrix(csrMatrix);
    free_ELLPACKMatrix(ellpackMatrix);
    free_ELLPACKMatrix(subEllpackMatrix);
    free_HLLMatrix(hllMatrix);
    free_ResultVector(csr_product);

    fclose(f);

    return 0;
}

int main(int argc, char *argv[]) {
    char *folder = NULL;

    if (argc != 2) {
        printf("Usage: ./main matrixFolder\n");

        printf("Trying to execute via default folders\n");
        if (checkFolder(NULL, &folder)) {
            perror("No available folder found. Please pass a folder path to program arguments or check if in the project folder exist 'matrix' or 'matrixTest' folder.");

            return -1;
        }
    } else {
        if (checkFolder(argv[1], &folder)) {
            perror("No available folder found. Please pass a folder path to program arguments or check if in the project folder exist 'matrix' or 'matrixTest' folder.");

            return -1;
        }
    }

    if (!folder) {
        perror("No folder found. Please check program arguments");

        return -1;
    }

    struct dirent *entry;
    DIR *dir = opendir(folder);

    if (dir == NULL) {
        perror("Error opening folder");
        return -1;
    }

    printf("Folder path: %s\n", folder);

    while ((entry = readdir(dir))) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;  // Skip loop iteration
        }

        char *filePath = malloc(strlen(folder) + strlen(entry->d_name) + 1);
        if (filePath == NULL) {
            perror("Memory allocation failed");
            closedir(dir);
            return -1;
        }

        snprintf(filePath, strlen(folder) + strlen(entry->d_name) + 1, "%s%s", folder, entry->d_name);
        printf("FilePath: %s\n", filePath);

        computeMatrixFile(filePath);

        free(filePath);
    }

    free(folder);

    return 0;
}



