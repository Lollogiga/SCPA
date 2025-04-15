#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "./include/fileUtils.h"
#include "./include/mtxFree.h"
#include "./include/preprocessing.h"
#include "./include/performance.h"
#include "./include/cuda/computeCUDA.cuh"
#include "./include/openmp/computeOpenMP.h"

int computeMatrix(const char *matrixFile) {
    FILE *f;
    PerformanceResult performance = {0};

    // Open Matrix Market file
    if ((f = fopen(matrixFile, "r")) == NULL) {
        perror("Error opening file\n");
        return -1;
    }

    strcpy(performance.matrix_name, matrixFile);

    // Read file and save matrix into MatrixMarket format
    RawMatrix *rawMatrix = read_matrix(f);
    if (rawMatrix == NULL) {
        perror("Error reading matrix data\n");

        fclose(f);

        return -1;
    }

    performance.NZ = rawMatrix->NZ;

    // Close FILE pointer
    fclose(f);

    // Convert in CSR format:
    CSRMatrix *csrMatrix = convert_to_CSR(rawMatrix);
    if (csrMatrix == NULL) {
        perror("Error convert_to_CSR\n");

        return -1;
    }

    // Free file matrix
    free_MatrixData(rawMatrix);

    // Convert to HLL format:
    HLLMatrix *hllMatrix = convert_to_HLL(csrMatrix, HACK_SIZE);
    if (hllMatrix == NULL) {
        perror("Error convert_to_HLL\n");

        free_CSRMatrix(csrMatrix);

        return -1;
    }

    // Convert to HLL format:
    HLLMatrixAligned *hllMatrixAligned = convert_to_HLL_aligned(csrMatrix, HACK_SIZE);
    if (hllMatrixAligned == NULL) {
        perror("Error convert_to_HLL_aligned\n");

        free_CSRMatrix(csrMatrix);
        free_HLLMatrix(hllMatrix);

        return -1;
    }

    printf("\n-------------- COMPUTING OpenMP --------------\n");

    int num_threads = 40;
    printf("\n\033[32;7m# of threads:\033[0;32m %d\033[0m\n", num_threads);

    strcpy(performance.implementation, "OpenMP");
    performance.threads = num_threads;

    // Compute OpenMP calculus
    if (computeOpenMP(csrMatrix, hllMatrix, hllMatrixAligned, num_threads, &performance)) {
        perror("Error computeOpenMP\n");
        free_CSRMatrix(csrMatrix);
        free_HLLMatrix(hllMatrix);
        free_HLLMatrixAligned(hllMatrixAligned);

        return -1;
    }

    printf("\n-------------- COMPUTING CUDA --------------\n");

    strcpy(performance.implementation, "OpenMP");
    performance.block_size = 32;
    performance.blocks_per_grid = 192;

    if (computeCUDA(csrMatrix, hllMatrix, hllMatrixAligned, num_threads)) {
        free_CSRMatrix(csrMatrix);
        free_HLLMatrix(hllMatrix);
        free_HLLMatrixAligned(hllMatrixAligned);

        return -1;
    }

    free_CSRMatrix(csrMatrix);
    free_HLLMatrix(hllMatrix);
    free_HLLMatrixAligned(hllMatrixAligned);

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
            perror(
                "No available folder found. Please pass a folder path to program arguments or check if in the project folder exist 'matrix' or 'matrixTest' folder.");

            return -1;
        }
    }

    if (!folder) {
        perror("No folder found. Please check program arguments");

        return -1;
    }

    DIR *dir = opendir(folder);
    if (dir == NULL) {
        perror("Error opening folder");
        return -1;
    }

    printf("Folder path: %s\n", folder);

    FILE *result_file = csv_logger_init(NULL);
    if (!result_file) {
        printf("Error opening file result_file\n");
        return -1;
    }

#ifndef TEST
    struct dirent *entry;
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

        computeMatrix(result_file, filePath);

        free(filePath);
    }
#else
    printf("TESTING ON SINGLE FILE\n\n");

    // computeMatrix("../matrixTest/ns_example.mtx");
    // computeMatrix("../matrixTest/pat_example.mtx");
    // computeMatrix("../matrixTest/sym_example.mtx");


    // printf("\n ../matrixTest/mhda416.mtx\n");
    // computeMatrix("../matrixTest/mhda416.mtx");

    // printf("\n../matrixTest/cant.mtx\n");
    computeMatrix("../matrixTest/cant.mtx");
    // computeMatrix("../matrixTest/Cube_Coup_dt0.mtx");
    // computeMatrix("../matrix/ML_Laplace.mtx");
#endif

    csv_logger_close(result_file);
    closedir(dir);
    free(folder);

    printf("\n\033[32;7mEND\033[0m\n");

    return 0;
}
