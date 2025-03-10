#include <dirent.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "./include/fileUtils.h"
#include "./include/matrixBalance.h"
#include "./include/matrixPreProcessing.h"
#include "./include/matricFree.h"
#include "./include/matrixPrint.h"
#include "./include/serialProduct.h"
#include "./include/createVectorUtil.h"
#include "./include/openmpCSR.h"
#include "./include/openmpHLL.h"

int csr_product(CSRMatrix *matrix, MatVal *vector) {
    ResultVector *csr_product = NULL;
    double start =0, end = 0;

    // Serial solution
    start = omp_get_wtime();
    csr_product = csr_serialProduct(matrix, vector);
    if (csr_product == NULL) {
        perror("Error csr_SerialProduct\n");
        return -1;
    }
    end = omp_get_wtime();
    free_ResultVector(csr_product);
    printf("csr_serial: Elapsed mean time = %lf\n", end - start);

    // OpenMP solution 1
    start = omp_get_wtime();
    csr_product = csr_openmpProduct_sol1(matrix, vector);
    if (csr_product == NULL) {
        perror("Error csr_openmpProduct_sol1\n");
        return -1;
    }
    end = omp_get_wtime();
    free_ResultVector(csr_product);
    printf("csr_openmp1: Elapsed mean time = %lf\n", end - start);

    // OpenMP solution 2
    start = omp_get_wtime();
    csr_product = csr_openmpProduct_sol2(matrix, vector);
    if (csr_product == NULL) {
        perror("Error csr_openmpProduct_sol2\n");
        return -1;
    }
    end = omp_get_wtime();
    free_ResultVector(csr_product);
    printf("csr_openmp2: Elapsed mean time = %lf\n", end - start);

    // OpenMP solution 3
    start = omp_get_wtime();
    csr_product = csr_openmpProduct_sol3(matrix, vector);
    if (csr_product == NULL) {
        perror("Error csr_openmpProduct_sol3\n");
        return -1;
    }
    end = omp_get_wtime();
    free_ResultVector(csr_product);
    printf("csr_openmp3: Elapsed mean time = %lf\n", end - start);

    // OpenMP solution 4
    start = omp_get_wtime();
    csr_product = csr_openmpProduct_sol4(matrix, vector);
    if (csr_product == NULL) {
        perror("Error csr_openmpProduct_sol4\n");
        return -1;
    }
    end = omp_get_wtime();
    free_ResultVector(csr_product);
    printf("csr_openmp4: Elapsed mean time = %lf\n", end - start);

    // OpenMP solution 5
    ThreadDataRange *tdr = matrixBalanceCSR(matrix, 2);
    start = omp_get_wtime();
    csr_product = csr_openmpProduct_sol5(matrix, vector, 2, tdr);
    if (csr_product == NULL) {
        perror("Error csr_openmpProduct_sol4\n");
        return -1;
    }
    end = omp_get_wtime();
    free_ResultVector(csr_product);
    printf("csr_openmp5: Elapsed mean time = %lf\n", end - start);

    return 0;
}

int hll_product(HLLMatrix *matrix, MatVal *vector) {
    ResultVector *hll_product = NULL;
    double start =0, end = 0;

    // Serial solution
    start = omp_get_wtime();
    hll_product = hll_serialProduct(matrix, vector);
    if (hll_product == NULL) {
        perror("Error csr_SerialProduct\n");
        return -1;
    }
    end = omp_get_wtime();
    free_ResultVector(hll_product);
    printf("hll_serial: Elapsed mean time = %lf\n", end - start);

    // OpenMP solution 1
    start = omp_get_wtime();
    hll_product = hll_openmpProduct_sol1(matrix, vector);
    if (hll_product == NULL) {
        perror("Error hll_openmpProduct_sol1\n");
        return -1;
    }
    end = omp_get_wtime();
    free_ResultVector(hll_product);
    printf("hll_openmp1: Elapsed mean time = %lf\n", end - start);

    // OpenMP solution 2
    start = omp_get_wtime();
    hll_product = hll_openmpProduct_sol2(matrix, vector);
    if (hll_product == NULL) {
        perror("Error csr_openmpProduct_sol2\n");
        return -1;
    }
    end = omp_get_wtime();
    free_ResultVector(hll_product);
    printf("hll_openmp2: Elapsed mean time = %lf\n", end - start);

    // OpenMP solution 3: added some preprocessing
    ThreadDataRange *tdr = matrixBalanceHLL(matrix, 2);
    start = omp_get_wtime();
    hll_product = hll_openmpProduct_sol3(matrix, vector, 2, tdr);
    if (hll_product == NULL) {
        perror("Error csr_openmpProduct_sol2\n");
        return -1;
    }
    end = omp_get_wtime();
    free_ResultVector(hll_product);
    printf("hll_openmp3: Elapsed mean time = %lf\n", end - start);

    return 0;
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

    ELLPACKMatrix *subEllpackMatrix = convert_to_ELLPACK_parametrized(csrMatrix, 0, HACK_SIZE);
    if (subEllpackMatrix == NULL) {
        perror("Error convert_to_ELLPACK_parametrized\n");
    }

    // Convert to HLL format:
    HLLMatrix *hllMatrix = convert_to_HLL(csrMatrix, HACK_SIZE);;
    if (hllMatrix == NULL) {
        perror("Error convert_to_HLL\n");
    }

    MatVal *csr_vector = create_vector(csrMatrix->N);
    if (csr_vector == NULL) {
        perror("Error create_vector\n");
    }

    csr_product(csrMatrix, csr_vector);

    MatVal *hll_vector = create_vector(ellpackMatrix->N);
    if (hll_vector == NULL) {
        perror("Error create_vector\n");
    }

    hll_product(hllMatrix, hll_vector);

    ResultVector *hll_product = hll_serialProduct(hllMatrix, hll_vector);
    if (hll_product == NULL) {
        perror("Error hll_serialProduct\n");
    }

    /* Print for debugging:*/
    // printf("\nMatrix: \n");
    // print_matrix_data_verbose(rawMatrixData,false);
    //
    // printf("\nCSR Matrix: \n");
    // print_csr_matrix_verbose(csrMatrix, false);
    //
    // printf("\nELLPACK Matrix: \n");
    // print_ellpack_matrix_verbose(ellpackMatrix, false);
    //
    // printf("\nELLPACK Matrix reduced: \n");
    // print_ellpack_matrix_verbose(subEllpackMatrix, false);
    //
    // printf("\nResult csr vector: \n");
    // print_result_vector(csr_product);
    //
    // printf("\nResult hll vector: \n");
    // print_result_vector(hll_product);

    //Free memory:
    free_MatrixData(rawMatrixData);

    free_CSRMatrix(csrMatrix);
    free_ELLPACKMatrix(ellpackMatrix);
    free_ELLPACKMatrix(subEllpackMatrix);
    free_HLLMatrix(hllMatrix);

    //free_ResultVector(csr_product);
    free_ResultVector(hll_product);

    free(csr_vector);
    free(hll_vector);

    if (f && f != stdin) fclose(f);

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

    DIR *dir = opendir(folder);
    if (dir == NULL) {
        perror("Error opening folder");
        return -1;
    }

    printf("Folder path: %s\n", folder);

#ifndef TEST_SINGLE_FILE
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
#else
    printf("TEST_SINGcsrLE_FILE else\n\n");

    computeMatrixFile("../matrixTest/ns_example.mtx");
    computeMatrixFile("../matrixTest/cant.mtx");
    // computeMatrixFile("../matrix/Cube_Coup_dt0.mtx");
    // computeMatrixFile("../matrix/ML_Laplace.mtx");
#endif

    closedir(dir);
    free(folder);

    return 0;
}



