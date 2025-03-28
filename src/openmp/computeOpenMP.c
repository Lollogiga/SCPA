//
// Created by buniy on 20/03/2025.
//

#include <omp.h>

#include "../include/matrixBalance.h"
#include "../include/matrixFree.h"
#include "../include/openmpCSR.h"
#include "../include/openmpHLL.h"
#include "../include/serialProduct.h"
#include "../include/flops.h"
#include "../include/checkResultVector.h"

ResultVector* csrProduct_Serial(CSRMatrix *csrMatrix, const MatVal *vector) {
    const MatT NZ = csrMatrix->NZ;

    ResultVector *res = NULL;
    double start = 0, end = 0;

    // Serial solution
    start = omp_get_wtime();
    res = csr_serialProduct(csrMatrix, vector);
    if (res == NULL) {
        perror("Error csr_SerialProduct\n");
        return NULL;
    }
    end = omp_get_wtime();
    printf("csr_serial: GFLOPS = %lf\n", computeFlops(NZ, end - start));

    return res;
}

int csrProduct_OpenMP(CSRMatrix *csrMatrix, MatVal *vector, const int num_threads, const ResultVector *serial_res) {
    const MatT NZ = csrMatrix->NZ;

    ResultVector *res = NULL;
    double start = 0, end = 0;

    // Csr solution 1
    start = omp_get_wtime();
    res = csr_openmpProduct_sol1(csrMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error csr_openmpProduct_sol1\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        perror("Error checkResultVector in Csr solution 1 \n");
        free_ResultVector(res);
        return -1;
    }
    free_ResultVector(res);
    printf("csr_openmp1: GFLOPS = %lf\n", computeFlops(NZ, end - start));


    // Csr solution 2
    start = omp_get_wtime();
    res = csr_openmpProduct_sol2(csrMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error csr_openmpProduct_sol2\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        perror("Error checkResultVector in Csr solution 2\n");
        free_ResultVector(res);
        return -1;
    }
    free_ResultVector(res);
    printf("csr_openmp2: GFLOPS = %lf\n", computeFlops(NZ, end - start));


    // Csr solution 3
    start = omp_get_wtime();
    res = csr_openmpProduct_sol3(csrMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error csr_openmpProduct_sol3\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        perror("Error checkResultVector in Csr solution 3\n");
        free_ResultVector(res);
        return -1;
    }
    free_ResultVector(res);
    printf("csr_openmp3: GFLOPS = %lf\n", computeFlops(NZ, end - start));


    // Csr solution 4
    start = omp_get_wtime();
    res = csr_openmpProduct_sol4(csrMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error csr_openmpProduct_sol4\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        perror("Error checkResultVector in Csr solution 4\n");
        free_ResultVector(res);
        return -1;
    }
    free_ResultVector(res);
    printf("csr_openmp4: GFLOPS = %lf\n", computeFlops(NZ, end - start));


    // Csr solution 5
    ThreadDataRange *tdr = matrixBalanceCSR(csrMatrix, num_threads);
    start = omp_get_wtime();
    res = csr_openmpProduct_sol5(csrMatrix, vector, num_threads, tdr);
    if (res == NULL) {
        perror("Error csr_openmpProduct_sol4\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        perror("Error checkResultVector in Csr solution 5\n");
        free_ResultVector(res);
        return -1;
    }
    free_ResultVector(res);
    printf("csr_openmp5: GFLOPS = %lf\n", computeFlops(NZ, end - start));


    return 0;
}

ResultVector* hllProduct_Serial(HLLMatrix *hllMatrix, MatVal *vector) {
    const MatT NZ = hllMatrix->NZ;

    ResultVector *hll_product = NULL;
    double start = 0, end = 0;

    // Serial solution
    start = omp_get_wtime();
    hll_product = hll_serialProduct(hllMatrix, vector);
    if (hll_product == NULL) {
        perror("Error hll_SerialProduct\n");
        return NULL;
    }
    end = omp_get_wtime();
    printf("hll_serial: GFLOPS = %lf\n", computeFlops(NZ, end - start));

    return hll_product;
}

int hllProduct_OpenMP(HLLMatrix *hllMatrix, MatVal *vector, int num_threads, ResultVector *serial_res) {
    const MatT NZ = hllMatrix->NZ;

    ResultVector *res = NULL;
    double start = 0, end = 0;

    // OpenMP solution 1
    start = omp_get_wtime();
    res = hll_openmpProduct_sol1(hllMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error hll_openmpProduct_sol1\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        perror("Error checkResultVector Hll solution 1\n");
        free_ResultVector(res);
        return -1;
    }
    free_ResultVector(res);
    printf("hll_openmp1: GFLOPS = %lf\n", computeFlops(NZ, end - start));

    // OpenMP solution 2
    start = omp_get_wtime();
    res = hll_openmpProduct_sol2(hllMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error hll_openmpProduct_sol2\n");
        return -1;
    }
    end = omp_get_wtime();
    for (int i = 0; i < res->len_vector; i++) {
        if (serial_res->val[i] != res->val[i]) printf("i: %d, [%f]-[%f]\n", i, serial_res->val[i], res->val[i]);
    }
    if (checkResultVector(serial_res,res) < 0) {
        perror("Error checkResultVector Hll solution 2\n");
        free_ResultVector(res);
        return -1;
    }
    free_ResultVector(res);
    printf("hll_openmp2: GFLOPS = %lf\n", computeFlops(NZ, end - start));

    // OpenMP solution 3: added some preprocessing
    ThreadDataRange *tdr = matrixBalanceHLL(hllMatrix, num_threads);
    start = omp_get_wtime();
    res = hll_openmpProduct_sol3(hllMatrix, vector, num_threads, tdr);
    if (res == NULL) {
        perror("Error hll_openmpProduct_sol2\n");
        return -1;
    }
    end = omp_get_wtime();
    printf("Matrix lenght: %d\n", hllMatrix->NZ);
    for (int i = 0; i < res->len_vector; i++) {
        if (serial_res->val[i] != res->val[i]) printf("i: %d, [%f]-[%f]\n", i, serial_res->val[i], res->val[i]);
    }
    double check = checkResultVector(serial_res,res);
    if (check < 0) {
        perror("Error checkResultVector Hll solution 3\n");
        free_ResultVector(res);
        return -1;
    }
    free_ResultVector(res);
    printf("hll_openmp3: GFLOPS = %lf\n", computeFlops(NZ, end - start));

    return 0;
}

ResultVector* hllAlignedProduct_Serial(HLLMatrixAligned *hllMatrix, MatVal *vector) {
    const MatT NZ = hllMatrix->NZ;

    ResultVector *hll_product = NULL;
    double start = 0, end = 0;

    // Serial solution
    start = omp_get_wtime();
    hll_product = hllAligned_serialProduct(hllMatrix, vector);
    if (hll_product == NULL) {
        perror("Error hll_SerialProduct\n");
        return NULL;
    }
    end = omp_get_wtime();
    printf("hllAligned_serial: GFLOPS: %lf\n", computeFlops(NZ, end - start));

    return hll_product;
}

int hllAlignedProduct_OpenMP(HLLMatrixAligned *hllMatrix, MatVal *vector, int num_threads, ResultVector *serial_res) {
    const MatT NZ = hllMatrix->NZ;

    ResultVector *hll_product = NULL;
    double start = 0, end = 0;

    //OpemMP solution:
    start = omp_get_wtime();
    hll_product = hllAligned_openmpProduct(hllMatrix, vector, num_threads);
    if (hll_product == NULL) {
        perror("Error hll_SerialProduct\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,hll_product) < 0) {
        perror("Error checkResultVector Hll_Aligned solution 1\n");
        free_ResultVector(hll_product);
        return -1;
    }
    free_ResultVector(hll_product);
    printf("hllAligned_openmpProduct: GFLOPS: %f\n", computeFlops(NZ, end - start));

    //OpemMP solution 2:
    start = omp_get_wtime();
    hll_product = hllAligned_openmpProduct_sol2(hllMatrix, vector, num_threads);
    if (hll_product == NULL) {
        perror("Error hll_SerialProduct\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,hll_product) < 0) {
        perror("Error checkResultVector HllAligned solution 3\n");
        free_ResultVector(hll_product);
        return -1;
    }
    free_ResultVector(hll_product);
    printf("hllAligned_openmpProduct_sol2: GFLOPS: %f\n", computeFlops(NZ, end - start));

    // OpenMP solution 3: added some preprocessing
    ThreadDataRange *tdr = matrixBalanceHLL_sol2(hllMatrix, num_threads);
    start = omp_get_wtime();
    hll_product = hllAligned_openmpProduct_sol3(hllMatrix, vector, num_threads, tdr);
    if (hll_product == NULL) {
        perror("Error hll_openmpProduct_sol2\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,hll_product) < 0) {
        perror("Error checkResultVector HllAligned solution 3\n");
        free_ResultVector(hll_product);
        return -1;
    }
    free_ResultVector(hll_product);
    //printf("hll_openmp3: Elapsed mean time = %lf\n", end - start);
    printf("hllAligned_openmpProduct_sol3: GFLOPS = %lf\n", computeFlops(NZ, end - start));

    return 0;
}

int computeOpenMP(CSRMatrix *csrMatrix, HLLMatrix *hllMatrix, HLLMatrixAligned *hllMatrixAligned, int num_threads) {
    MatVal *vector = create_vector(csrMatrix->N);
    if (vector == NULL) {
        perror("Error create_vector\n");

        return -1;
    }

    ResultVector *csr_product = csrProduct_Serial(csrMatrix, vector);
    csrProduct_OpenMP(csrMatrix, vector, num_threads, csr_product);
    free_ResultVector(csr_product);

    ResultVector *hll_product = hllProduct_Serial(hllMatrix, vector);
    hllProduct_OpenMP(hllMatrix, vector, num_threads, hll_product);
    free_ResultVector(hll_product);

    ResultVector *hllAligned_product = hllAlignedProduct_Serial(hllMatrixAligned, vector);
    hllAlignedProduct_OpenMP(hllMatrixAligned, vector, num_threads, hllAligned_product);
    free_ResultVector(hllAligned_product);

    return 0;
}

