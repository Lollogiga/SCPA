#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/checkResultVector.h"
#include "../include/createVector.h"
#include "../include/flops.h"
#include "../include/mtxBalance.h"
#include "../include/mtxFree.h"
#include "../include/openmp/computeOpenMP.h"
#include "../include/openmp/CSR.h"
#include "../include/openmp/HLL.h"
#include "../include/openmp/Serial.h"
#include "../include/performance.h"

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

int csrProduct_OpenMP(CSRMatrix *csrMatrix, MatVal *vector, const int num_threads, const ResultVector *serial_res, PerformanceResult *performance) {
    const MatT NZ = csrMatrix->NZ;

    ResultVector *res = NULL;
    INIT_BENCHMARK_OPENMP(start, end, cumulative);

    // Csr solution 1
    BEGIN_BENCHMARK_OPENMP(performance, "csr_openmpProduct_sol1")
    res = csr_openmpProduct_sol1(csrMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error csr_openmpProduct_sol1\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        // perror("Error checkResultVector in Csr solution 1 \n");

        analyzeErrorVector(serial_res, res, performance);
    }
    free_ResultVector(res);
    END_BENCHMARK_OPENMP(performance, start, end, cumulative)
    printf("csr_openmp1: GFLOPS = %lf\n", computeFlops(NZ, performance->avg_time_ms));

    // Csr solution 2
    BEGIN_BENCHMARK_OPENMP(performance, "csr_openmpProduct_sol2")
    start = omp_get_wtime();
    res = csr_openmpProduct_sol2(csrMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error csr_openmpProduct_sol2\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        // perror("Error checkResultVector in Csr solution 2\n");

        analyzeErrorVector(serial_res, res, performance);
    }
    free_ResultVector(res);
    END_BENCHMARK_OPENMP(performance, start, end, cumulative)
    printf("csr_openmp2: GFLOPS = %lf\n", computeFlops(NZ, performance->avg_time_ms));


    // Csr solution 3
    BEGIN_BENCHMARK_OPENMP(performance, "csr_openmpProduct_sol3")
    start = omp_get_wtime();
    res = csr_openmpProduct_sol3(csrMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error csr_openmpProduct_sol3\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        // perror("Error checkResultVector in Csr solution 3\n");

        analyzeErrorVector(serial_res, res, performance);
    }
    free_ResultVector(res);
    END_BENCHMARK_OPENMP(performance, start, end, cumulative)
    printf("csr_openmp3: GFLOPS = %lf\n", computeFlops(NZ, end - start));


    // Csr solution 4
    BEGIN_BENCHMARK_OPENMP(performance, "csr_openmpProduct_sol4")
    start = omp_get_wtime();
    res = csr_openmpProduct_sol4(csrMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error csr_openmpProduct_sol4\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        // perror("Error checkResultVector in Csr solution 4\n");

        analyzeErrorVector(serial_res, res, performance);;
    }
    free_ResultVector(res);
    END_BENCHMARK_OPENMP(performance, start, end, cumulative)
    printf("csr_openmp4: GFLOPS = %lf\n", computeFlops(NZ, end - start));


    // Csr solution 5
    ThreadDataRange *tdr = matrixBalanceCSR(csrMatrix, num_threads);
    BEGIN_BENCHMARK_OPENMP(performance, "csr_openmpProduct_sol5")
    start = omp_get_wtime();
    res = csr_openmpProduct_sol5(csrMatrix, vector, num_threads, tdr);
    if (res == NULL) {
        perror("Error csr_openmpProduct_sol4\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        // perror("Error checkResultVector in Csr solution 5\n");

        analyzeErrorVector(serial_res, res, performance);
    }
    free_ResultVector(res);
    END_BENCHMARK_OPENMP(performance, start, end, cumulative)
    free(tdr);
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

int hllProduct_OpenMP(HLLMatrix *hllMatrix, MatVal *vector, int num_threads, ResultVector *serial_res, PerformanceResult *performance) {
    const MatT NZ = hllMatrix->NZ;

    ResultVector *res = NULL;
    INIT_BENCHMARK_OPENMP(start, end, cumulative);

    // OpenMP solution 1
    BEGIN_BENCHMARK_OPENMP(performance, "hll_openmpProduct_sol1")
    start = omp_get_wtime();
    res = hll_openmpProduct_sol1(hllMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error hll_openmpProduct_sol1\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        // perror("Error checkResultVector Hll solution 1\n");

        analyzeErrorVector(serial_res, res, performance);
    }
    free_ResultVector(res);
    END_BENCHMARK_OPENMP(performance, start, end, cumulative)
    printf("hll_openmp1: GFLOPS = %lf\n", computeFlops(NZ, end - start));

    // OpenMP solution 2
    BEGIN_BENCHMARK_OPENMP(performance, "hll_openmpProduct_sol2")
    start = omp_get_wtime();
    res = hll_openmpProduct_sol2(hllMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error hll_openmpProduct_sol2\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        // perror("Error checkResultVector Hll solution 2\n");

        analyzeErrorVector(serial_res, res, performance);
    }
    free_ResultVector(res);
    END_BENCHMARK_OPENMP(performance, start, end, cumulative)
    printf("hll_openmp2: GFLOPS = %lf\n", computeFlops(NZ, end - start));

    // OpenMP solution 3: added some preprocessing
    ThreadDataRange *tdr = matrixBalanceHLL(hllMatrix, num_threads);
    BEGIN_BENCHMARK_OPENMP(performance, "hll_openmpProduct_sol3")
    start = omp_get_wtime();
    res = hll_openmpProduct_sol3(hllMatrix, vector, num_threads, tdr);
    if (res == NULL) {
        perror("Error hll_openmpProduct_sol3\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        // perror("Error checkResultVector Hll solution 3\n");

        analyzeErrorVector(serial_res, res, performance);
    }
    free_ResultVector(res);
    END_BENCHMARK_OPENMP(performance, start, end, cumulative)
    free(tdr);
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

int hllAlignedProduct_OpenMP(HLLMatrixAligned *hllMatrix, MatVal *vector, int num_threads, ResultVector *serial_res, PerformanceResult *performance) {
    const MatT NZ = hllMatrix->NZ;

    ResultVector *res = NULL;
    INIT_BENCHMARK_OPENMP(start, end, cumulative);

    //OpemMP solution:
    BEGIN_BENCHMARK_OPENMP(performance, "hll_openmpProduct_sol3")
    start = omp_get_wtime();
    res = hllAligned_openmpProduct(hllMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error hll_SerialProduct\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        // perror("Error checkResultVector Hll_Aligned solution 1\n");

        analyzeErrorVector(serial_res, res, performance);
    }
    free_ResultVector(res);
    END_BENCHMARK_OPENMP(performance, start, end, cumulative)
    printf("hllAligned_openmp1: GFLOPS: %f\n", computeFlops(NZ, end - start));

    //OpemMP solution 2:
    BEGIN_BENCHMARK_OPENMP(performance, "hll_openmpProduct_sol3")
    start = omp_get_wtime();
    res = hllAligned_openmpProduct_sol2(hllMatrix, vector, num_threads);
    if (res == NULL) {
        perror("Error hll_SerialProduct\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        // perror("Error checkResultVector HllAligned solution 2\n");

        analyzeErrorVector(serial_res, res, performance);
    }
    free_ResultVector(res);
    END_BENCHMARK_OPENMP(performance, start, end, cumulative)
    printf("hllAligned_openmp2: GFLOPS: %f\n", computeFlops(NZ, end - start));

    // OpenMP solution 3: added some preprocessing
    ThreadDataRange *tdr = matrixBalanceHLL_sol2(hllMatrix, num_threads);
    BEGIN_BENCHMARK_OPENMP(performance, "hll_openmpProduct_sol3")
    start = omp_get_wtime();
    res = hllAligned_openmpProduct_sol3(hllMatrix, vector, num_threads, tdr);
    if (res == NULL) {
        perror("Error hll_openmpProduct_sol2\n");
        return -1;
    }
    end = omp_get_wtime();
    if (checkResultVector(serial_res,res) < 0) {
        // perror("Error checkResultVector HllAligned solution 3\n");

        analyzeErrorVector(serial_res, res, performance);
    }
    free_ResultVector(res);
    END_BENCHMARK_OPENMP(performance, start, end, cumulative)
    free(tdr);
    printf("hllAligned_openmp3: GFLOPS = %lf\n", computeFlops(NZ, end - start));

    return 0;
}

int computeOpenMP(CSRMatrix *csrMatrix, HLLMatrix *hllMatrix, HLLMatrixAligned *hllMatrixAligned, int num_threads, PerformanceResult *performance) {
    int res = 0;

    MatVal *vector = create_vector(csrMatrix->N);
    if (vector == NULL) {
        perror("Error create_vector\n");

        return -1;
    }

    // CSR format
    strcpy(performance->format, "CSR");

    ResultVector *csr_product = csrProduct_Serial(csrMatrix, vector);
    res = csrProduct_OpenMP(csrMatrix, vector, num_threads, csr_product, performance);
    free_ResultVector(csr_product);
    if (res) return -1;

    if (num_threads > hllMatrix->numBlocks) {
        num_threads = hllMatrix->numBlocks;
        printf("\n\033[32;7mNumber of threads exceeds the number of hll blocks\033[0m\n");
        printf("\n\033[32;7m# of threads per hll:\033[0;32m %d\033[0m\n", num_threads);
    }

    // HLL format
    strcpy(performance->format, "HLL");

    ResultVector *hll_product = hllProduct_Serial(hllMatrix, vector);
    res = hllProduct_OpenMP(hllMatrix, vector, num_threads, hll_product, performance);
    free_ResultVector(hll_product);
    if (res) return -1;

    // HLLAlign format
    strcpy(performance->format, "HLLAlign");

    ResultVector *hllAligned_product = hllAlignedProduct_Serial(hllMatrixAligned, vector);
    res = hllAlignedProduct_OpenMP(hllMatrixAligned, vector, num_threads, hllAligned_product, performance);
    free_ResultVector(hllAligned_product);
    if (res) return -1;

    return 0;
}
