#include <stdlib.h>

#include "../include/matrixBalance.h"
#include "../include/matrixPreProcessing.h"

ThreadDataRange *matrixBalanceCSR(CSRMatrix *csr, int numThreads) {
    if (!csr) {
        perror("matrixBalanceCSR: csr is NULL");
        return NULL;
    }

    int *row_weights = calloc(csr->M, sizeof(int));
    for (int i = 0; i < csr->M; i++) {
        row_weights[i] = csr->IRP[i + 1] - csr->IRP[i];
    }

    /** Total_weight = # of NZ */
    int total_weight = csr->NZ;

    int weight_per_thread = total_weight / numThreads;
    ThreadDataRange *threadRanges = calloc(numThreads, sizeof(ThreadDataRange));
    if (!threadRanges) {
        perror("matrixBalanceCSR: threadRanges allocation error");
        free(row_weights);
        return NULL;
    }

    //TODO: Io peso che thread_idx debba essere 0 e non 1.
    int current_weight = 0, thread_idx = 0;
    threadRanges[0].start = 0;
    for (int i = 0; i < csr->M; i++) {
        current_weight += row_weights[i];
        if (current_weight >= weight_per_thread && thread_idx < numThreads) {
            threadRanges[thread_idx].end = i;
            threadRanges[++thread_idx].start = i + 1;
            current_weight = 0;
        }
    }
    threadRanges[numThreads - 1].end = csr->M;

    free(row_weights);

    return threadRanges;
}

ThreadDataRange *matrixBalanceHLL(HLLMatrix *hll, int numThreads) {
    if(!hll) {
        perror("hll_serialProduct: hll is NULL");
        return NULL;
    }

    int *block_weights = calloc(hll->numBlocks, sizeof(MatT));
    if (!block_weights){
        perror("matrixBalanceHLL: block_weights allocation error");
        return NULL;
    }

    int total_weight = 0;
    for (int i = 0; i < hll->numBlocks; i++) {
        ELLPACKMatrix *ell = hll->blocks[i];
        for (MatT j = 0; j < ell->M; j++) {
            for (MatT k = 0; k < ell->MAXNZ; k++) {
                (ell->AS[j][k] != 0) ? block_weights[i]++ : 0;
            }
        }

        total_weight += block_weights[i];
    }

    int avg_weight_per_thread = total_weight / numThreads;

    // Allocazione della struttura di assegnazione dei blocchi per thread
    ThreadDataRange *threadRanges = malloc(sizeof(ThreadDataRange) * numThreads);
    if (!threadRanges) {
        perror("matrixBalanceHLL: threadRanges allocation error");
        free(block_weights);
        return NULL;
    }

    // Assegnazione dei blocchi in base ai pesi
    int current_weight = 0, thread_idx = 0;
    threadRanges[0].start = 0;
    for (int i = 0; i < hll->numBlocks; i++) {
        current_weight += block_weights[i];
        if (current_weight >= avg_weight_per_thread && thread_idx < numThreads - 1) {
            threadRanges[thread_idx].end = i;
            threadRanges[++thread_idx].start = i + 1;
            current_weight = 0;
        }
    }
    threadRanges[numThreads - 1].end = hll->numBlocks - 1;

    free(block_weights);

    return threadRanges;
}