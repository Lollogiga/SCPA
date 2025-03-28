#include <math.h>
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

    int current_weight = 0, thread_idx = 0;
    threadRanges[0].start = 0;
    for (int i = 0; i < csr->M; i++) {
        current_weight += row_weights[i];
        if (current_weight >= weight_per_thread && thread_idx < numThreads) {
            threadRanges[thread_idx].end = i;
            threadRanges[++thread_idx].start = i;
            current_weight = 0;
        }
    }
    threadRanges[numThreads - 1].end = csr->M;

    free(row_weights);

    return threadRanges;
}

ThreadDataRange *matrixBalanceHLL(HLLMatrix *hll, int numThreads) {
    if (!hll) {
        perror("hll_serialProduct: hll is NULL");
        return NULL;
    }

    if (hll->numBlocks == 0) {
        perror("matrixBalanceHLL: no blocks in the matrix");
        return NULL;
    }

    if (numThreads <= 0) {
        perror("matrixBalanceHLL: invalid number of threads");
        return NULL;
    }

    int *block_weights = calloc(hll->numBlocks, sizeof(int));
    if (!block_weights) {
        perror("matrixBalanceHLL: block_weights allocation error");
        return NULL;
    }

    int total_weight = 0;
    for (int i = 0; i < hll->numBlocks; i++) {
        ELLPACKMatrix *ell = hll->blocks[i];
        for (MatT j = 0; j < ell->M; j++) {
            for (MatT k = 0; k < ell->MAXNZ; k++) {
                if (ell->AS[j][k] != 0) {
                    block_weights[i]++;
                }
            }
        }
        total_weight += block_weights[i];
    }

    if (total_weight == 0) {
        free(block_weights);
        perror("matrixBalanceHLL: total_weight is zero, no work to distribute");
        return NULL;
    }

    int avg_weight_per_thread = (int)ceil((double)total_weight / numThreads);

    // Allocazione della struttura di assegnazione dei blocchi per thread
    ThreadDataRange *threadRanges = calloc(numThreads, sizeof(ThreadDataRange));
    if (!threadRanges) {
        perror("matrixBalanceHLL: threadRanges allocation error");
        free(block_weights);
        return NULL;
    }

    if (numThreads > hll->numBlocks) numThreads = hll->numBlocks;

    int current_weight = 0;
    int thread_idx = 0;
    threadRanges[thread_idx].start = 0;

    for (int i = 0; i < hll->numBlocks; i++) {
        current_weight += block_weights[i];

        if (current_weight >= avg_weight_per_thread && thread_idx < numThreads - 1) {
            threadRanges[thread_idx].end = i;  // Assegna l'ultimo blocco al thread corrente
            thread_idx++;

            // Controlla che non si esca dal numero massimo di thread
            if (thread_idx < numThreads) {
                threadRanges[thread_idx].start = i;
            }
            current_weight = 0;
        }
    }

    // Assicurati che l'ultimo thread copra gli ultimi blocchi rimanenti
    threadRanges[numThreads - 1].end = hll->numBlocks;

    free(block_weights);
    return threadRanges;
}

ThreadDataRange *matrixBalanceHLL_sol2(HLLMatrixAligned *hll, int numThreads) {
    if (!hll) {
        perror("hll_serialProduct: hll is NULL");
        return NULL;
    }

    if (hll->numBlocks == 0) {
        perror("matrixBalanceHLL: no blocks in the matrix");
        return NULL;
    }

    if (numThreads <= 0) {
        perror("matrixBalanceHLL: invalid number of threads");
        return NULL;
    }

    // Allocazione dei pesi per i blocchi
    int *block_weights = calloc(hll->numBlocks, sizeof(MatT));
    if (!block_weights) {
        perror("matrixBalanceHLL: block_weights allocation error");
        return NULL;
    }

    if (numThreads > hll->numBlocks) numThreads = hll->numBlocks;

    int total_weight = 0;

    // Iterazione sui blocchi HLL
    for (int i = 0; i < hll->numBlocks; i++) {
        ELLPACKMatrixAligned *ell = hll->blocks[i];  // Blocchi della matrice HLL

        // Iterazione sulle righe della matrice ELLPACK
        for (MatT j = 0; j < ell->M; j++) {
            // Iterazione sugli elementi non zero nella riga j
            for (MatT k = 0; k < ell->MAXNZ; k++) {
                if (ell->AS[j * ell->MAXNZ + k] != 0) {
                    block_weights[i]++;  // Incrementa il peso del blocco
                }
            }
        }

        total_weight += block_weights[i];  // Somma totale del peso dei blocchi
    }

    int avg_weight_per_thread = total_weight / numThreads;

    // Allocazione della struttura che gestisce le assegnazioni dei blocchi per thread
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
            threadRanges[++thread_idx].start = i;
            current_weight = 0;
        }
    }
    threadRanges[numThreads - 1].end = hll->numBlocks;

    free(block_weights);  // Dealloca il vettore dei pesi

    return threadRanges;
}
