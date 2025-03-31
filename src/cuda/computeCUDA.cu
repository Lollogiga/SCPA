#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/mtxStructs.h"

__global__ void cudaKernel(int *d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] += 1;
    }
}

int prova() {
    const int size = 10;
    int h_data[size] = {0}; // Array sulla CPU
    int *d_data;

    // Alloca memoria sulla GPU
    if (cudaMalloc((void **)&d_data, size * sizeof(int)) != cudaSuccess) {
        printf("Errore di allocazione memoria GPU!\n");
        return -1;
    }

    // Azzeramento della memoria sulla GPU (opzionale ma consigliato)
    cudaMemset(d_data, 0, size * sizeof(int));

    // Copia dati dalla CPU alla GPU
    if (cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Errore nella copia CPU -> GPU: %s\n", cudaGetErrorString(cudaGetLastError()));
        cudaFree(d_data);
        return -1;
    }

    // Configura i blocchi e i thread (modifica se necessario)
    int blockSize = 10; // Numero di thread per blocco
    int numBlocks = (size + blockSize - 1) / blockSize; // Numero di blocchi necessari

    // Lancio del kernel
    cudaKernel<<<numBlocks, blockSize>>>(d_data, size);

    // Verifica errori nel lancio del kernel
    cudaError_t errKernel = cudaGetLastError();
    if (errKernel != cudaSuccess) {
        printf("Errore nel lancio del kernel: %s\n", cudaGetErrorString(errKernel));
        cudaFree(d_data);
        return -1;
    }

    // Sincronizzazione del dispositivo
    cudaDeviceSynchronize();

    // Copia risultati dalla GPU alla CPU
    if (cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("Errore nella copia CPU -> GPU: %s\n", cudaGetErrorString(cudaGetLastError()));
        cudaFree(d_data);
        return -1;
    }

    // Stampa il risultato
    printf("Risultati CUDA: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Libera memoria GPU
    cudaFree(d_data);

    return 0;
}

extern "C" int computeCUDA(CSRMatrix *csrMatrix, HLLMatrix *hllMatrix, HLLMatrixAligned *hllMatrixAligned, int num_threads) {
    // int res = 0;

    // MatVal *vector = create_vector(csrMatrix->N);
    // if (vector == NULL) {
    //     perror("Error create_vector\n");
    //
    //     return -1;
    // }


    return 0;
}
