#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "../include/computeCUDA.cuh"

// __global__ void cudaKernel(int *d_data, int N) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < N) {
//         d_data[idx] += 1;
//     }
// }

extern "C" int computeCUDA() {
    // const int size = 10;
    // int h_data[size] = {0}; // Array sulla CPU
    // int *d_data;
    //
    // // Alloca memoria sulla GPU
    // if (cudaMalloc((void **)&d_data, size * sizeof(int)) != cudaSuccess) {
    //     printf("Errore di allocazione memoria GPU!\n");
    //     return;
    // }
    //
    // // Azzeramento della memoria sulla GPU (opzionale ma consigliato)
    // cudaMemset(d_data, 0, size * sizeof(int));
    //
    // // Copia dati dalla CPU alla GPU
    // if (cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
    //     printf("Errore nella copia CPU -> GPU!\n");
    //     cudaFree(d_data);
    //     return;
    // }
    //
    // // Lancio del kernel
    // cudaKernel<<<1, 10>>>(d_data, size);
    //
    // // Controllo errori CUDA dopo il lancio del kernel
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("Errore nel lancio del kernel: %s\n", cudaGetErrorString(err));
    //     cudaFree(d_data);
    //     return;
    // }
    //
    // // Copia risultati dalla GPU alla CPU
    // if (cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
    //     printf("Errore nella copia GPU -> CPU!\n");
    //     cudaFree(d_data);
    //     return;
    // }
    //
    // // Stampa il risultato
    // printf("Risultati CUDA: ");
    // for (int i = 0; i < size; i++) {
    //     printf("%d ", h_data[i]);
    // }
    // printf("\n");
    //
    // // Libera memoria GPU
    // cudaFree(d_data);

    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("Errore nel recuperare il numero di dispositivi CUDA: %s\n", cudaGetErrorString(err));
        return -1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        struct cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, i);
        if (err != cudaSuccess) {
            printf("Errore nel recuperare le proprietà del dispositivo %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", i, deviceProp.name);
        printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    }

    return 0;
}
