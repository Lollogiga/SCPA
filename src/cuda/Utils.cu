#include "../include/cuda/Utils.cuh"

CSRMatrix* uploadCSRToDevice(const CSRMatrix *h_csr) {
    // 1. Crea struttura temporanea su host
    CSRMatrix d_csr;
    d_csr.M = h_csr->M;
    d_csr.N = h_csr->N;
    d_csr.NZ = h_csr->NZ;

    // 2. Alloca array su device e copia i contenuti
    cudaMalloc((void**)&d_csr.IRP, (h_csr->M + 1) * sizeof(MatT));
    cudaMemcpy(d_csr.IRP, h_csr->IRP, (h_csr->M + 1) * sizeof(MatT), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_csr.JA, h_csr->NZ * sizeof(MatT));
    cudaMemcpy(d_csr.JA, h_csr->JA, h_csr->NZ * sizeof(MatT), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_csr.AS, h_csr->NZ * sizeof(MatVal));
    cudaMemcpy(d_csr.AS, h_csr->AS, h_csr->NZ * sizeof(MatVal), cudaMemcpyHostToDevice);

    // 3. Alloca CSRMatrix su device
    CSRMatrix *d_csr_ptr;
    cudaMalloc((void**)&d_csr_ptr, sizeof(CSRMatrix));

    // 4. Copia la struttura su device
    cudaMemcpy(d_csr_ptr, &d_csr, sizeof(CSRMatrix), cudaMemcpyHostToDevice);

    return d_csr_ptr;
}

void freeCSRDevice(CSRMatrix *d_csr_ptr) {
    CSRMatrix h_temp;
    cudaMemcpy(&h_temp, d_csr_ptr, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);

    cudaFree(h_temp.IRP);
    cudaFree(h_temp.JA);
    cudaFree(h_temp.AS);
    cudaFree(d_csr_ptr);
}