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

ResultVector* uploadResultVectorToDevice(const ResultVector *h_vec) {
    // 1. Crea struttura temporanea su host
    ResultVector d_vec;
    d_vec.len_vector = h_vec->len_vector;

    // 2. Alloca e copia l'array dinamico su device
    cudaMalloc((void**)&d_vec.val, h_vec->len_vector * sizeof(MatVal));
    cudaMemcpy(d_vec.val, h_vec->val, h_vec->len_vector * sizeof(MatVal), cudaMemcpyHostToDevice);

    // 3. Alloca la struttura ResultVector su device
    ResultVector *d_vec_ptr;
    cudaMalloc((void**)&d_vec_ptr, sizeof(ResultVector));

    // 4. Copia la struttura (con puntatore già device) sul device
    cudaMemcpy(d_vec_ptr, &d_vec, sizeof(ResultVector), cudaMemcpyHostToDevice);

    return d_vec_ptr;
}

void downloadResultVectorToHost(ResultVector *hostResultVector, const ResultVector *deviceResultVector) {
    cudaMemcpy(hostResultVector, deviceResultVector, sizeof(ResultVector), cudaMemcpyDeviceToHost);

    MatVal *device_val_ptr = hostResultVector->val;

    hostResultVector->val = (MatVal*)malloc(hostResultVector->len_vector * sizeof(MatVal));

    cudaMemcpy(hostResultVector->val, device_val_ptr, hostResultVector->len_vector * sizeof(MatVal), cudaMemcpyDeviceToHost);
}

void freeCSRDevice(CSRMatrix *d_csr_ptr) {
    CSRMatrix h_temp;
    cudaMemcpy(&h_temp, d_csr_ptr, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);

    cudaFree(h_temp.IRP);
    cudaFree(h_temp.JA);
    cudaFree(h_temp.AS);
    cudaFree(d_csr_ptr);
}

void freeResultVectorFromDevice(ResultVector *d_result_vector) {
    // 1. Copia la struttura dal device all'host per accedere ai campi
    ResultVector h_temp;
    cudaMemcpy(&h_temp, d_result_vector, sizeof(ResultVector), cudaMemcpyDeviceToHost);

    // 2. Libera il campo val (array device)
    cudaFree(h_temp.val);

    // 3. Libera la struttura vera e propria (device)
    cudaFree(d_result_vector);
}

