#include "../include/cuda/Utils.cuh"

#include <stdio.h>

CSRMatrix* uploadCSRToDevice(const CSRMatrix *h_csr) {
    cudaError_t err;

    CSRMatrix d_csr;
    d_csr.M = h_csr->M;
    d_csr.N = h_csr->N;
    d_csr.NZ = h_csr->NZ;

    err = cudaMalloc((void**)&d_csr.IRP, (h_csr->M + 1) * sizeof(MatT));
    if (err != cudaSuccess) {
        printf("\033[31muploadCSRToDevice - cudaMalloc IRP failed: %s\033[0m\n", cudaGetErrorString(err));
        return nullptr;
    }
    err = cudaMemcpy(d_csr.IRP, h_csr->IRP, (h_csr->M + 1) * sizeof(MatT), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("\033[31muploadCSRToDevice - cudaMemcpy IRP failed: %s\033[0m\n", cudaGetErrorString(err));

        cudaFree(d_csr.IRP);
        return nullptr;
    }

    err = cudaMalloc((void**)&d_csr.JA, h_csr->NZ * sizeof(MatT));
    if (err != cudaSuccess) {
        printf("\033[31muploadCSRToDevice - cudaMalloc JA failed: %s\033[0m\n", cudaGetErrorString(err));

        cudaFree(d_csr.IRP);
        return nullptr;
    }
    err = cudaMemcpy(d_csr.JA, h_csr->JA, h_csr->NZ * sizeof(MatT), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("\033[31muploadCSRToDevice - cudaMemcpy JA failed: %s\033[0m\n", cudaGetErrorString(err));

        cudaFree(d_csr.IRP);
        cudaFree(d_csr.JA);
        return nullptr;
    }

    err = cudaMalloc((void**)&d_csr.AS, h_csr->NZ * sizeof(MatVal));
    if (err != cudaSuccess) {
        printf("\033[31muploadCSRToDevice - cudaMalloc AS failed: %s\033[0m\n", cudaGetErrorString(err));

        cudaFree(d_csr.IRP);
        cudaFree(d_csr.JA);
        return nullptr;
    }
    err = cudaMemcpy(d_csr.AS, h_csr->AS, h_csr->NZ * sizeof(MatVal), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("\033[31muploadCSRToDevice - cudaMemcpy AS failed: %s\033[0m\n", cudaGetErrorString(err));

        cudaFree(d_csr.IRP);
        cudaFree(d_csr.JA);
        cudaFree(d_csr.AS);
        return nullptr;
    }

    CSRMatrix *d_csr_ptr;
    err = cudaMalloc((void**)&d_csr_ptr, sizeof(CSRMatrix));
    if (err != cudaSuccess) {
        printf("\033[31muploadCSRToDevice - cudaMalloc CSRMatrix failed: %s\033[0m\n", cudaGetErrorString(err));

        cudaFree(d_csr.IRP);
        cudaFree(d_csr.JA);
        cudaFree(d_csr.AS);
        return nullptr;
    }
    err = cudaMemcpy(d_csr_ptr, &d_csr, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("\033[31muploadCSRToDevice - cudaMemcpy CSRMatrix failed: %s\033[0m\n", cudaGetErrorString(err));

        cudaFree(d_csr.IRP);
        cudaFree(d_csr.JA);
        cudaFree(d_csr.AS);
        cudaFree(d_csr_ptr);
        return nullptr;
    }

    return d_csr_ptr;
}
int freeCSRDevice(CSRMatrix *d_csr_ptr) {
    cudaError_t err;

    CSRMatrix h_temp;
    err = cudaMemcpy(&h_temp, d_csr_ptr, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("\033[31mfreeCSRDevice - cudaMemcpy CSRMatrix failed: %s\033[0m\n", cudaGetErrorString(err));

        return -1;
    }

    cudaFree(h_temp.IRP);
    cudaFree(h_temp.JA);
    cudaFree(h_temp.AS);
    cudaFree(d_csr_ptr);

    return 0;
}

HLLMatrix* uploadHLLToDevice(const HLLMatrix* h_hll) {
    HLLMatrix* d_hll;
    cudaMalloc(&d_hll, sizeof(HLLMatrix));

    // Copia metadati base
    cudaMemcpy(d_hll, h_hll, sizeof(HLLMatrix), cudaMemcpyHostToDevice);

    // Alloca array di puntatori ai blocchi su device
    ELLPACKMatrix** d_blocks;
    cudaMalloc(&d_blocks, h_hll->numBlocks * sizeof(ELLPACKMatrix*));

    for (MatT i = 0; i < h_hll->numBlocks; i++) {
        ELLPACKMatrix* h_block = h_hll->blocks[i];
        ELLPACKMatrix* d_block;
        cudaMalloc(&d_block, sizeof(ELLPACKMatrix));

        // Copia campi non-puntatore
        cudaMemcpy(d_block, h_block, sizeof(ELLPACKMatrix), cudaMemcpyHostToDevice);

        // Gestione JA
        MatT** d_JA;
        cudaMalloc(&d_JA, h_block->M * sizeof(MatT*));
        for (MatT j = 0; j < h_block->M; j++) {
            MatT* d_row;
            cudaMalloc(&d_row, h_block->MAXNZ * sizeof(MatT));
            cudaMemcpy(d_row, h_block->JA[j], h_block->MAXNZ * sizeof(MatT), cudaMemcpyHostToDevice);
            cudaMemcpy(&d_JA[j], &d_row, sizeof(MatT*), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(&d_block->JA, &d_JA, sizeof(MatT**), cudaMemcpyHostToDevice);

        // Gestione AS (analogo a JA)
        MatVal** d_AS;
        cudaMalloc(&d_AS, h_block->M * sizeof(MatVal*));
        for (MatT j = 0; j < h_block->M; j++) {
            MatVal* d_row;
            cudaMalloc(&d_row, h_block->MAXNZ * sizeof(MatVal));
            cudaMemcpy(d_row, h_block->AS[j], h_block->MAXNZ * sizeof(MatVal), cudaMemcpyHostToDevice);
            cudaMemcpy(&d_AS[j], &d_row, sizeof(MatVal*), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(&d_block->AS, &d_AS, sizeof(MatVal**), cudaMemcpyHostToDevice);

        // Aggiorna array blocchi
        cudaMemcpy(&d_blocks[i], &d_block, sizeof(ELLPACKMatrix*), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(&d_hll->blocks, &d_blocks, sizeof(ELLPACKMatrix**), cudaMemcpyHostToDevice);
    return d_hll;
}

void freeHLLDevice(HLLMatrix* d_hll) {
    ELLPACKMatrix** d_blocks;
    cudaMemcpy(&d_blocks, &d_hll->blocks, sizeof(ELLPACKMatrix**), cudaMemcpyDeviceToHost);

    MatT numBlocks;
    cudaMemcpy(&numBlocks, &d_hll->numBlocks, sizeof(MatT), cudaMemcpyDeviceToHost);

    for (MatT i = 0; i < numBlocks; i++) {
        ELLPACKMatrix* d_block;
        cudaMemcpy(&d_block, &d_blocks[i], sizeof(ELLPACKMatrix*), cudaMemcpyDeviceToHost);

        MatT M;
        cudaMemcpy(&M, &d_block->M, sizeof(MatT), cudaMemcpyDeviceToHost);

        // Dealloca JA
        MatT** d_JA;
        cudaMemcpy(&d_JA, &d_block->JA, sizeof(MatT**), cudaMemcpyDeviceToHost);
        for (MatT j = 0; j < M; j++) {
            MatT* d_row;
            cudaMemcpy(&d_row, &d_JA[j], sizeof(MatT*), cudaMemcpyDeviceToHost);
            cudaFree(d_row);
        }
        cudaFree(d_JA);

        // Dealloca AS (analogo a JA)
        MatVal** d_AS;
        cudaMemcpy(&d_AS, &d_block->AS, sizeof(MatVal**), cudaMemcpyDeviceToHost);
        for (MatT j = 0; j < M; j++) {
            MatVal* d_row;
            cudaMemcpy(&d_row, &d_AS[j], sizeof(MatVal*), cudaMemcpyDeviceToHost);
            cudaFree(d_row);
        }
        cudaFree(d_AS);

        cudaFree(d_block);
    }

    cudaFree(d_blocks);
    cudaFree(d_hll);
}




ResultVector* uploadResultVectorToDevice(const ResultVector *h_vec) {
    cudaError_t err;

    ResultVector d_vec;
    d_vec.len_vector = h_vec->len_vector;

    err = cudaMalloc((void**)&d_vec.val, h_vec->len_vector * sizeof(MatVal));
    if (err != cudaSuccess) {
        printf("\033[31muploadResultVectorToDevice - cudaMalloc val failed: %s\033[0m\n", cudaGetErrorString(err));
        return nullptr;
    }
    err = cudaMemcpy(d_vec.val, h_vec->val, h_vec->len_vector * sizeof(MatVal), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("\033[31muploadResultVectorToDevice - cudaMemcpy val failed: %s\033[0m\n", cudaGetErrorString(err));

        cudaFree(d_vec.val);
        return nullptr;
    }

    ResultVector *d_vec_ptr;
    err = cudaMalloc((void**)&d_vec_ptr, sizeof(ResultVector));
    if (err != cudaSuccess) {
        printf("\033[31muploadResultVectorToDevice - cudaMalloc ResultVector failed: %s\033[0m\n", cudaGetErrorString(err));

        cudaFree(d_vec.val);
        return nullptr;
    }
    err = cudaMemcpy(d_vec_ptr, &d_vec, sizeof(ResultVector), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("\033[31muploadResultVectorToDevice - cudaMemcpy ResultVector failed: %s\033[0m\n", cudaGetErrorString(err));

        cudaFree(d_vec.val);
        cudaFree(d_vec_ptr);
        return nullptr;
    }

    return d_vec_ptr;
}



int freeResultVectorFromDevice(ResultVector *d_result_vector) {
    cudaError_t err;

    ResultVector h_temp;
    err = cudaMemcpy(&h_temp, d_result_vector, sizeof(ResultVector), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("\033[31mfreeResultVectorFromDevice - cudaMemcpy ResultVector failed: %s\033[0m\n", cudaGetErrorString(err));

        return -1;
    }

    cudaFree(h_temp.val);
    cudaFree(d_result_vector);

    return 0;
}

int downloadResultVectorToHost(ResultVector *hostResultVector, const ResultVector *deviceResultVector) {
    cudaError_t err;

    err = cudaMemcpy(hostResultVector, deviceResultVector, sizeof(ResultVector), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("\033[31mdownloadResultVectorToHost - cudaMemcpy ResultVector failed: %s\033[0m\n", cudaGetErrorString(err));

        printf("hostResultVector: %p, deviceResultVector: %p\n", hostResultVector, deviceResultVector);

        return -1;
    }

    MatVal *device_val_ptr = hostResultVector->val;

    hostResultVector->val = (MatVal*)malloc(hostResultVector->len_vector * sizeof(MatVal));
    if (hostResultVector->val == NULL) {
        printf("\033[31mdownloadResultVectorToHost - Memory allocation for val failed\033[0m\n");
        return -1;
    }

    err = cudaMemcpy(hostResultVector->val, device_val_ptr, hostResultVector->len_vector * sizeof(MatVal), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("\033[31mdownloadResultVectorToHost - cudaMemcpy val failed: %s\033[0m\n", cudaGetErrorString(err));

        return -1;
    }

    return 0;
}