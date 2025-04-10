#ifndef CUDACSR_CUH
#define CUDACSR_CUH

#include "../mtxStructs.h"
#include "../result.h"

#define WARP_SIZE 32 // Impostato a 32 rispetto al server di dipartimento con sopra montata una Quadro RTX 5000
#define BLOCK_SIZE 32 // Il server ha a disposizione al massimo 1024 thread attivi contemporaneamente

__global__ void csr_cudaProduct_sol1(CSRMatrix *csr, MatVal *v, ResultVector *result);

__global__ void csr_cudaProduct_sol2_product(CSRMatrix *csr, MatVal *v_product, MatVal *x);
__global__ void csr_cudaProduct_sol2_reduce(CSRMatrix *csr, MatVal *v_product, ResultVector *result);

__global__ void csr_cudaProduct_sol3(CSRMatrix *csr, MatVal *v, ResultVector *result);

#endif //CUDACSR_CUH
