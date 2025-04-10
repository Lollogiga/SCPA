#ifndef CUDAUTILS_CUH
#define CUDAUTILS_CUH

#include "../mtxStructs.h"
#include "../result.h"

#define CUDA_EVENT_CREATE(start, stop) \
cudaEvent_t start, stop; \
cudaEventCreate(&start); \
cudaEventCreate(&stop);

#define CUDA_EVENT_DESTROY(start, stop) \
cudaEventDestroy(start); \
cudaEventDestroy(stop);

#define CUDA_EVENT_START(start) \
cudaEventRecord(start);

#define CUDA_EVENT_STOP(stop) \
cudaDeviceSynchronize(); \
cudaEventRecord(stop); \
cudaEventSynchronize(stop);

// elapsedTime is counted into milliseconds
#define CUDA_EVENT_ELAPSED(start, stop, elapsedTime) \
cudaEventElapsedTime(&elapsedTime, start, stop); \
elapsedTime /= 1000;

CSRMatrix* uploadCSRToDevice(const CSRMatrix *h_csr);
ResultVector* uploadResultVectorToDevice(const ResultVector *h_vec);

int downloadResultVectorToHost(ResultVector *hostResultVector, const ResultVector *deviceResultVector);

int freeCSRDevice(CSRMatrix *d_csr_ptr);
int freeResultVectorFromDevice(ResultVector *d_result_vector);

#endif//CUDAUTILS_CUH