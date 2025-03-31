#include "../include/flops.h"

double computeFlops(MatT NZ, double timer) {
    return 2 * NZ / (timer * 1e9); // Compute GFLOPS
}