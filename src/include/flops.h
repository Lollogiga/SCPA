#ifndef FLOPS_H
#define FLOPS_H

#include "../include/constants.h"

#ifdef __cplusplus
extern "C" {
#endif

    double computeFlops(MatT NZ, double timer);

#ifdef __cplusplus
}
#endif

#endif //FLOPS_H
