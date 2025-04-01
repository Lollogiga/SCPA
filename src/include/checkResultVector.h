#ifndef CHECKRESULTVECTO_H
#define CHECKRESULTVECTO_H

#include "./createVector.h"

#ifdef __cplusplus
extern "C" {
#endif

double checkResultVector(const ResultVector *serial_result, const ResultVector *parallel_result);

#ifdef __cplusplus
}
#endif

#endif //CHECKRESULTVECTO_H
