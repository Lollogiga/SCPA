#ifndef CHECKRESULTVECTO_H
#define CHECKRESULTVECTO_H

#include "./createVector.h"
#include "./performance.h"

#ifdef __cplusplus
extern "C" {
#endif

double checkResultVector(const ResultVector *serial_result, const ResultVector *parallel_result);
void analyzeErrorVector(const ResultVector *serial_result, const ResultVector *parallel_result, PerformanceResult *performance);

#ifdef __cplusplus
}
#endif

#endif //CHECKRESULTVECTO_H
