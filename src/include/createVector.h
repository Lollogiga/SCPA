#ifndef UTILSPRODUCT_H
#define UTILSPRODUCT_H

#include "./constants.h"
#include "./result.h"

#ifdef __cplusplus
extern "C" {
#endif

MatVal *create_vector(MatT len_vector);
void free_vector(MatVal *vector);

ResultVector *create_result_vector(MatT len_vector);
void free_result_vector(ResultVector *result);

#ifdef __cplusplus
}
#endif

#endif //UTILSPRODUCT_H
