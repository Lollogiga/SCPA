#ifndef UTILSPRODUCT_H
#define UTILSPRODUCT_H

#include "./constants.h"
#include "./result.h"

#ifdef __cplusplus
extern "C" {
#endif

MatVal *create_vector(MatT len_vector);
ResultVector *create_result_vector(MatT len_vector);

#ifdef __cplusplus
}
#endif

#endif //UTILSPRODUCT_H
