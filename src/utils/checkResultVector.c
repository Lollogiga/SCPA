#include <math.h>
#include <stdio.h>

#include "../include/checkResultVector.h"
#include "../include/constants.h"

// Funzione per calcolare il massimo valore assoluto tra due numeri
double maxAbs(double a, double b) {
    return (fabs(a) > fabs(b)) ? fabs(a) : fabs(b);
}

/**
 * @brief Confronta due ResultVector e calcola l'errore relativo medio.
 * @param serial_result Il risultato seriale.
 * @param parallel_result Il risultato parallelo.
 * @return Errore relativo medio se il check è passato, valore negativo se fallisce.
 *
 * Valori di ritorno:
 * - `-1.0`: Errore critico - i vettori hanno lunghezze diverse.
 * - `-2.0`: Errore - i risultati non coincidono entro la tolleranza.
 * - `0.0`: I risultati coincidono perfettamente.
 * - `> 0.0`: Media delle differenze relative tra i valori divergenti.
 */
double checkResultVector(const ResultVector *serial_result, const ResultVector *parallel_result) {
    if (serial_result->len_vector != parallel_result->len_vector) {
        return -1.0;  // Errore critico: lunghezze diverse
    }

    double totalRelativeDiff = 0.0;
    int count = 0;
    int errors_found = 0;

    for (MatT i = 0; i < serial_result->len_vector; i++) {
        double maxAbsValue = maxAbs(serial_result->val[i], parallel_result->val[i]);
        if (maxAbsValue < REL_TOLERANCE) {
            maxAbsValue = REL_TOLERANCE;
        }

        double currentDiff = fabs(serial_result->val[i] - parallel_result->val[i]);
        double relativeDiff = (currentDiff <= ABS_TOLERANCE) ? 0.0 : currentDiff / maxAbsValue;

        // DEBUG, TODO: REMOVE
        if (i == 62432 && 0) {
            printf("maxAbsValue: %f\n", maxAbsValue);
            printf("currentDiff: %f\n", currentDiff);
            printf("relativeDiff: %f\n", relativeDiff);
        }

        if (relativeDiff > REL_TOLERANCE) {
            errors_found = 1;
        }

        if (relativeDiff > 0.0) {
            totalRelativeDiff += relativeDiff;
            count++;
        }
    }

    if (errors_found) {
        return -2.0;  // Errore: i risultati non coincidono entro la tolleranza
    }

    return (count > 0) ? totalRelativeDiff / count : 0.0;
}

void analyzeErrorVector(const ResultVector *serial_result, const ResultVector *parallel_result, PerformanceResult *performance) {
    if (serial_result->len_vector != parallel_result->len_vector) {
        printf("Errore: vettori di lunghezza diversa\n");
        return;
    }

    double totalRelativeDiff = 0.0;
    double totalAbsDiff = 0.0;
    double maxRelativeDiff = 0.0;
    double maxAbsDiff = 0.0;
    double squaredDiffSum = 0.0;

    MatT len = serial_result->len_vector;

    for (MatT i = 0; i < len; i++) {
        double val_serial = serial_result->val[i];
        double val_parallel = parallel_result->val[i];
        double absDiff = fabs(val_serial - val_parallel);

        double maxAbsVal = maxAbs(val_serial, val_parallel);
        if (maxAbsVal < REL_TOLERANCE) {
            maxAbsVal = REL_TOLERANCE;
        }

        double relDiff = absDiff / maxAbsVal;

        totalAbsDiff += absDiff;
        totalRelativeDiff += relDiff;
        squaredDiffSum += absDiff * absDiff;

        if (absDiff > maxAbsDiff) {
            maxAbsDiff = absDiff;
        }
        if (relDiff > maxRelativeDiff) {
            maxRelativeDiff = relDiff;
        }
    }

    performance->has_error = 1;
    performance->max_abs_error = maxAbsDiff;
    performance->max_rel_error = maxRelativeDiff;
    performance->avg_abs_error = totalAbsDiff / len;
    performance->avg_rel_error = totalRelativeDiff / len;
    performance->error_L2 = sqrt(squaredDiffSum);

    // printf("\n=== Analisi degli errori ===\n");
    // printf("Errore assoluto massimo   : %.12e\n", maxAbsDiff);
    // printf("Errore relativo massimo   : %.12e\n", maxRelativeDiff);
    // printf("Errore medio assoluto     : %.12e\n", totalAbsDiff / len);
    // printf("Errore medio relativo     : %.12e\n", totalRelativeDiff / len);
    // printf("Errore L2 (norma euclidea): %.12e\n", sqrt(squaredDiffSum));
    // printf("===========================\n\n");
}