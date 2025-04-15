#include "../include/performance.h"

#include <errno.h>
#include <sys/stat.h>

#define PERFORMANCE_CSV_FILENAME "./data/results.csv"
#define CSV_HEADER "Implementation,Format,Matrix,Function,NZ,Repetitions,AvgTime_ms,GFLOPS,Threads,BlockSize,WarpSize,BlocksPerGrid,HasError,MaxAbsError,MaxRelError,AvgAbsError,AvgRelError,ErrorL2\n"

static FILE* fp;

FILE* csv_logger_init(const char* filename) {
    if (!filename) filename = PERFORMANCE_CSV_FILENAME;

    if (mkdir("data", 0755) != 0 && errno != EEXIST) {
        perror("Error creating directory");
        return NULL;
    }

    printf("Opening file: %s\n", filename);

    fp = fopen(filename, "w+");
    if (!fp) {
        perror("Error opening CSV file");
        return NULL;
    }

    fprintf(fp, CSV_HEADER);
    return fp;
}

void csv_logger_write(const PerformanceResult* result) {
    if (!fp || !result) return;

    if (result->has_error) {
        fprintf(
            // TODO capire la granularità che ci interessa avere sul file csv
            // fp, "%s,%s,%s,%s,%d,%d,%.6f,%.2f,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            fp, "%s,%s,%s,%s,%d,%d,%f,%f,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f\n",
            result->implementation,         // Tipo di implementazione ("OpenMP" o "CUDA")
            result->format,                 // Formato della matrice ("CSR", "HLL", etc.)
            result->matrix_name,            // Nome della matrice (ad esempio "thermal1")
            result->curr_func,              // Nome dalla funzione chiamata (ad esempio "csr_openmpProduct_sol1")
            result->NZ,                     // Numero di non zero (NZ)
            result->repetitions,            // Numero di ripetizioni del test
            result->avg_time_ms,            // Tempo medio in millisecondi
            result->gflops,                 // Performance in GFLOPS
            result->threads,                // Numero di thread per OpenMP, 0 per CUDA
            result->block_size,             // Numero di thread per blocco in CUDA
            result->warp_size,              // Warp size di CUDA
            result->blocks_per_grid,        // Numero di blocchi per griglia in CUDA
            result->has_error,              // 1 se c'è stato un errore, 0 altrimenti
            result->max_abs_error,          // Massimo errore assoluto
            result->max_rel_error,          // Massimo errore relativo
            result->avg_abs_error,          // Errore medio assoluto
            result->avg_rel_error,          // Errore medio relativo
            result->error_L2                // Errore L2 (norma euclidea)
        );
    } else {
        fprintf(
            // TODO capire la granularità che ci interessa avere sul file csv
            // fp, "%s,%s,%s,%s,%d,%d,%.6f,%.2f,%d,%d,%d,%d\n",
            fp, "%s,%s,%s,%s,%d,%d,%f,%f,%d,%d,%d,%d\n",
            result->implementation,         // Tipo di implementazione ("OpenMP" o "CUDA")
            result->format,                 // Formato della matrice ("CSR", "HLL", etc.)
            result->matrix_name,            // Nome della matrice
            result->curr_func,              // Nome dalla funzione chiamata (ad esempio "csr_openmpProduct_sol1")
            result->NZ,                     // Numero di non zero (NZ)
            result->repetitions,            // Numero di ripetizioni del test
            result->avg_time_ms,            // Tempo medio in millisecondi
            result->gflops,                 // Performance in GFLOPS
            result->threads,                // Numero di thread per OpenMP, 0 per CUDA
            result->block_size,             // Numero di thread per blocco in CUDA
            result->warp_size,              // Warp size di CUDA
            result->blocks_per_grid         // Numero di blocchi per griglia in CUDA
        );
    }
}

void csv_logger_close() {
    if (fp) fclose(fp);
}