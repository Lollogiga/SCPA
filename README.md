# SCPA Project

## Things to do:

> - [x] Scaricare le matrici di test

> - [x] Gestire le matrici base
> - [x] Gestire le matrici simmetriche
> - [x] Gestire le matrici che presentano pattern

> - [x] Convertire le matrici in formato ```CSR```
> - [x] Convertire le matrici in formato ```ELLPACK```
> - [x] Convertire le matrici in formato ```HLL```

> - [x] Realizzare il prodotto _**matrice**_ x _**vettore**_ nei seguenti formati:
>   - [x] Prodotto in ```Seriale```
>   - [x] Prodotto in ```OpenMP```
>   - [x] Prodotto in ```CUDA```

> - [ ] Valutare le performance

## Execute memory analyses with valgrind

Per prima cosa è necessario installare valgrind sul sistema operativo (sotto sono elencati i comandi per un sistema Ubuntu):
```sh
  sudo apt update
  sudo apt install valgrind
```

Aggiungere al file `CMake` utilizzato per la compilazione i seguenti flag:
```c
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
```
- `-g` aggiunge i simboli di debug
- `-O0` disabilita le ottimizzazioni per una migliore analisi di Valgrind

Dopodiché è necessario andare a rigenerare il progetto con CMake:
```sh
  rm -rf build
  mkdir build
  cd build
  cmake ..
  make
```

Per eseguire valgrind utilizzare il seguente comando:
```sh
  valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./SCPA
```

> NOTA: questa esecuzione genera una cartella `build` all'interno del progetto. Il comando `valgrind` deve essere eseguito al suo interno.

> NOTA 2: Prima di mandare il codice in produzione (o eseguire l'analisi delle prestazioni), assicurarsi che i flag di compilazione aggiunti siano rimossi.
