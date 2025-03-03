#!/bin/bash

# Rimuove e ricrea la directory di build
rm -rf build
mkdir build
cd build

# Genera il progetto con CMake e compila
cmake ..
make

# Esegue Valgrind con il binario risultante
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./SCPA
