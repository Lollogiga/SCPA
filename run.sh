#!/bin/bash

dir="build"

# Controlla se la cartella esiste e la elimina
if [ -d "$dir" ]; then
  echo "Deleting $dir folder"
  rm -rf "$dir"
fi

mkdir "$dir"
cd "$dir" || exit 1  # Esce se il cd fallisce

# Forza il rilevamento di CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

cmake -DCMAKE_CUDA_ARCHITECTURES=89 .. || { echo "CMake failed"; exit 1; }
make || { echo "Make failed"; exit 1; }

# Controlla se l'eseguibile esiste prima di eseguirlo
if [ -f "./SCPA" ]; then
  ./SCPA
else
  echo "Error: ./SCPA not found!"
  exit 1
fi
