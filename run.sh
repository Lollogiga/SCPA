#!/bin/bash

dir="build"

# Check if folder exist
if [ -d "$dir" ]; then
  echo "Deleting $dir folder"
  rm -rf "$dir"
fi

mkdir "$dir"
cd "$dir" || exit 1  # If cd command fails, exit from script

cmake ..
make

./SCPA
