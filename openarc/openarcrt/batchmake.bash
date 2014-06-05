#! /bin/bash

make purge
make OMP=1 MODE=normal
make clean
make OMP=1 MODE=profile
make clean
make OMP=0 MODE=normal
make clean
make OMP=0 MODE=profile
make clean
make binUtil PRINT_LOG=1
make clean
make res
gcc -o Timer Timer.c
