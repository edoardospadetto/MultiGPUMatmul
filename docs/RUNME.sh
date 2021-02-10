#!/bin/bash
echo "1) Compile Cuda Wrappers"
gcc -DCUBLAS_GFORTRAN -c -I /etc/alternatives/cuda/include -I/etc/alternatives/cuda/src ./src/fortran.c -o fortran.o

gcc -DCUBLAS_GFORTRAN -c -I /etc/alternatives/cuda/include -I/etc/alternatives/cuda/src cuda_wrappers/cuda.cpp -o cuda.o
echo "			DONE !"
echo
echo "2) Compile the actual code"
gfortran  -DCUBLAS_GFORTRAN -c -g -cpp -L /etc/alternatives/cuda/lib64 -fopenmp -lcublas_static -lculibos -lcudart_static  multigpumatmul.F90  -o test.o
echo "			DONE !" 
echo
echo "3) Link .o files"
gfortran -DCUBLAS_GFORTRAN  test.o cuda.o  fortran.o -llapack -L /etc/alternatives/cuda/lib64  -fopenmp -lcublas -lculibos -lcudart_static -lrt -ldl -o test 
echo "			DONE !"
echo 
echo "RUNNING.."
./test

