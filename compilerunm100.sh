echo "Compile CUDA  Wrappers"
echo "If other wrappers are missing probably are the ones provided by cuda in the fortran.h and fortran.cpp files"
echo 
gcc -DCUBLAS_GFORTRAN -c -I/cineca/prod/opt/compilers/cuda/10.1/none/update2/include cuda_wrappers/cuda_custom.cpp -o cuda.o
echo 		DONE!
echo REMEMBER:
echo  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cineca/prod/opt/libraries/lapack/3.9.0/gnu--8.4.0/lib:/cineca/prod/opt/compilers/cuda/10.1/none/update2/lib64:/cineca/prod/opt/compilers/gnu/8.4.0/none/lib64
echo 


echo Compile MAtmul code
gfortran  -DCUBLAS_GFORTRAN -c -g -cpp -L/cineca/prod/opt/compilers/cuda/10.1/none/update2/lib64  -lcublas -lculibos -lcudart  multigpumatmul.F90  -o test.o
echo 
echo 	DONE!
echo
echo LINK everything

gfortran -shared  -DCUBLAS_GFORTRAN  test.o cuda.o fortran.o -L /cineca/prod/opt/libraries/lapack/3.9.0/gnu--8.4.0/lib -L /cineca/prod/opt/libraries/blas/3.8.0/gnu--8.4.0/lib -fopenmp -llapack -l:libblas.a -L/cineca/prod/opt/compilers/cuda/10.1/none/update2/lib64 -L /cineca/prod/opt/compilers/gnu/8.4.0/none/lib64 -lcublas -lculibos -lcudart -lrt -ldl -o test 
echo 	DONE!
 


