CC=gcc 
FORTRAN=gfortran

CFLAGS=-DCUBLAS_GFORTRAN -O2 -c 
FFLAGS=-DCUBLAS_GFORTRAN  -c -O2 -cpp
LFLAGS=-DCUBLAS_GFORTRAN 

ICUDA=-I /etc/alternatives/cuda/include -I /etc/alternatives/cuda/src
LIBS = -L /etc/alternatives/cuda/lib64 -llapack -fopenmp -lcublas -lcublas -lculibos -lcudart_static  
WRAPPER_LIBS = -lrt -ldl 

BUILD=./build/
BIN=./bin/

test: cuda_wrapp.o fortran.o test.o 
	$(FORTRAN) $(LFLAGS) $(BUILD)test.o  $(BUILD)cuda_wrapp.o $(BUILD)fortran.o  $(LIBS) $(WRAPPER_LIBS) -o $(BIN)test 
test.o: 
	$(FORTRAN) $(FFLAGS)  $(LIBS)  multigpumatmul.F90  -o $(BUILD)test.o
	rm *.mod

fortran.o: 
	 $(CC) $(CFLAGS) $(ICUDA)  ./src/fortran.c -o $(BUILD)fortran.o

cuda_wrapp.o: 
	 $(CC) $(CFLAGS) $(ICUDA)  ./cuda_wrappers/cuda.cpp -o $(BUILD)cuda_wrapp.o


clean: 
	rm ./build/*
	
	

	

