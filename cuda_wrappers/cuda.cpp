#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__GNUC__)
#    include <stdint.h>
#endif                    /* __GNUC__ */
#include "cuda_runtime.h" /* CUDA public header file     */
#include "cusparse_fortran_common.h"
#include "cusparse.h" /* CUSPARSE public header file */
#include "cusparse_fortran.h"

#define MY_TEST             	my_test_
#define CUDA_GET_DEVICE_COUNT   cuda_get_device_count_
#define CUDA_GET_DEVICE         cuda_get_device_
#define CUDA_SET_DEVICE         cuda_set_device_
#define CUDA_MEM_GET_INFO	cuda_mem_get_info_
#include "cuda.h"


/*---------------------------------------------------------------------------*/
/*------------------------- AUXILIARY FUNCTIONS -----------------------------*/
/*---------------------------------------------------------------------------*/

void MY_TEST (){
	printf("I am a dummy function from c++");
	 
}	


int CUDA_GET_DEVICE_COUNT(int* count) {
	
	int error = (int) cudaGetDeviceCount (count);
	
	return error;
	
}

int CUDA_SET_DEVICE(int* device) {
	return (int) cudaSetDevice(*device);

}

int CUDA_GET_DEVICE(int* device) {
	return (int) cudaGetDevice (device);
}


	
// Maybe the cast in the following is unsafe? 
int  CUDA_MEM_GET_INFO(float* free,float* total){
	//const float bytetomb = 1048576.0;
	size_t free_t, total_t;
	int error = cudaMemGetInfo(&free_t,&total_t);
	*free = (float) (free_t) ;
	*total = (float) (total_t) ;
	return error;
	}

	
