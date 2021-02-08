

typedef size_t ptr_t;

/*
 * Example of Fortran callable thin wrappers for a few CUDA functions.
 */
#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */
void MY_TEST ();

int CUDA_MALLOC(ptr_t *devPtr,
                int *size);

int CUDA_SET_DEVICE(int* device);

int CUDA_GET_DEVICE(int* device);

int CUDA_GET_DEVICE_COUNT(int* count);

int CUDA_MEM_GET_INFO(float* free, float* total);



#if defined(__cplusplus)
}
#endif /* __cplusplus */

