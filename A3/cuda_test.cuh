#indef CUDA_TEST_HPP
#define CUDA_TEST_HPP

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

void cudaCallAddVectorKernel(
	const uint block_count,
	const uint per_block_thread_count,
	const float *a,
	const float *b
	float *c,
	const uint size);

#endif