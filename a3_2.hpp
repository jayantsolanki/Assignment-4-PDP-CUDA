/*  YOUR_FIRST_NAME: Jayant
 *  YOUR_LAST_NAME: Solanki
 *  YOUR_UBIT_NAME: jayantso
 */
#include <iostream>
#include <cuda_runtime_api.h>
#ifndef A3_HPP
#define A3_HPP
#define THREADS_PER_BLOCK	256
#define BLOCKS_PER_GRID_ROW 128

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
	float *d_Mean;
	float *h_Mean;
	for(int i = 0; i<n; i++)
		std::cout << "Elements are "<<x[i]<<""<<std::endl;
	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;
	h_resultAvg = (float *)malloc(sizeof(float) * n / THREADS_PER_BLOCK);
	CUDA_SAFE_CALL( cudaMalloc( (void **)&d_data, sizeof(float) * MAX_DATA_SIZE) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&d_Mean, sizeof(float) * n / THREADS_PER_BLOCK) );


	for (int dataAmount = n; dataAmount > BLOCKS_PER_GRID_ROW*THREADS_PER_BLOCK; dataAmount /= 2)
	{
		float tempMean;
		int blockGridWidth = BLOCKS_PER_GRID_ROW;
		int blockGridHeight = (dataAmount / THREADS_PER_BLOCK) / blockGridWidth;
		dim3 blockGridRows(blockGridWidth, blockGridHeight);
		dim3 threadBlockRows(THREADS_PER_BLOCK, 1);
		CUDA_SAFE_CALL( cudaMemcpy(d_x, x, sizeof(float) * dataAmount, cudaMemcpyHostToDevice) );
		meanGPU<<<blockGridRows, threadBlockRows>>>(n, d_x, d_Mean);
		CUDA_SAFE_CALL( cudaThreadSynchronize() );d_Mean
	}

} // gaussian_kde

void meanGPU(int n, float *x, float *AVG)
{

	// int index = blockIdx.x * blockDim.x + threadIdx.x;
	// int stride = blockDim.x * gridDim.x;
	// for (int i = index; i < n; i += stride)
	//     y[i] = x[i] + y[i];
	__shared__ float mean[256];
	int nTotalThreads = blockDim.x;	// Total number of active threads


	while(nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.

		if (threadIdx.x < halfPoint)
		{
			// when calculating the average, sum and divide
			mean[threadIdx.x] += mean[threadIdx.x + halfPoint];

			mean[threadIdx.x] /= 2;
		}
		__syncthreads();

		nTotalThreads = (nTotalThreads >> 1);	// divide by two.

	}
	if (threadIdx.x == 0)
	{
		AVG[128*blockIdx.y + blockIdx.x] = mean[0];
	}
}

#endif // A3_HPP
