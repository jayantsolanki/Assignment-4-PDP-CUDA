/*  YOUR_FIRST_NAME: Jayant
 *  YOUR_LAST_NAME: Solanki
 *  YOUR_UBIT_NAME: jayantso
 */

/*
* Referred from Mark Harris Blog: https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/
* Some stats: https://goo.gl/zQLXWN
*
*/
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#ifndef A3_HPP
#define A3_HPP
#define PI 3.1416


__global__ void gpuGaussianKernel(int n, float *x, float *y, float h)
{

	// int index = threadIdx.x;
	// int stride = blockDim.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		float temp = 0;
		float temp2 = 0;
		for(int j = 0; j<n; j++)
		{
			temp2 = (x[i] - x[j])/h;
			temp2 = temp2*temp2;
			temp2 = -1*temp2/2;
			temp = temp + exp(temp2)/sqrt(2*PI); //calculated K for each x
		}
		y[i] = temp/(n*h);
	}
  	
}

void cpuGaussianKernel(int n, float *x, std::vector<float>& y, float h)//calculating GKDE using hot device only
{
	for(int i = 0; i<n; i++)
	{
		float temp = 0;
		float temp2 = 0;
		for(int j = 0; j<n; j++){
			temp2 = (x[i] - x[j])/h;
			temp2 = temp2*temp2;
			temp2 = -1*temp2/2;
			temp = temp + exp(temp2)/sqrt(2*PI); //calculated K for each x
		}
		y[i] = temp/(n*h);
	}

}
void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
	// const float *X = &x[0];
	// float *XK;
	float *X, *Y;
	// Allocate Unified Memory – accessible from CPU or GPU;
	cudaMallocManaged(&X, n*sizeof(float));
	cudaMallocManaged(&Y, n*sizeof(float));
	// cudaMallocManaged(&XK, n*sizeof(float));
	// initialize x and y arrays on the host
	for (int i = 0; i < n; i++) {//abusing the vector as I couldnt debug the error related to const and vector
		X[i] = x[i];
		// Y[i] = y[i];
	}
	// for(int i = 0; i<n; i++)
	// 	std::cout << "Elements of x are "<<x[i]<<""<<std::endl;
	// std::cout << "=============GKDE using CPU=================="<<std::endl;
	// cpuGaussianKernel(n, X, y, h);
	// for(int i = 0; i<n; i++)
	// 	std::cout << "Elements of y are "<<y[i]<<""<<std::endl;
	std::cout << "=============GKDE using GPU=================="<<std::endl;
	int blockSize = 256;//didnt take 512 or 1024
	int numBlocks = (n + blockSize - 1) / blockSize;
	gpuGaussianKernel<<<numBlocks, blockSize>>>(n, X, Y, h);//vector was not getting identified so I had to resort to this hack
	cudaDeviceSynchronize();
	for (int i = 0; i < n; i++) {//abusing the vector as I couldnt debug the error related to const and vector
		y[i] = Y[i];
		// Y[i] = y[i];
	}
	// *y = &Y[0];
	// for(int i = 0; i<n; i++)
	// 	std::cout << "Elements of y are "<<y[i]<<""<<std::endl;
	// // Check for errors (all GPU calculated values should be equal to CPU calculated values)
	// float maxError = 0.0f;
	// for (int i = 0; i < n; i++)
	// 	maxError = fmax(maxError, fabs(Y[i]-y[i]));
	// std::cout << "Max error: " << maxError << std::endl;
	// Free memory
	cudaFree(X);
	cudaFree(Y);
	
	

} // gaussian_kde



#endif // A3_HPP
