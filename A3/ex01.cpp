// ex01.cpp
// nvcc -O3 ex01.cpp -o ex01
#include <vector>
#include <cuda_runtime_api.h>
int main(int argc, char* argv[]) {
int n = 4 * 1024 * 1024;
int size = n * sizeof(float);
std::vector<float> x(n);
std::vector<float> y(n);
float* d_x;
float* d_y;
cudaMalloc((void**)&d_x, size);
cudaMalloc((void**)&d_y, size);
cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);
cudaMemcpy(d_y, y.data(), size, cudaMemcpyHostToDevice);
// ...
cudaMemcpy(d_y, y.data(), size, cudaMemcpyDeviceToHost);
cudaFree(d_x);
return 0;
}