#include <cstdio>
#include <cuda_runtime.h>

__global__ void addOne(int* v){ *v += 1; }

int main(){
    int h = 41;
    int* d = nullptr;
    cudaError_t err = cudaMalloc(&d, sizeof(int));
    if(err != cudaSuccess){ printf("cudaMalloc error\n"); return 1; }

    cudaMemcpy(d, &h, sizeof(int), cudaMemcpyHostToDevice);
    addOne<<<1,1>>>(d);
    cudaDeviceSynchronize();
    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);

    printf("Result = %d\n", h); // attendu: 42
    return 0;
}
