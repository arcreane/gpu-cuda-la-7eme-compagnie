#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    std::cout << "=== CUDA Detection Test ===" << std::endl;
    std::cout << "cudaGetDeviceCount returned: " << cudaGetErrorString(err) << std::endl;
    std::cout << "Device count: " << deviceCount << std::endl;

    if (err == cudaSuccess && deviceCount > 0) {
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
            std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
        }
    } else {
        std::cout << "\nNo CUDA devices found or error occurred." << std::endl;
        std::cout << "Error details: " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}
