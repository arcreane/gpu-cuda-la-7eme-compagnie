#include "compute.hpp"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void step_kernel(Particle* d, size_t n, SimParams p) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle s = d[i];

    float dx = p.mouseX - s.x;
    float dy = p.mouseY - s.y;
    float d2 = dx * dx + dy * dy;

    if (d2 < p.range * p.range && d2 > 1e-6f) {
        float invd = rsqrtf(d2);
        float fx = p.mouseForce * dx * invd;
        float fy = p.mouseForce * dy * invd;
        s.vx = (s.vx + fx * p.dt) * p.damping;
        s.vy = (s.vy + fy * p.dt) * p.damping;
    } else {
        s.vx *= p.damping;
        s.vy *= p.damping;
    }

    s.x += s.vx * p.dt;
    s.y += s.vy * p.dt;

    d[i] = s;
}

namespace {
struct BackendCUDA : IComputeBackend {
    Particle* d_ptr = nullptr;
    size_t n = 0;

    explicit BackendCUDA(size_t n_) : n(n_) {
        cudaError_t e = cudaMalloc(&d_ptr, n * sizeof(Particle));
        if (e != cudaSuccess) {
            std::printf("[CUDA] cudaMalloc failed: %s\n", cudaGetErrorString(e));
            d_ptr = nullptr;
            n = 0;
        }
    }

    ~BackendCUDA() override {
        if (d_ptr) cudaFree(d_ptr);
    }

    void upload(const std::vector<Particle>& host) override {
        if (!d_ptr || host.size() < n) return;
        cudaMemcpy(d_ptr, host.data(), n * sizeof(Particle), cudaMemcpyHostToDevice);
    }

    void step(SimParams p) override {
        if (!d_ptr || n == 0) return;
        int block = 256;
        int grid = (int)((n + block - 1) / block);
        step_kernel<<<grid, block>>>(d_ptr, n, p);
        cudaDeviceSynchronize();
    }

    void download(std::vector<Particle>& host) override {
        if (!d_ptr || n == 0) return;
        host.resize(n);
        cudaMemcpy(host.data(), d_ptr, n * sizeof(Particle), cudaMemcpyDeviceToHost);
    }

    size_t size() const override { return n; }
};
} //namespace

//Essai cuda d'abord puis cpu si cuda marche pas
IComputeBackend* make_backend(size_t n) {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
        std::printf("[Factory] No CUDA device, using CPU backend.\n");
        return make_backend_cpu(n);
    }

    std::printf("[Factory] CUDA device detected, using GPU backend.\n");
    IComputeBackend* gpu = new BackendCUDA(n);

    //si l’allocation gpu a raté on rebascule cpu
    if (gpu->size() == 0) {
        delete gpu;
        std::printf("[Factory] GPU alloc failed, falling back to CPU backend.\n");
        return make_backend_cpu(n);
    }
    return gpu;
}
