#include "compute.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

//Kernel GPU : même logique que BackendCPU::step, mais exécuté en parallèle
__global__ void step_kernel(Particle* particles, SimParams p, size_t n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle s = particles[i];

    float dx = p.mouseX - s.x;
    float dy = p.mouseY - s.y;
    float d2 = dx * dx + dy * dy;

    if (d2 < p.range * p.range && d2 > 1e-6f) {
        float invd = rsqrtf(d2); // 1/sqrt(d2) version rapide
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

    particles[i] = s;
}

namespace
{
    struct BackendCUDA : IComputeBackend {
        size_t n;
        Particle* d_buf = nullptr;
        std::vector<Particle> h_tmp;

        explicit BackendCUDA(size_t n_)
            : n(n_), h_tmp(n_)
        {
            // On suppose que backend_factory ne crée cet objet que si un device CUDA est dispo.
            cudaError_t e = cudaMalloc(&d_buf, n * sizeof(Particle));
            if (e != cudaSuccess) {
                std::printf("[CUDA backend] cudaMalloc failed: %s\n",
                            cudaGetErrorString(e));
                d_buf = nullptr;
            }
        }

        ~BackendCUDA() override
        {
            if (d_buf) cudaFree(d_buf);
        }

        void upload(const std::vector<Particle>& host) override
        {
            h_tmp = host;
            if (!d_buf) return;
            cudaMemcpy(d_buf, h_tmp.data(),
                       n * sizeof(Particle),
                       cudaMemcpyHostToDevice);
        }

        void step(SimParams p) override
        {
            if (!d_buf) return;
            const int blockSize = 256;
            const int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);
            step_kernel<<<gridSize, blockSize>>>(d_buf, p, n);
            cudaDeviceSynchronize();
        }

        void download(std::vector<Particle>& host) override
        {
            host.resize(n);
            if (!d_buf) {
                host = h_tmp; // au pire on renvoie la dernière copie host
                return;
            }
            cudaMemcpy(host.data(), d_buf,
                       n * sizeof(Particle),
                       cudaMemcpyDeviceToHost);
        }

        size_t size() const override { return n; }
    };
}

// Fabrique CUDA : appelée uniquement si un device est dispo
IComputeBackend* make_backend_cuda(size_t n)
{
    return new BackendCUDA(n);
}
