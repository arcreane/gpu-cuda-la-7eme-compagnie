#include "compute.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

//Petit helper pour checker les erreurs CUDA
static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] " << msg << " : " << cudaGetErrorString(err) << "\n";
    }
}

//Kernel CUDA : un thread par particule
//Version simplifiee : force souris + damping + murs, sans collisions inter-particules pour l'instant.
__global__ void step_kernel(Particle* buf, std::size_t n, SimParams p) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle s = buf[i];

    const float range2 = p.range * p.range;

    //Force de la souris
    float dx = p.mouseX - s.x;
    float dy = p.mouseY - s.y;
    float d2 = dx*dx + dy*dy;

    if (d2 < range2 && d2 > 1e-6f && p.mouseForce != 0.0f) {
        float invd = rsqrtf(d2); //1/sqrt(d2)
        float fx = p.mouseForce * dx * invd;
        float fy = p.mouseForce * dy * invd;

        s.vx = (s.vx + fx * p.dt) * p.damping;
        s.vy = (s.vy + fy * p.dt) * p.damping;
    } else {
        s.vx *= p.damping;
        s.vy *= p.damping;
    }

    //Integration de la position
    s.x += s.vx * p.dt;
    s.y += s.vy * p.dt;

    //Collisions avec les murs
    if (p.worldWidth > 0.0f) {
        if (s.x - s.rad < 0.0f) {
            s.x = s.rad;
            s.vx = -s.vx;
        } else if (s.x + s.rad > p.worldWidth) {
            s.x = p.worldWidth - s.rad;
            s.vx = -s.vx;
        }
    }

    if (p.worldHeight > 0.0f) {
        if (s.y - s.rad < 0.0f) {
            s.y = s.rad;
            s.vy = -s.vy;
        } else if (s.y + s.rad > p.worldHeight) {
            s.y = p.worldHeight - s.rad;
            s.vy = -s.vy;
        }
    }

    //On reecrit la particule mise à jour
    buf[i] = s;
}

//Backend CUDA concret
class BackendCUDA : public IComputeBackend {
public:
    explicit BackendCUDA(std::size_t n)
        : m_n(n), d_buf(nullptr) 
    {
        std::size_t bytes = m_n * sizeof(Particle);
        checkCuda(cudaMalloc(&d_buf, bytes), "cudaMalloc d_buf");
    }

    ~BackendCUDA() override {
        if (d_buf) {
            cudaFree(d_buf);
        }
    }

    void upload(const std::vector<Particle>& host) override {
        if (host.size() != m_n) {
            std::cerr << "[BackendCUDA] upload: host size mismatch\n";
            return;
        }
        std::size_t bytes = m_n * sizeof(Particle);
        checkCuda(cudaMemcpy(d_buf, host.data(), bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy H->D");
    }

    void step(SimParams p) override {
        const int blockSize = 256;
        const int gridSize  = static_cast<int>((m_n + blockSize - 1) / blockSize);

        step_kernel<<<gridSize, blockSize>>>(d_buf, m_n, p);
        cudaError_t err = cudaGetLastError();
        checkCuda(err, "kernel launch");
        if (err == cudaSuccess) {
            checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }
    }

    void download(std::vector<Particle>& host) override {
        host.resize(m_n);
        std::size_t bytes = m_n * sizeof(Particle);
        checkCuda(cudaMemcpy(host.data(), d_buf, bytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy D->H");
    }

    std::size_t size() const override {
        return m_n;
    }

private:
    std::size_t m_n;
    Particle* d_buf;
};

//Fabrique le backend CUDA
IComputeBackend* make_backend_cuda(std::size_t n) {
    return new BackendCUDA(n);
}
