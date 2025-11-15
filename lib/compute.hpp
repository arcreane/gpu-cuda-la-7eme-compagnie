#pragma once
#include <vector>
#include "particles_types.hpp"

class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;
    virtual void upload(const std::vector<Particle>& host) = 0;
    virtual void step(SimParams p) = 0;
    virtual void download(std::vector<Particle>& host) = 0;
    virtual size_t size() const = 0;
};

// backends concrets
IComputeBackend* make_backend_cpu(size_t n);
IComputeBackend* make_backend_cuda(size_t n); // peut retourner nullptr si pas de gpu

// factory unique
IComputeBackend* make_backend(size_t n);
