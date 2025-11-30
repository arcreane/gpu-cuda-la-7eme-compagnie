#include "compute.hpp"
#include <iostream>

//Flag global : false = CPU, true = CUDA
static bool g_use_cuda = false;

void set_backend_use_cuda(bool enabled) {
    g_use_cuda = enabled;
}

IComputeBackend* make_backend(std::size_t n) {
    if (g_use_cuda) {
        std::cout << "[backend_factory] Using CUDA backend (" << n << " particles)\n";
        return make_backend_cuda(n);
    } else {
        std::cout << "[backend_factory] Using CPU backend (" << n << " particles)\n";
        return make_backend_cpu(n);
    }
}

