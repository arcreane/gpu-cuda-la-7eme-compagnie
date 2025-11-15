#include "compute.hpp"
#include <iostream>

IComputeBackend* make_backend(size_t n) {
    std::cout << "[factory] Using CPU backend\n";
    return make_backend_cpu(n);
}
