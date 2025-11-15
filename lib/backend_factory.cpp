#include "compute.hpp"
#include <iostream>

IComputeBackend* make_backend(size_t n) {
    // On essaiera le GPU plus tard ; pour l'instant, on reste CPU-only,
    // mais l’API est prête.
    std::cout << "[factory] Using CPU backend\n";
    return make_backend_cpu(n);
}
