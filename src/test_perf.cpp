#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "compute.hpp"

int main() {
    const size_t N = 100000; //nombre de particules
    const int steps = 1000;  //nombre de pas de simulation

    std::vector<Particle> init(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-500.f, 500.f);
    std::uniform_int_distribution<int>   C(50, 255);

    for (auto& p : init) {
        p.x = U(rng); p.y = U(rng);
        p.vx = 0.f;   p.vy = 0.f;
        p.r = (uint8_t)C(rng);
        p.g = (uint8_t)C(rng);
        p.b = (uint8_t)C(rng);
        p.a = 255;
    }

    IComputeBackend* backend = make_backend(N);
    backend->upload(init);

    SimParams params;
    params.dt = 0.016f;
    params.damping = 0.99f;
    params.mouseX = 0.f;
    params.mouseY = 0.f;
    params.mouseForce = 120.f;
    params.range = 200.f;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < steps; ++i) {
        backend->step(params);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dt = t1 - t0;
    std::cout << "Simulated " << steps << " steps with "
              << N << " particles in " << dt.count() << " ms.\n";

    std::vector<Particle> out;
    backend->download(out);
    std::cout << "Example x = " << out[0].x << "\n";

    delete backend;
    return 0;
}
