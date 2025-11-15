#include <iostream>
#include <vector>
#include <random>
#include "compute.hpp"

int main() {
    const size_t N = 128;
    std::vector<Particle> init(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-100.f, 100.f);
    for (auto& p : init) { p.x=U(rng); p.y=U(rng); p.vx=p.vy=0; p.r=p.g=p.b=200; p.a=255; }

    IComputeBackend* backend = make_backend_cpu(N);
    backend->upload(init);

    SimParams p; p.mouseX=0; p.mouseY=0; p.mouseForce=120.f; p.range=150.f;
    for (int i=0; i<100; ++i) backend->step(p);

    std::vector<Particle> out;
    backend->download(out);
    std::cout << "OK CPU-only: " << out.size() << " particles. Example x=" << out[0].x << "\n";
    delete backend;
    return 0;
}
