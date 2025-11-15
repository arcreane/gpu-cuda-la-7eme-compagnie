#include "compute.hpp"
#include <cmath>
#include <vector>

namespace {
struct BackendCPU : IComputeBackend {
    std::vector<Particle> buf;
    explicit BackendCPU(size_t n) : buf(n) {}

    void upload(const std::vector<Particle>& host) override { buf = host; }

    void step(SimParams p) override {
        for (auto& s : buf) {
            float dx = p.mouseX - s.x;
            float dy = p.mouseY - s.y;
            float d2 = dx*dx + dy*dy;

            if (d2 < p.range * p.range && d2 > 1e-6f) {
                float invd = 1.0f / std::sqrt(d2);
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
        }
    }

    void download(std::vector<Particle>& host) override { host = buf; }
    size_t size() const override { return buf.size(); }
};
}

IComputeBackend* make_backend_cpu(size_t n) {
    return new BackendCPU(n);
}

