#include "sim_world.hpp"
#include <random>

SimWorld::SimWorld(size_t count, float width, float height)
    : m_width(width),
      m_height(height),
      m_host(count),
      m_backend(make_backend(count))
{
}

void SimWorld::randomInit(unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> X(0.f, m_width);
    std::uniform_real_distribution<float> Y(0.f, m_height);

    //rayon entre 2 et 6 pixels (on pourra parametrer plus tard sur Qt)
    std::uniform_real_distribution<float> R(2.f, 6.f);

    for (auto& p : m_host) {
        p.x = X(rng);
        p.y = Y(rng);
        p.vx = 0.f;
        p.vy = 0.f;
        p.rad = R(rng);
        p.r = 255;
        p.g = 255;
        p.b = 255;
        p.a = 255;
    }

    m_backend->upload(m_host);
}

void SimWorld::step(const SimParams& params) {
    m_backend->step(params);
    m_backend->download(m_host);
}
