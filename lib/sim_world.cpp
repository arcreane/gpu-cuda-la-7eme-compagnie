#include "sim_world.hpp"
#include <random>
#include <iostream>

SimWorld::SimWorld(size_t count, float width, float height)
    : m_width(width), m_height(height)
{
    m_host.resize(count);

    //Sélection cpu ou cuda automatiquement
    m_backend.reset(make_backend(count));
    if (!m_backend) {
        std::cerr << "[SimWorld] ERREUR: backend nul\n";
    }
}

void SimWorld::randomInit(unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> UX(0.f, m_width);
    std::uniform_real_distribution<float> UY(0.f, m_height);
    std::uniform_int_distribution<int> C(120, 255);

    for (auto& p : m_host) {
        p.x = UX(rng);
        p.y = UY(rng);
        p.vx = 0.f;
        p.vy = 0.f;
        p.r = (uint8_t)C(rng);
        p.g = (uint8_t)C(rng);
        p.b = (uint8_t)C(rng);
        p.a = 255;
    }

    m_backend->upload(m_host);
}

void SimWorld::step(const SimParams& params)
{
    // Effectue le calcul sur cpu ou cuda
    m_backend->step(params);

    // Récupère les nouvelles positions
    m_backend->download(m_host);
}
