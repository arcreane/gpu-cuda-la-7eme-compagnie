#pragma once
#include <vector>
#include <memory>
#include "particles_types.hpp"
#include "compute.hpp"

//G estion particules utilisant CPU ou CUDA via la factory
class SimWorld {
public:
    SimWorld(size_t count, float width, float height);

    //Génère des particules aléatoires dans [0,width] x [0,height]
    void randomInit(unsigned int seed = 42);

    // Fait avancer la simulation d'un pas
    void step(const SimParams& params);

    // Retourne la liste de particules (pour affichage Raylib/Qt)
    const std::vector<Particle>& particles() const { return m_host; }

    size_t size() const { return m_host.size(); }

private:
    float m_width;
    float m_height;
    std::vector<Particle> m_host;
    std::unique_ptr<IComputeBackend> m_backend;
};
