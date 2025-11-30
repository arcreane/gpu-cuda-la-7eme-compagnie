#pragma once

#include <cstddef>
#include <vector>
#include "particles_types.hpp"

//Interface commune à tous les backends de calcul (CPU ou GPU)
class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;

    //Copie les données utilisateur to backend (CPU/GPU)
    virtual void upload(const std::vector<Particle>& host) = 0;

    //Effectue un pas de simulation avec les paramètres donnés
    virtual void step(SimParams p) = 0;

    //Copie les données backend vers utilisateur
    virtual void download(std::vector<Particle>& host) = 0;

    //Nombre de particules gerées
    virtual std::size_t size() const = 0;
};

//Backend spécifique CPU (implémentée dans compute_cpu.cpp)
IComputeBackend* make_backend_cpu(std::size_t n);

//Backend spécifique CUDA (implémentée dans compute_cuda.cu)
IComputeBackend* make_backend_cuda(std::size_t n);

//Backend qui choisit CPU ou GPU selon un flag global
IComputeBackend* make_backend(std::size_t n);

//Permet d'activer/désactiver l'utilisation de CUDA
void set_backend_use_cuda(bool enabled);
