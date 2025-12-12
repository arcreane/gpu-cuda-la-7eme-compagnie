#pragma once

#include <cstddef>
#include <vector>
#include "particles_types.hpp"

//Interface commune à tous les backends (CPU ou GPU)
class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;

    //Copie les donnees utilisateur vers backend (CPU/GPU)
    virtual void upload(const std::vector<Particle>& host) = 0;

    //Effectue un pas de simulation avec les parametres donnés
    virtual void step(SimParams p) = 0;

    //Copie les donnees backend vers utilisateur
    virtual void download(std::vector<Particle>& host) = 0;

    //Nombre de particules gerees
    virtual std::size_t size() const = 0;
};

//Backend specifique CPU (implementé dans compute_cpu.cpp)
IComputeBackend* make_backend_cpu(std::size_t n);

//Backend specifique CUDA (implementée dans compute_cuda.cu)
IComputeBackend* make_backend_cuda(std::size_t n);

//Backend qui choisit CPU ou GPU selon un flag global
IComputeBackend* make_backend(std::size_t n);

//Permet d'activer/desactiver l'utilisation de CUDA
void set_backend_use_cuda(bool enabled);
