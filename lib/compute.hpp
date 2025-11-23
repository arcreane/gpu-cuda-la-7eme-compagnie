#pragma once

#include <cstddef>
#include <vector>
#include "particles_types.hpp"

//Interface generique pour un backend de calcul (CPU ou CUDA)
struct IComputeBackend {
    virtual ~IComputeBackend() = default;

    //Copie les particules depuis l'hote (CPU) vers le backend (CPU/GPU)
    virtual void upload(const std::vector<Particle>& host) = 0;

    //Fait avancer la simulation d'un pas de temps
    virtual void step(SimParams p) = 0;

    //Copie les particules depuis le backend vers l'hote (CPU)
    virtual void download(std::vector<Particle>& host) = 0;

    //Donne le nombre de particules gerees par ce backend
    virtual std::size_t size() const = 0;
};

//Fabrique un backend (CPU ou GPU) en fonction de la config globale
IComputeBackend* make_backend(std::size_t n);

//Fabrique le backend CPU
IComputeBackend* make_backend_cpu(std::size_t n);

//Fabrique le backend CUDA (implementé dans compute_cuda.cu).
IComputeBackend* make_backend_cuda(std::size_t n);

//Permet de dire à la fabrique si on veut utiliser CUDA ou non
//-false : backend CPU
//-true  : backend CUDA
void set_backend_use_cuda(bool enabled);
