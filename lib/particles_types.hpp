#pragma once

#include <cstdint>

//Represente une particule dans la simulation
struct Particle {
    float x;    //position X
    float y;    //position Y
    float vx;   //vitesse X
    float vy;   //vitesse Y
    float rad;  //rayon des particules (pour les collisions entre particules)

    std::uint8_t r; //rouge (0-255)
    std::uint8_t g; //vert
    std::uint8_t b; //bleu
    std::uint8_t a; //alpha (0-255)
};

//Parametres de la simulation
//Ce struct est partagé entre CPU et CUDA
struct SimParams {
    //step de temps
    float dt        = 0.016f;

    //Facteur de damping global sur les vitesses
    //(0.0 = tout s’arrête, ~1.0 = très peu de frottement)
    float damping   = 0.99f;

    //Position de la souris dans la zone de rendu
    float mouseX    = 0.f;
    float mouseY    = 0.f;

    //Force exercee par la souris (attraction/repulsion)
    float mouseForce = 0.f;

    //Rayon d’influence de la souris
    float range      = 150.f;

    // Dimensions du monde (zone de rendu/fenetre Raylib)
    // Si <= 0, on considere qu'il n'y a pas de murs pour ce backend
    float worldWidth  = 0.f;
    float worldHeight = 0.f;

    //Coefficient d'elasticité des collisions entre particules (epsilon)
    //0.0 = choc parfaitement non elastique, 1.0 = choc parfaitement elastique.
    float elasticity  = 1.0f;
};
