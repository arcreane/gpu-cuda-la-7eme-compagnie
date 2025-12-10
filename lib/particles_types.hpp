#pragma once

#include <cstdint>

//Represente une particule dans le monde 2D
struct Particle {
    float   x;
    float   y;
    float   vx;
    float   vy;
    float   rad;
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

//Parametres globaux de la simulation (CPU et GPU utilisent exactement la meme struct)
struct SimParams {
    float dt         = 0.016f; //step de temps
    float damping    = 0.99f;  //amortissement des vitesses

    float mouseX     = 0.0f;
    float mouseY     = 0.0f;
    float mouseForce = 0.0f;
    float range      = 150.0f; //rayon d'action de la souris

    float worldWidth  = 0.0f;  //largeur de la zone de simulation
    float worldHeight = 0.0f;  //hauteur de la zone de simulation

    float elasticity  = 1.0f;  //coefficient d'elasticité des collisions

    float gravity     = 0.0f;  //acceleration verticale (pixels/s^2, vers le bas si > 0)
};

