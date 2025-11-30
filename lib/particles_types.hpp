#pragma once

#include <cstdint>

//Représente une particule dans le monde 2D
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

// Paramètres globaux de la simulation (CPU et GPU utilisent exactement la même struct)
struct SimParams {
    float dt         = 0.016f; //step de temps
    float damping    = 0.99f;  //amortissement des vitesses

    float mouseX     = 0.0f;
    float mouseY     = 0.0f;
    float mouseForce = 0.0f;
    float range      = 150.0f; //rayon d'influence de la souris

    float worldWidth  = 0.0f;  //largeur de la zone de simulation
    float worldHeight = 0.0f;  //hauteur de la zone de simulation

    float elasticity  = 1.0f;  //coefficient d'élasticité des collisions

    float gravity     = 0.0f;  //accélération verticale (pixels/s^2, vers le bas si > 0)
};

