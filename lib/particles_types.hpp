#pragma once
#include <cstdint>

struct Particle {
    float x, y;
    float vx, vy;
    uint8_t r, g, b, a;
};

struct SimParams {
    float dt = 0.016f;
    float damping = 0.99f;
    float mouseX = 0.f, mouseY = 0.f;
    float mouseForce = 0.f;   // + attractif, - répulsif
    float range = 150.f;
};
