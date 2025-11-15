#include <iostream>
#include "sim_world.hpp"

int main() {
    SimWorld world(2000, 800, 600);
    world.randomInit();

    SimParams p;
    p.dt = 0.016f;
    p.damping = 0.99f;
    p.mouseX = 400.f;
    p.mouseY = 300.f;
    p.mouseForce = 200.f;
    p.range = 200.f;

    for (int i = 0; i < 300; ++i)
        world.step(p);

    const auto& pts = world.particles();
    std::cout << "Example particle: x=" << pts[0].x << " y=" << pts[0].y << "\n";
}
