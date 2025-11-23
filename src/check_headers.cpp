#include <iostream>
#include "particles_types.hpp"
#include "compute.hpp"

int main() {
    Particle p{0,0,0,0,3.0f,255,255,255,255};
    SimParams s;
    std::cout << "Headers OK: dt=" << s.dt << ", px=" << p.x << ", rad=" << p.rad << "\n";
    return 0;
}

