#include "raylib.h"
#include "sim_world.hpp"
#include <cmath>

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int main() {
    const int screenWidth  = 1280;
    const int screenHeight = 720;

    InitWindow(screenWidth, screenHeight, "GPU Particle Sim - Raylib Demo (CPU)");
    SetTargetFPS(60);

    //Nombre de particules
    const size_t N = 3000;
    SimWorld world(N, (float)screenWidth, (float)screenHeight);
    world.randomInit();

    //Parametres de simulation
    SimParams params;
    params.dt         = 1.0f / 60.0f;
    params.damping    = 0.98f;
    params.mouseX     = 0.f;
    params.mouseY     = 0.f;
    params.mouseForce = 0.f;
    params.range      = 150.f;

    params.worldWidth  = (float)screenWidth;
    params.worldHeight = (float)screenHeight;

    //Coefficient d'elasticité des collisions entre particules
    params.elasticity  = 1.0f; //1.0 = tres elastique

    bool paused = false;

    //Pour mesurer la vitesse du curseur
    Vector2 prevMouse = GetMousePosition();
    float   mouseSpeed = 0.0f;  //En pixels/seconde (approx)

    while (!WindowShouldClose()) {

        //INPUT GLOBAL/CONTROLES
	float frameDt = GetFrameTime();
        frameDt = clampf(frameDt, 0.0f, 0.05f);

        //Pause/reprise
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }

        //Reset des particules
        if (IsKeyPressed(KEY_R)) {
            world.randomInit();
        }

        //Ajuster le damping
        if (IsKeyDown(KEY_UP)) {
            params.damping = clampf(params.damping + 0.5f * frameDt, 0.80f, 0.999f);
        }
        if (IsKeyDown(KEY_DOWN)) {
            params.damping = clampf(params.damping - 0.5f * frameDt, 0.80f, 0.999f);
        }

        //Ajuster l'elasticité (epsilon)
        if (IsKeyDown(KEY_RIGHT)) {
            params.elasticity = clampf(params.elasticity + 0.5f * frameDt, 0.0f, 1.0f);
        }
        if (IsKeyDown(KEY_LEFT)) {
            params.elasticity = clampf(params.elasticity - 0.5f * frameDt, 0.0f, 1.0f);
        }

        //MISE A JOUR SOURIS & FORCE
	Vector2 mouse = GetMousePosition();
        params.mouseX = mouse.x;
        params.mouseY = mouse.y;

        //Calcul de la vitesse du curseur (distance/dt)
        float dx = mouse.x - prevMouse.x;
        float dy = mouse.y - prevMouse.y;
        float dist = std::sqrt(dx*dx + dy*dy);

        if (frameDt > 1e-6f) {
            mouseSpeed = dist / frameDt;//pixels/seconde (approx)
        } else {
            mouseSpeed = 0.0f;
        }

        //Coefficient d'echelle pour transformer la vitesse en force
        const float forceScale = 0.5f; //a ajuster si on veut

        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            //Attraction proportionnelle à la vitesse du curseur
            params.mouseForce =  forceScale * mouseSpeed;
        } else if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
            //Repulsion proportionnelle à la vitesse du curseur
            params.mouseForce = -forceScale * mouseSpeed;
        } else {
            params.mouseForce = 0.0f;
        }

        prevMouse = mouse;

        //UPDATE
        params.dt = frameDt;

        if (!paused) {
            world.step(params);
        }
        const auto& pts = world.particles();

        //DRAW
        BeginDrawing();
        ClearBackground(BLACK);

        //Dessin des particules (cercles rayon p.rad)
        for (const auto& p : pts) {
            int ix = (int)clampf(p.x, 0.f, (float)screenWidth  - 1.f);
            int iy = (int)clampf(p.y, 0.f, (float)screenHeight - 1.f);

            Color c;
            c.r = (unsigned char)p.r;
            c.g = (unsigned char)p.g;
            c.b = (unsigned char)p.b;
            c.a = (unsigned char)p.a;

            DrawCircle((float)ix, (float)iy, p.rad, c);
        }

        //HUD/Infos
        DrawText(TextFormat("dt = %.3f ms", frameDt * 1000.0f), 10, 10, 20, WHITE);
        DrawText(TextFormat("mouseSpeed = %.1f px/s", mouseSpeed), 10, 30, 20, WHITE);
        DrawText(TextFormat("mouseForce = %.1f", params.mouseForce), 10, 50, 20, WHITE);
        DrawText(TextFormat("N = %d", (int)pts.size()), 10, 70, 20, WHITE);
        DrawText(TextFormat("FPS = %d", GetFPS()), 10, 90, 20, WHITE);
        DrawText(TextFormat("damping = %.3f", params.damping), 10, 110, 20, WHITE);
        DrawText(TextFormat("elasticity (epsilon) = %.3f", params.elasticity), 10, 130, 20, WHITE);

        DrawText("SPACE: pause/resume", 10, 160, 18, GRAY);
        DrawText("R: reset particles", 10, 180, 18, GRAY);
        DrawText("UP/DOWN: change damping", 10, 200, 18, GRAY);
        DrawText("LEFT/RIGHT: change elasticity", 10, 220, 18, GRAY);
        DrawText("LMB/RMB + move mouse: force ~ cursor speed", 10, 240, 18, GRAY);

        if (paused) {
            DrawText("PAUSED", screenWidth - 150, 10, 30, YELLOW);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
