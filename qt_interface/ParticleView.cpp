#include "ParticleView.h"

#include <QVBoxLayout>
#include <QResizeEvent>
#include <QWindow>
#include <cmath>

#include "raylib.h"

//Fenetre Raylib partagée
static bool     g_raylibInit  = false;
static QWindow* g_rayWindow   = nullptr;

ParticleView::ParticleView(QWidget* parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_OpaquePaintEvent);
    setAttribute(Qt::WA_NoSystemBackground);

    //Layout pour mettre la fenetre Raylib dans Qt
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    ensureRaylibInitialized();

    if (g_rayWindow) {
        //Crée un conteneur Qt pour la fenetre Raylib
        m_container = QWidget::createWindowContainer(g_rayWindow, this);
        layout->addWidget(m_container);
    }
}

void ParticleView::ensureRaylibInitialized()
{
    if (g_raylibInit)
        return;

    int w = width()  > 0 ? width()  : 800;
    int h = height() > 0 ? height() : 600;

    InitWindow(w, h, "GPU Particles - Raylib/Qt");
    SetTargetFPS(60);

    BeginDrawing();
    ClearBackground(BLACK);
    EndDrawing();

    void* handle = GetWindowHandle();
    g_rayWindow = QWindow::fromWinId(reinterpret_cast<WId>(handle));

    g_raylibInit = true;
    m_raylibInitialized = true;
}

void ParticleView::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);

    if (g_raylibInit) {
        SetWindowSize(width(), height());
    }
}

void ParticleView::setMotionBlurEnabled(bool enabled)
{
    m_motionBlur = enabled;
}

void ParticleView::setGlowEnabled(bool enabled)
{
    m_glow = enabled;
}

void ParticleView::setRainbowEnabled(bool enabled)
{
    m_rainbow = enabled;
}

void ParticleView::render(const std::vector<Particle>& particles, float fps)
{
    if (!g_raylibInit)
        return;

    BeginDrawing();

    if (m_motionBlur) {
        //Effet trail/motionblur
        Color fade = { 0, 0, 0, 20 }; //augmenter l'alpha pour effacer plus vite
        DrawRectangle(0, 0, GetScreenWidth(), GetScreenHeight(), fade);
    } else {
        //Mode normal : on efface completement l'ecran
        ClearBackground(BLACK);
    }

    //Dessin des particules
    for (const Particle& pt : particles) {
        Vector2 pos{ pt.x, pt.y };

        Color c;
        if (m_rainbow) {
            //Effet arc en ciel simple en fonction de la position
            float phase = (pt.x + pt.y) * 0.02f;
            unsigned char r = static_cast<unsigned char>(128.0f + 127.0f * std::sin(phase));
            unsigned char g = static_cast<unsigned char>(128.0f + 127.0f * std::sin(phase + 2.094f));  // +120°
            unsigned char b = static_cast<unsigned char>(128.0f + 127.0f * std::sin(phase + 4.188f));  // +240°
            c = { r, g, b, pt.a };
        } else {
            c = { pt.r, pt.g, pt.b, pt.a };
        }

        //Halo neon
        if (m_glow) {
            float haloRadius = pt.rad * 2.5f; //halo plus large que la particule
            DrawCircleV(pos, haloRadius, ColorAlpha(c, 0.5f));
        }

        //Noyau de la particule
        DrawCircleV(pos, pt.rad, c);
    }

    EndDrawing();
}
