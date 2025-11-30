#include "ParticleView.h"

#include <QVBoxLayout>
#include <QResizeEvent>
#include <QWindow>

#include "raylib.h"

//Fenetre Raylib partagée
static bool     g_raylibInit  = false;
static QWindow* g_rayWindow   = nullptr;

ParticleView::ParticleView(QWidget* parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_OpaquePaintEvent);
    setAttribute(Qt::WA_NoSystemBackground);

    //Layout pour héberger la fenêtre native Raylib
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    ensureRaylibInitialized();

    if (g_rayWindow) {
        //Crée un conteneur Qt pour la fenêtre Raylib
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

void ParticleView::render(const std::vector<Particle>& particles, float fps)
{
    if (!g_raylibInit)
        return;

    BeginDrawing();
    ClearBackground(BLACK);

    //Dessin des particules
    for (const Particle& pt : particles) {
        Color c{ pt.r, pt.g, pt.b, pt.a };
        Vector2 pos{ pt.x, pt.y };
        DrawCircleV(pos, pt.rad, c);
    }


    EndDrawing();
}
