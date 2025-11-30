#include "projet_interface.h"

#include <QSlider>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QDebug>
#include <QResizeEvent>
#include <QLayout>
#include <algorithm>
#include <vector>

#include "raylib.h"
#include "sim_world.hpp"
#include "compute.hpp"
#include "ParticleView.h"

projet_interface::projet_interface(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    //Backend par défaut : CPU
    set_backend_use_cuda(false);

    //Enleve les marges du layout central pour dviter les bandes grises
    if (auto* lay = ui.centralWidget->layout()) {
        lay->setContentsMargins(0, 0, 0, 0);
    }

    // ============================================================
    // 1) CREATION DU WIDGET DE RENDU SUR LA PARTIE DROITE
    // ============================================================

    //cache l'ancien widget noir défini dans le .ui
    ui.widget->hide();

    //Vue de par
    m_view = new ParticleView(ui.centralWidget);

    //Geométrie initiale alignée sur la zone de droite
    int leftWidth = ui.frame->width();
    int h = ui.centralWidget->height();
    int w = ui.centralWidget->width() - leftWidth;
    if (w < 1) w = 1;
    m_view->setGeometry(leftWidth, 0, w, h);
    m_view->show();

    //Label FPS côté bandeau gauche
    ui.label_fps->setText("FPS: --");

    // ============================================================
    // 1.2) COMBO BACKEND CPU / GPU
    // ============================================================

    if (ui.combo_backend) {
        ui.combo_backend->clear();
        ui.combo_backend->addItem("CPU");
        ui.combo_backend->addItem("GPU");
        ui.combo_backend->setCurrentIndex(0); //CPU par défaut

        connect(ui.combo_backend, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this](int index) {
            //CPU = 0, GPU = 1
            bool useCuda = (index == 1);
            set_backend_use_cuda(useCuda);

            qDebug() << "Backend switched to" << (useCuda ? "CUDA" : "CPU");

            //On arrete la simu si elle tournait
            bool wasRunning = m_running;
            m_running = false;
            if (m_timer) m_timer->stop();

            //On détruit l'ancien monde (qui utilisait l'ancien backend)
            m_world.reset();

            //Reset FPS
            m_fpsFrames = 0;
            m_fpsValue  = 0.0f;
            m_fpsTimer.restart();
            ui.label_fps->setText("FPS: --");

            //Ecran vide pour Raylib (pas de particules)
            std::vector<Particle> empty;
            if (m_view) {
                m_view->render(empty, 0.0f);
            }

            //On recrée un monde avec le nouveau backend (si N > 0)
            createWorldIfNeeded();

            //On relance la simu si elle était active
            if (wasRunning && m_world) {
                m_running = true;
                if (m_timer) m_timer->start();
            }
        });
    }

    // ============================================================
    // 2) CONFIGURATION DES SLIDERS / SPINBOX
    // ============================================================

    //Nombre de particules
    ui.slider_nbparticule->setRange(0, 20000);
    ui.nb_particule->setRange(0, 20000);
    ui.nb_particule->setValue(1000);
    ui.slider_nbparticule->setValue(1000);

    connect(ui.slider_nbparticule, &QSlider::valueChanged,
            ui.nb_particule, &QSpinBox::setValue);
    connect(ui.nb_particule, QOverload<int>::of(&QSpinBox::valueChanged),
            ui.slider_nbparticule, &QSlider::setValue);

    //Coefficient de restitution (0 -> 1)
    ui.slider_restitution->setRange(0, 100);
    ui.restitution->setRange(0.0, 1.0);
    ui.restitution->setDecimals(2);
    ui.restitution->setSingleStep(0.01);
    ui.slider_restitution->setValue(100);
    ui.restitution->setValue(1.0);

    connect(ui.slider_restitution, &QSlider::valueChanged,
            this, [this](int v) {
                ui.restitution->setValue(v / 100.0);
            });
    connect(ui.restitution,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double v) {
                ui.slider_restitution->setValue(static_cast<int>(v * 100));
            });

    //Force d'exposition (puissance de la souris)
    ui.doubleSpinBox->setRange(0.0, 2000.0);
    ui.doubleSpinBox->setDecimals(2);
    ui.doubleSpinBox->setSingleStep(10.0);
    ui.doubleSpinBox->setValue(800.0);

    //Rayon d'action
    ui.doubleSpinBox_2->setRange(0.0, 500.0);
    ui.doubleSpinBox_2->setDecimals(1);
    ui.doubleSpinBox_2->setSingleStep(5.0);
    ui.doubleSpinBox_2->setValue(150.0);

    //Friction de l'air
    ui.doubleSpinBox_3->setRange(0.0, 0.99);
    ui.doubleSpinBox_3->setDecimals(3);
    ui.doubleSpinBox_3->setSingleStep(0.01);
    ui.doubleSpinBox_3->setValue(0.02);

    //Gravité
    ui.label_gravite->setText("Gravité: ");
    ui.spin_gravite->setRange(0.0, 20.0);
    ui.spin_gravite->setSingleStep(0.1);
    ui.spin_gravite->setValue(0.0);

    // ============================================================
    // 3) INITIALISATION DES PARAMÈTRES
    // ============================================================
    m_params.dt          = 1.0f / 60.0f;
    m_params.damping     = 0.98f;
    m_params.elasticity  = static_cast<float>(ui.restitution->value());
    m_params.range       = static_cast<float>(ui.doubleSpinBox_2->value());
    m_params.mouseForce  = 0.0f; //pas de force tant qu'on ne clique pas

    m_params.worldWidth  = static_cast<float>(GetScreenWidth());
    m_params.worldHeight = static_cast<float>(GetScreenHeight());

    m_params.gravity     = static_cast<float>(ui.spin_gravite->value());

    //Init compteur FPS
    m_fpsTimer.start();
    m_fpsFrames = 0;
    m_fpsValue  = 0.0f;

    // ============================================================
    // 4) TIMER DE SIMULATION (~60 Hz) + FPS + INPUT RAYLIB
    // ============================================================
    m_timer = new QTimer(this);
    m_timer->setInterval(16);
    connect(m_timer, &QTimer::timeout, this, [this]() {
        if (!m_world || !m_running)
            return;

        // --- Mise à jour des paramètres depuis l'UI ---
        updateSimParamsFromUi();

        // --- Gestion de la souris via Raylib ---
        Vector2 mouse = GetMousePosition();
        m_params.mouseX = mouse.x;
        m_params.mouseY = mouse.y;

        float baseForce = static_cast<float>(ui.doubleSpinBox->value());
        m_params.mouseForce = 0.0f;

        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            //Attraction
            m_params.mouseForce = baseForce;
        } else if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
            //Repulsion
            m_params.mouseForce = -baseForce;
        }

        // --- Step de simulation ---
        m_world->step(m_params);

        // --- Calcul FPS ---
        m_fpsFrames++;
        qint64 elapsedMs = m_fpsTimer.elapsed();
        if (elapsedMs >= 500) {
            m_fpsValue = static_cast<float>(m_fpsFrames) * 1000.0f / static_cast<float>(elapsedMs);
            m_fpsFrames = 0;
            m_fpsTimer.restart();

            //Mise à jour de l'affichage FPS sur le bandeau gauche
            ui.label_fps->setText(
                QString("FPS: %1").arg(m_fpsValue, 0, 'f', 1)
            );
        }

        // --- Rendu Raylib à l'intérieur du widget ---
        m_view->render(m_world->particles(), m_fpsValue);
    });

    // ============================================================
    // 5) BOUTON "RESET"
    // ============================================================
    connect(ui.btn_reset, &QPushButton::clicked, this, [this]() {
        //Stoppe la simu
        m_running = false;
        if (m_timer) m_timer->stop();

        //Detruit le monde
        m_world.reset();

        //Reinitialise les forces/paramètres
        m_params.mouseForce = 0.0f;

        //On laisse la gravité contrôlée par le spinbox
        m_params.gravity = static_cast<float>(ui.spin_gravite->value());

        //Reset FPS
        m_fpsFrames = 0;
        m_fpsValue  = 0.0f;
        m_fpsTimer.restart();
        ui.label_fps->setText("FPS: --");

        //Raylib dessine un écran vide
        std::vector<Particle> empty;
        m_view->render(empty, 0.0f);
    });

    // ============================================================
    // 6) BOUTON "PLAY" (lance ou met en pause la simu)
    // ============================================================
    connect(ui.pushButton, &QPushButton::clicked, this, [this]() {
        if (!m_world)
            createWorldIfNeeded();
        if (!m_world)
            return;

        m_running = !m_running;

        if (m_running) {
            updateSimParamsFromUi();
            m_fpsFrames = 0;
            m_fpsTimer.restart();
            ui.label_fps->setText("FPS: --");
            m_timer->start();
            qDebug() << "Simulation started";
        } else {
            m_timer->stop();
            qDebug() << "Simulation paused";
        }
    });
}

projet_interface::~projet_interface()
{
}

void projet_interface::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);

    if (!m_view)
        return;

    //Taille totale disponible dans le centralWidget
    int totalW = ui.centralWidget->width();
    int totalH = ui.centralWidget->height();

    //Largeur actuelle du panneau de gauche
    int leftWidth = ui.frame->width();

    //On force le frame à occuper toute la hauteur sur la gauche
    ui.frame->setGeometry(0, 0, leftWidth, totalH);

    //Et la vue Raylib occupe tout le reste à droite, sur toute la hauteur
    int wRight = totalW - leftWidth;
    if (wRight < 1) wRight = 1;
    m_view->setGeometry(leftWidth, 0, wRight, totalH);

    //On garde les tailles de la fenêtre Raylib comme référence pour le monde
    m_params.worldWidth  = static_cast<float>(GetScreenWidth());
    m_params.worldHeight = static_cast<float>(GetScreenHeight());
}

void projet_interface::updateSimParamsFromUi()
{
    m_params.dt = 1.0f / 60.0f;

    // Taille du monde = taille de la fenêtre Raylib
    m_params.worldWidth  = static_cast<float>(GetScreenWidth());
    m_params.worldHeight = static_cast<float>(GetScreenHeight());

    m_params.elasticity = static_cast<float>(ui.restitution->value());
    m_params.range      = static_cast<float>(ui.doubleSpinBox_2->value());

    float friction = static_cast<float>(ui.doubleSpinBox_3->value());
    friction = std::clamp(friction, 0.0f, 0.99f);
    m_params.damping = 1.0f - friction;

    m_params.gravity  = static_cast<float>(ui.spin_gravite->value()) * 30.0f;
}

void projet_interface::createWorldIfNeeded()
{
    const int N = ui.nb_particule->value();
    if (N <= 0)
        return;

    float w = static_cast<float>(GetScreenWidth());
    float h = static_cast<float>(GetScreenHeight());

    m_world = std::make_unique<SimWorld>(
        static_cast<std::size_t>(N),
        w,
        h
    );

    m_world->randomInit();

    qDebug() << "World created with" << N << "particles";
}
