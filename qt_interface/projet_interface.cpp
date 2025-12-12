#include "projet_interface.h"

#include <QSlider>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QDebug>
#include <QResizeEvent>
#include <QLayout>
#include <QToolButton>
#include <QMenu>
#include <QAction>
#include <QMessageBox>
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

    //Backend par defaut : CPU
    set_backend_use_cuda(false);

    //Style moderne dark theme gaming
    this->setStyleSheet(
        "QMainWindow {"
        "   background-color: #1a1a1a;"
        "}"
        "QFrame {"
        "   background-color: #252525;"
        "   border: 2px solid #00d4ff;"
        "   border-radius: 10px;"
        "}"
        "QPushButton {"
        "   background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #00d4ff, stop:1 #0088cc);"
        "   color: white;"
        "   border: none;"
        "   border-radius: 8px;"
        "   padding: 10px;"
        "   font-weight: bold;"
        "   font-size: 12px;"
        "}"
        "QPushButton:hover {"
        "   background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #00ffff, stop:1 #00aadd);"
        "}"
        "QPushButton:pressed {"
        "   background-color: #006688;"
        "}"
        "QLabel {"
        "   color: #00d4ff;"
        "   font-weight: bold;"
        "   font-size: 11px;"
        "}"
        "#label_fps, #label_score {"
        "   color: #00d4ff;"
        "   font-weight: bold;"
        "   font-size: 11px;"
        "   border: none;"
        "   background-color: transparent;"
        "   padding: 5px;"
        "}"
        "QSlider::groove:horizontal {"
        "   background-color: #3a3a3a;"
        "   height: 8px;"
        "   border-radius: 4px;"
        "}"
        "QSlider::handle:horizontal {"
        "   background-color: #00d4ff;"
        "   border: 2px solid #00ffff;"
        "   width: 18px;"
        "   margin: -5px 0;"
        "   border-radius: 9px;"
        "}"
        "QSlider::handle:horizontal:hover {"
        "   background-color: #00ffff;"
        "   border: 2px solid #ffffff;"
        "}"
        "QSpinBox, QDoubleSpinBox {"
        "   background-color: #2a2a2a;"
        "   color: #00d4ff;"
        "   border: 2px solid #3a3a3a;"
        "   border-radius: 5px;"
        "   padding: 5px;"
        "   font-weight: bold;"
        "}"
        "QSpinBox:focus, QDoubleSpinBox:focus {"
        "   border: 2px solid #00d4ff;"
        "}"
        "QComboBox {"
        "   background-color: #2a2a2a;"
        "   color: #00d4ff;"
        "   border: 2px solid #3a3a3a;"
        "   border-radius: 5px;"
        "   padding: 5px;"
        "   font-weight: bold;"
        "}"
        "QComboBox:hover {"
        "   border: 2px solid #00d4ff;"
        "}"
        "QComboBox::drop-down {"
        "   border: none;"
        "}"
        "QComboBox QAbstractItemView {"
        "   background-color: #2a2a2a;"
        "   color: #00d4ff;"
        "   selection-background-color: #00d4ff;"
        "   selection-color: black;"
        "}"
        "QToolButton {"
        "   background-color: #2a2a2a;"
        "   color: #00d4ff;"
        "   border: 2px solid #3a3a3a;"
        "   border-radius: 5px;"
        "   padding: 5px;"
        "   font-weight: bold;"
        "}"
        "QToolButton:hover {"
        "   border: 2px solid #00d4ff;"
        "   background-color: #3a3a3a;"
        "}"
        "QMenu {"
        "   background-color: #2a2a2a;"
        "   color: #00d4ff;"
        "   border: 2px solid #00d4ff;"
        "}"
        "QMenu::item:selected {"
        "   background-color: #00d4ff;"
        "   color: black;"
        "}"
    );

    //Enleve les marges du layout central pour fit la fenetre
    if (auto* lay = ui.centralWidget->layout()) {
        lay->setContentsMargins(0, 0, 0, 0);
    }

    //============================================================
    //1) CREATION DU WIDGET DE RENDU SUR LA PARTIE DROITE
    //============================================================

    //cache l'ancien widget noir défini dans le .ui
    ui.widget->hide();

    //Vue de particules (Raylib dans un QWidget)
    m_view = new ParticleView(ui.centralWidget);

    //Geometrie initiale alignee sur la zone de droite
    int leftWidth = ui.frame->width();
    int h = ui.centralWidget->height();
    int w = ui.centralWidget->width() - leftWidth;
    if (w < 1) w = 1;
    m_view->setGeometry(leftWidth, 0, w, h);
    m_view->show();

    //Label FPS côté bandeau gauche
    ui.label_fps->setText("FPS: --");

    //Menu d'effets (motion blur, etc.)
    m_effectsMenu = new QMenu(this);

    //Action pour activer/désactiver le motion blur
    m_actionMotionBlur = m_effectsMenu->addAction("Motion blur");
    m_actionMotionBlur->setCheckable(true);
    m_actionMotionBlur->setChecked(false); //désactivé par defaut

    //Action pour activer/désactiver le halo neon
    m_actionGlow = m_effectsMenu->addAction("Halo néon");
    m_actionGlow->setCheckable(true);
    m_actionGlow->setChecked(false); //desactivé par défaut

    //Action pour activer/désactiver l'effet arc en ciel
    m_actionRainbow = m_effectsMenu->addAction("Arc-en-ciel");
    m_actionRainbow->setCheckable(true);
    m_actionRainbow->setChecked(false); //désactivé par défaut

    //Tourbillon
    m_actionCenterGravity = m_effectsMenu->addAction("Tourbillon (Gravité centrale)");
    m_actionCenterGravity->setCheckable(true);
    m_actionCenterGravity->setChecked(false); //desactivé par défaut
    m_centralGravity = m_actionCenterGravity->isChecked();

    //On associe le menu au bouton d'effets dans l'UI
    ui.toolButtonEffects->setMenu(m_effectsMenu);
    ui.toolButtonEffects->setPopupMode(QToolButton::InstantPopup);

    //Etat initial des effets côté vue
    if (m_view) {
        m_view->setMotionBlurEnabled(m_actionMotionBlur->isChecked());
        m_view->setGlowEnabled(m_actionGlow->isChecked());
        m_view->setRainbowEnabled(m_actionRainbow->isChecked());
    }

    //Quand on coche/decoche Motion blur dans le menu, on met à jour la vue
    connect(m_actionMotionBlur, &QAction::toggled,
            this, [this](bool checked) {
        if (m_view) {
            m_view->setMotionBlurEnabled(checked);
        }
    });

    //Quand on coche/decoche Halo neon dans le menu, on met à jour la vue
    connect(m_actionGlow, &QAction::toggled,
            this, [this](bool checked) {
        if (m_view) {
            m_view->setGlowEnabled(checked);
        }
    });

    //Quand on coche/decoche Arc-en-ciel, on met à jour la vue
    connect(m_actionRainbow, &QAction::toggled,
            this, [this](bool checked) {
        if (m_view) {
            m_view->setRainbowEnabled(checked);
        }
    });

    //Quand on coche/decoche Gravité centrale, on met à jour le flag
    connect(m_actionCenterGravity, &QAction::toggled,
            this, [this](bool checked) {
        m_centralGravity = checked;
    });

    //============================================================
    //1.2) COMBO BACKEND CPU/GPU
    //============================================================

    if (ui.combo_backend) {
        ui.combo_backend->clear();
        ui.combo_backend->addItem("CPU");
        ui.combo_backend->addItem("GPU");
        ui.combo_backend->setCurrentIndex(0); //CPU par defaut

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

            //On detruit l'ancien monde (qui utilisait l'ancien backend)
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

    //============================================================
    //2) CONFIGURATION DES SLIDERS/SPINBOX
    //===========================================================

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

    //Force de la souris
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

    //============================================================
    //3) INITIALISATION DES PARAMETRES
    //============================================================
    m_params.dt          = 1.0f / 60.0f;
    m_params.damping     = 0.98f;
    m_params.elasticity  = static_cast<float>(ui.restitution->value());
    m_params.range       = static_cast<float>(ui.doubleSpinBox_2->value());
    m_params.mouseForce  = 0.0f; //pas de force tant qu'on ne clique pas

    m_params.worldWidth  = static_cast<float>(GetScreenWidth());
    m_params.worldHeight = static_cast<float>(GetScreenHeight());

    //Gravité : positive = verticale, negative = gravité centrale
    {
        float g0 = static_cast<float>(ui.spin_gravite->value()) * 30.0f;
        m_params.gravity = m_centralGravity ? -g0 : g0;
    }

    //Init compteur FPS
    m_fpsTimer.start();
    m_fpsFrames = 0;
    m_fpsValue  = 0.0f;

    //============================================================
    //4) TIMER DE SIMULATION (60 Hz) + FPS + INPUT RAYLIB
    //============================================================
    m_timer = new QTimer(this);
    m_timer->setInterval(16);
    connect(m_timer, &QTimer::timeout, this, [this]() {
        if (!m_world || !m_running)
            return;

        //Mise à jour des parametres depuis l'UI
        updateSimParamsFromUi();

        //Gestion de la souris via Raylib
        Vector2 mouse = GetMousePosition();
        m_params.mouseX = mouse.x;
        m_params.mouseY = mouse.y;

        float baseForce = static_cast<float>(ui.doubleSpinBox->value());
        m_params.mouseForce = 0.0f;

        bool mouseActive = false;
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            //Attraction
            m_params.mouseForce = baseForce;
            mouseActive = true;
        } else if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
            //Repulsion
            m_params.mouseForce = -baseForce;
            mouseActive = true;
        }

        //Système de score si le jeu est actif
        if (m_gameActive && mouseActive) {
            //Compte combien de particules sont dans le rayon d'action
            int affectedParticles = 0;
            float range = m_params.range;
            float rangeSq = range * range;

            for (const auto& p : m_world->particles()) {
                float dx = p.x - m_params.mouseX;
                float dy = p.y - m_params.mouseY;
                float distSq = dx * dx + dy * dy;
                if (distSq < rangeSq) {
                    affectedParticles++;
                }
            }

            //Ajoute des points basés sur les particules affectées
            if (affectedParticles > 0) {
                m_gameScore += affectedParticles * m_comboMultiplier;

                //Augmente le compteur d'actions
                m_actionCounter++;

                //Augmente le combo seulement toutes les 10 actions
                if (m_lastActionTimer.elapsed() < 1000) {
                    if (m_actionCounter >= 10) {
                        m_comboMultiplier = std::min(m_comboMultiplier + 1, 50);
                        m_actionCounter = 0;  //Reset le compteur
                    }
                } else {
                    m_comboMultiplier = 1;
                    m_actionCounter = 0;
                }
                m_lastActionTimer.restart();
            }
        } else if (m_gameActive) {
            //Réinitialise le combo si on s'arrête trop longtemps
            if (m_lastActionTimer.elapsed() > 1000) {
                m_comboMultiplier = 1;
            }
        }

        //Step de simulation
        m_world->step(m_params);

        //Calcul FPS
        m_fpsFrames++;
        qint64 elapsedMs = m_fpsTimer.elapsed();
        if (elapsedMs >= 500) {
            m_fpsValue = static_cast<float>(m_fpsFrames) * 1000.0f / static_cast<float>(elapsedMs);
            m_fpsFrames = 0;
            m_fpsTimer.restart();

            //Mise à jour de l'affichage FPS et temps
            if (m_gameActive) {
                int timeLeft = 30 - (m_gameTimer.elapsed() / 1000);
                if (timeLeft < 0) timeLeft = 0;
                ui.label_fps->setText(
                    QString("FPS: %1 | Time: %2s").arg(m_fpsValue, 0, 'f', 1).arg(timeLeft)
                );
                ui.label_score->setText(
                    QString("Score: %1 | Combo: x%2").arg(m_gameScore).arg(m_comboMultiplier)
                );

                //Fin du jeu après 30 secondes
                if (timeLeft <= 0) {
                    m_gameActive = false;
                    m_running = false;
                    m_timer->stop();

                    QMessageBox::information(this, "Partie terminée",
                        QString("Temps écoulé !\n\nScore final: %1 points\nCombo max: x%2")
                            .arg(m_gameScore).arg(m_comboMultiplier));
                }
            } else {
                ui.label_fps->setText(
                    QString("FPS: %1").arg(m_fpsValue, 0, 'f', 1)
                );
            }
        }

        //Rendu Raylib à l'interieur du widget
        m_view->render(m_world->particles(), m_fpsValue);
    });

    //============================================================
    //5) BOUTON RESET
    //============================================================
    connect(ui.btn_reset, &QPushButton::clicked, this, [this]() {
        //Stop la simu
        m_running = false;
        m_gameActive = false;
        if (m_timer) m_timer->stop();

        //Detruit le monde
        m_world.reset();

        //Reinitialise les forces/parametres
        m_params.mouseForce = 0.0f;

        //Gravité : positive = verticale, negative = gravité centrale
        float gReset = static_cast<float>(ui.spin_gravite->value()) * 30.0f;
        m_params.gravity = m_centralGravity ? -gReset : gReset;

        //Reset FPS
        m_fpsFrames = 0;
        m_fpsValue  = 0.0f;
        m_fpsTimer.restart();
        ui.label_fps->setText("FPS: --");

        //Reset jeu
        m_gameScore = 0;
        m_comboMultiplier = 1;
        m_actionCounter = 0;
        ui.label_score->setText("Score: --");

        //Raylib dessine un ecran vide
        std::vector<Particle> empty;
        m_view->render(empty, 0.0f);
    });

    //============================================================
    //6) BOUTON PLAY (lance ou met en pause la simu)
    //============================================================
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
            ui.label_score->setText("Score: 0 | Combo: x1");

            //Démarre le jeu
            m_gameActive = true;
            m_gameScore = 0;
            m_comboMultiplier = 1;
            m_actionCounter = 0;
            m_gameTimer.start();
            m_lastActionTimer.start();

            m_timer->start();

            qDebug() << "Game started - 30 seconds!";
        } else {
            m_timer->stop();
            m_gameActive = false;
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

    //Et la vue Raylib occupe tout le reste à droite sur toute la hauteur
    int wRight = totalW - leftWidth;
    if (wRight < 1) wRight = 1;
    m_view->setGeometry(leftWidth, 0, wRight, totalH);

    //On garde les tailles de la fenetre Raylib comme reference pour le monde
    m_params.worldWidth  = static_cast<float>(GetScreenWidth());
    m_params.worldHeight = static_cast<float>(GetScreenHeight());
}

void projet_interface::updateSimParamsFromUi()
{
    m_params.dt = 1.0f / 60.0f;

    //Taille du monde = taille de la fenetre Raylib
    m_params.worldWidth  = static_cast<float>(GetScreenWidth());
    m_params.worldHeight = static_cast<float>(GetScreenHeight());

    m_params.elasticity = static_cast<float>(ui.restitution->value());
    m_params.range      = static_cast<float>(ui.doubleSpinBox_2->value());

    float friction = static_cast<float>(ui.doubleSpinBox_3->value());
    friction = std::clamp(friction, 0.0f, 0.99f);
    m_params.damping = 1.0f - friction;

    //Gravité : positive = verticale, negative = gravité centrale
    float g = static_cast<float>(ui.spin_gravite->value()) * 30.0f;
    m_params.gravity = m_centralGravity ? -g : g;
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
