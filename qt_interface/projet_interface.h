#pragma once

#include <QtWidgets/QMainWindow>
#include <QTimer>
#include <QElapsedTimer>
#include <memory>
#include <QMenu>
#include <QAction>

#include "ui_projet_interface.h"
#include "sim_world.hpp"
#include "compute.hpp"
#include "ParticleView.h"

class projet_interface : public QMainWindow
{
    Q_OBJECT

public:
    explicit projet_interface(QWidget* parent = nullptr);
    ~projet_interface();

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    //Met à jour m_params à partir des widgets Qt
    void updateSimParamsFromUi();

    //Crée le monde si necessaire
    void createWorldIfNeeded();

private:
    Ui::projet_interfaceClass ui;

    ParticleView* m_view = nullptr; //zone de rendu

    std::unique_ptr<SimWorld> m_world;
    SimParams m_params{};
    QTimer* m_timer = nullptr;
    bool m_running = false;

    //FPS
    QElapsedTimer m_fpsTimer;
    int   m_fpsFrames = 0;
    float m_fpsValue  = 0.0f;

    //Menu d'effets
    QMenu*   m_effectsMenu         = nullptr;
    QAction* m_actionMotionBlur    = nullptr;
    QAction* m_actionGlow          = nullptr;
    QAction* m_actionRainbow       = nullptr; //nouvelle action : effet arc en ciel
    QAction* m_actionCenterGravity = nullptr; //nouvelle action : gravité centrale

    //Etat de la gravité centrale
    bool m_centralGravity = false;

    //Système de jeu
    bool m_gameActive = false;
    QElapsedTimer m_gameTimer;
    int m_gameScore = 0;
    int m_comboMultiplier = 1;
    QElapsedTimer m_lastActionTimer;
    int m_actionCounter = 0;  //Compteur pour ralentir l'augmentation du combo
};
