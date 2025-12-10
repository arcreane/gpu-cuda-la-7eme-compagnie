#pragma once

#include <QWidget>
#include <vector>
#include "particles_types.hpp"

//Widget Qt qui met une fenetre Raylib
class ParticleView : public QWidget
{
    Q_OBJECT

public:
    explicit ParticleView(QWidget* parent = nullptr);

    //Appelé à chaque frame pour dessiner les particules avec Raylib
    void render(const std::vector<Particle>& particles, float fps);

    //Active/desactive l'effet de motion blur
    void setMotionBlurEnabled(bool enabled);

    //Active/desactive l'effet de halo neon
    void setGlowEnabled(bool enabled);

    //Active/desactive l'effet d'arc en ciel
    void setRainbowEnabled(bool enabled);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    QWidget* m_container = nullptr; //QWidget qui contient la fenetre Raylib
    bool m_raylibInitialized = false;
    bool m_motionBlur = true;  //desactivé par defaut
    bool m_glow       = false; //halo néon desactivé par defaut
    bool m_rainbow    = false; //effet arc-en-ciel desactivé par defaut

    void ensureRaylibInitialized();
};
