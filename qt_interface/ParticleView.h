#pragma once

#include <QWidget>
#include <vector>
#include "particles_types.hpp"

//Widget Qt qui met une fenêtre Raylib
class ParticleView : public QWidget
{
    Q_OBJECT

public:
    explicit ParticleView(QWidget* parent = nullptr);

    //Appelé à chaque frame pour dessiner les particules avec Raylib
    void render(const std::vector<Particle>& particles, float fps);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    QWidget* m_container = nullptr; //QWidget qui contient la fenêtre Raylib
    bool m_raylibInitialized = false;

    void ensureRaylibInitialized();
};
