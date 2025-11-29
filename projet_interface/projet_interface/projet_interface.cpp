#include "projet_interface.h"

projet_interface::projet_interface(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    // Slider et box nombre de particules
    
    // Même plage pour les deux
    ui.nb_particule->setRange(ui.slider_nbparticule->minimum(),
        ui.slider_nbparticule->maximum());
    // Initialisation
    ui.nb_particule->setValue(ui.slider_nbparticule->value());
    // Slider → SpinBox
    connect(ui.slider_nbparticule, &QSlider::valueChanged,
        ui.nb_particule, &QSpinBox::setValue);
    // SpinBox → Slider
    connect(ui.nb_particule, QOverload<int>::of(&QSpinBox::valueChanged),
        ui.slider_nbparticule, &QSlider::setValue);


    // Slider et box restitution

    // Même plage pour les deux
    ui.restitution->setRange(0, 200);
    ui.restitution->setRange(0.0, 2.0);
    ui.restitution->setSingleStep(0.01);
    ui.restitution->setDecimals(2);
    // Initialisation
    ui.restitution->setValue(ui.slider_restitution->value()/100.0);
    // Slider → SpinBox
    connect(ui.slider_restitution, &QSlider::valueChanged,
        this, [this](int v) {
            ui.restitution->setValue(v / 100.0);
        });
    // SpinBox → Slider
    connect(ui.restitution,
        QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        this, [this](double v) {
            ui.slider_restitution->setValue(static_cast<int>(v * 100));
        });

    // Initialisation du label "Gravité" et de la spinbox
    ui.label_gravite->setText("Gravité: ");  
    ui.spin_gravite->setRange(0.0, 20.0);    
    ui.spin_gravite->setSingleStep(0.1);

    // Connexion du bouton "Valider"
    connect(ui.btn_set, &QPushButton::clicked, this, [this]() {
        ui.label_gravite->setText("Gravité: " + QString::number(ui.spin_gravite->value()));
        ui.spin_gravite->setEnabled(false); // bloque la spinbox après validation
        });

    // Connexion du bouton "Reset"
    connect(ui.btn_reset, &QPushButton::clicked, this, [this]() {
        ui.label_gravite->setText("Gravité: ");
        ui.spin_gravite->setEnabled(true);  // réactive la spinbox
        });
}

projet_interface::~projet_interface()
{
}
