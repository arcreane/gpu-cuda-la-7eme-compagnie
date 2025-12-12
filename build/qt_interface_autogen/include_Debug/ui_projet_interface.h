/********************************************************************************
** Form generated from reading UI file 'projet_interface.ui'
**
** Created by: Qt User Interface Compiler version 6.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PROJET_INTERFACE_H
#define UI_PROJET_INTERFACE_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_projet_interfaceClass
{
public:
    QWidget *centralWidget;
    QFrame *frame;
    QLabel *label;
    QSlider *slider_nbparticule;
    QLabel *label_gravite;
    QSlider *slider_restitution;
    QPushButton *pushButton;
    QLabel *label_4;
    QLabel *label_5;
    QLabel *label_6;
    QLabel *label_7;
    QSpinBox *nb_particule;
    QDoubleSpinBox *restitution;
    QDoubleSpinBox *doubleSpinBox;
    QDoubleSpinBox *doubleSpinBox_2;
    QDoubleSpinBox *doubleSpinBox_3;
    QPushButton *btn_reset;
    QPushButton *btn_set;
    QDoubleSpinBox *spin_gravite;
    QLabel *label_fps;
    QLabel *label_score;
    QLabel *label_2;
    QComboBox *combo_backend;
    QToolButton *toolButtonEffects;
    QWidget *widget;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *projet_interfaceClass)
    {
        if (projet_interfaceClass->objectName().isEmpty())
            projet_interfaceClass->setObjectName("projet_interfaceClass");
        projet_interfaceClass->resize(824, 540);
        centralWidget = new QWidget(projet_interfaceClass);
        centralWidget->setObjectName("centralWidget");
        QSizePolicy sizePolicy(QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralWidget->sizePolicy().hasHeightForWidth());
        centralWidget->setSizePolicy(sizePolicy);
        frame = new QFrame(centralWidget);
        frame->setObjectName("frame");
        frame->setGeometry(QRect(0, 0, 251, 461));
        sizePolicy.setHeightForWidth(frame->sizePolicy().hasHeightForWidth());
        frame->setSizePolicy(sizePolicy);
        frame->setStyleSheet(QString::fromUtf8("background-color: #1a1a1d;\n"
"color: white;\n"
""));
        frame->setFrameShape(QFrame::Shape::StyledPanel);
        frame->setFrameShadow(QFrame::Shadow::Raised);
        label = new QLabel(frame);
        label->setObjectName("label");
        label->setGeometry(QRect(10, 10, 121, 16));
        slider_nbparticule = new QSlider(frame);
        slider_nbparticule->setObjectName("slider_nbparticule");
        slider_nbparticule->setGeometry(QRect(10, 30, 111, 16));
        slider_nbparticule->setMaximum(2000);
        slider_nbparticule->setOrientation(Qt::Orientation::Horizontal);
        label_gravite = new QLabel(frame);
        label_gravite->setObjectName("label_gravite");
        label_gravite->setGeometry(QRect(10, 60, 51, 16));
        slider_restitution = new QSlider(frame);
        slider_restitution->setObjectName("slider_restitution");
        slider_restitution->setGeometry(QRect(10, 120, 111, 16));
        slider_restitution->setMaximum(200);
        slider_restitution->setOrientation(Qt::Orientation::Horizontal);
        pushButton = new QPushButton(frame);
        pushButton->setObjectName("pushButton");
        pushButton->setGeometry(QRect(40, 290, 80, 24));
        label_4 = new QLabel(frame);
        label_4->setObjectName("label_4");
        label_4->setGeometry(QRect(10, 90, 131, 16));
        label_5 = new QLabel(frame);
        label_5->setObjectName("label_5");
        label_5->setGeometry(QRect(10, 160, 101, 16));
        label_6 = new QLabel(frame);
        label_6->setObjectName("label_6");
        label_6->setGeometry(QRect(10, 180, 101, 16));
        label_7 = new QLabel(frame);
        label_7->setObjectName("label_7");
        label_7->setGeometry(QRect(10, 200, 91, 16));
        nb_particule = new QSpinBox(frame);
        nb_particule->setObjectName("nb_particule");
        nb_particule->setGeometry(QRect(140, 20, 101, 31));
        restitution = new QDoubleSpinBox(frame);
        restitution->setObjectName("restitution");
        restitution->setGeometry(QRect(140, 120, 101, 31));
        doubleSpinBox = new QDoubleSpinBox(frame);
        doubleSpinBox->setObjectName("doubleSpinBox");
        doubleSpinBox->setGeometry(QRect(150, 160, 91, 22));
        doubleSpinBox_2 = new QDoubleSpinBox(frame);
        doubleSpinBox_2->setObjectName("doubleSpinBox_2");
        doubleSpinBox_2->setGeometry(QRect(150, 180, 91, 22));
        doubleSpinBox_3 = new QDoubleSpinBox(frame);
        doubleSpinBox_3->setObjectName("doubleSpinBox_3");
        doubleSpinBox_3->setGeometry(QRect(150, 200, 91, 22));
        btn_reset = new QPushButton(frame);
        btn_reset->setObjectName("btn_reset");
        btn_reset->setGeometry(QRect(130, 290, 80, 24));
        btn_set = new QPushButton(frame);
        btn_set->setObjectName("btn_set");
        btn_set->setGeometry(QRect(40, 240, 171, 24));
        spin_gravite = new QDoubleSpinBox(frame);
        spin_gravite->setObjectName("spin_gravite");
        spin_gravite->setGeometry(QRect(140, 60, 91, 22));
        spin_gravite->setMaximum(150.990000000000009);
        spin_gravite->setSingleStep(1.000000000000000);
        spin_gravite->setValue(30.000000000000000);
        label_fps = new QLabel(frame);
        label_fps->setObjectName("label_fps");
        label_fps->setGeometry(QRect(10, 400, 380, 25));
        QFont font;
        font.setPointSize(8);
        font.setBold(true);
        label_fps->setFont(font);
        label_score = new QLabel(frame);
        label_score->setObjectName("label_score");
        label_score->setGeometry(QRect(10, 425, 380, 25));
        label_score->setFont(font);
        label_2 = new QLabel(frame);
        label_2->setObjectName("label_2");
        label_2->setGeometry(QRect(20, 360, 49, 16));
        combo_backend = new QComboBox(frame);
        combo_backend->setObjectName("combo_backend");
        combo_backend->setGeometry(QRect(90, 360, 101, 22));
        toolButtonEffects = new QToolButton(frame);
        toolButtonEffects->setObjectName("toolButtonEffects");
        toolButtonEffects->setGeometry(QRect(210, 360, 21, 22));
        toolButtonEffects->setPopupMode(QToolButton::ToolButtonPopupMode::InstantPopup);
        widget = new QWidget(centralWidget);
        widget->setObjectName("widget");
        widget->setGeometry(QRect(250, 0, 571, 461));
        sizePolicy.setHeightForWidth(widget->sizePolicy().hasHeightForWidth());
        widget->setSizePolicy(sizePolicy);
        widget->setStyleSheet(QString::fromUtf8("background-color: black;\n"
""));
        projet_interfaceClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(projet_interfaceClass);
        menuBar->setObjectName("menuBar");
        menuBar->setGeometry(QRect(0, 0, 824, 22));
        projet_interfaceClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(projet_interfaceClass);
        mainToolBar->setObjectName("mainToolBar");
        projet_interfaceClass->addToolBar(Qt::ToolBarArea::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(projet_interfaceClass);
        statusBar->setObjectName("statusBar");
        projet_interfaceClass->setStatusBar(statusBar);

        retranslateUi(projet_interfaceClass);

        QMetaObject::connectSlotsByName(projet_interfaceClass);
    } // setupUi

    void retranslateUi(QMainWindow *projet_interfaceClass)
    {
        projet_interfaceClass->setWindowTitle(QCoreApplication::translate("projet_interfaceClass", "projet_interface", nullptr));
        label->setText(QCoreApplication::translate("projet_interfaceClass", "Nombre de particules", nullptr));
        label_gravite->setText(QCoreApplication::translate("projet_interfaceClass", "Gravit\303\251", nullptr));
        pushButton->setText(QCoreApplication::translate("projet_interfaceClass", "Play", nullptr));
        label_4->setText(QCoreApplication::translate("projet_interfaceClass", "Coefficient de restitution", nullptr));
        label_5->setText(QCoreApplication::translate("projet_interfaceClass", "Force d'exposition", nullptr));
        label_6->setText(QCoreApplication::translate("projet_interfaceClass", "Rayon d'action", nullptr));
        label_7->setText(QCoreApplication::translate("projet_interfaceClass", "Friction de l'air", nullptr));
        btn_reset->setText(QCoreApplication::translate("projet_interfaceClass", "Reset", nullptr));
        btn_set->setText(QCoreApplication::translate("projet_interfaceClass", "Valider", nullptr));
        label_fps->setText(QCoreApplication::translate("projet_interfaceClass", "FPS : --", nullptr));
        label_score->setText(QCoreApplication::translate("projet_interfaceClass", "Score: --", nullptr));
        label_2->setText(QCoreApplication::translate("projet_interfaceClass", "Backend :", nullptr));
        toolButtonEffects->setText(QCoreApplication::translate("projet_interfaceClass", "...", nullptr));
    } // retranslateUi

};

namespace Ui {
    class projet_interfaceClass: public Ui_projet_interfaceClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PROJET_INTERFACE_H
