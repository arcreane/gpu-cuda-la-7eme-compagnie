/********************************************************************************
** Form generated from reading UI file 'QTProjectAppli_7ecompagnie.ui'
**
** Created by: Qt User Interface Compiler version 6.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTPROJECTAPPLI_7ECOMPAGNIE_H
#define UI_QTPROJECTAPPLI_7ECOMPAGNIE_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QTProjectAppli_7ecompagnieClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *QTProjectAppli_7ecompagnieClass)
    {
        if (QTProjectAppli_7ecompagnieClass->objectName().isEmpty())
            QTProjectAppli_7ecompagnieClass->setObjectName("QTProjectAppli_7ecompagnieClass");
        QTProjectAppli_7ecompagnieClass->resize(600, 400);
        menuBar = new QMenuBar(QTProjectAppli_7ecompagnieClass);
        menuBar->setObjectName("menuBar");
        QTProjectAppli_7ecompagnieClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(QTProjectAppli_7ecompagnieClass);
        mainToolBar->setObjectName("mainToolBar");
        QTProjectAppli_7ecompagnieClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(QTProjectAppli_7ecompagnieClass);
        centralWidget->setObjectName("centralWidget");
        QTProjectAppli_7ecompagnieClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(QTProjectAppli_7ecompagnieClass);
        statusBar->setObjectName("statusBar");
        QTProjectAppli_7ecompagnieClass->setStatusBar(statusBar);

        retranslateUi(QTProjectAppli_7ecompagnieClass);

        QMetaObject::connectSlotsByName(QTProjectAppli_7ecompagnieClass);
    } // setupUi

    void retranslateUi(QMainWindow *QTProjectAppli_7ecompagnieClass)
    {
        QTProjectAppli_7ecompagnieClass->setWindowTitle(QCoreApplication::translate("QTProjectAppli_7ecompagnieClass", "QTProjectAppli_7ecompagnie", nullptr));
    } // retranslateUi

};

namespace Ui {
    class QTProjectAppli_7ecompagnieClass: public Ui_QTProjectAppli_7ecompagnieClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTPROJECTAPPLI_7ECOMPAGNIE_H
