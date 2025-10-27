#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QTProjectAppli_7ecompagnie.h"

class QTProjectAppli_7ecompagnie : public QMainWindow
{
    Q_OBJECT

public:
    QTProjectAppli_7ecompagnie(QWidget *parent = nullptr);
    ~QTProjectAppli_7ecompagnie();

private:
    Ui::QTProjectAppli_7ecompagnieClass ui;
};

