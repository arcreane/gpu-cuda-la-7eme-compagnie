#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_projet_interface.h"

class projet_interface : public QMainWindow
{
    Q_OBJECT

public:
    projet_interface(QWidget *parent = nullptr);
    ~projet_interface();

private:
    Ui::projet_interfaceClass ui;
};
