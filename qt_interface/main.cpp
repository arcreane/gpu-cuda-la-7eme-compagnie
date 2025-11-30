#include "projet_interface.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    projet_interface window;
    window.show();
    return app.exec();
}
