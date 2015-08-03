#include "scorereader.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ScoreReader w;
    w.show();
    return a.exec();
}
