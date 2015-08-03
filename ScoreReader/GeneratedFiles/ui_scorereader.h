/********************************************************************************
** Form generated from reading UI file 'scorereader.ui'
**
** Created by: Qt User Interface Compiler version 5.5.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SCOREREADER_H
#define UI_SCOREREADER_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ScoreReaderClass
{
public:
    QAction *actionOpen;
    QAction *action_2;
    QAction *actionClose;
    QAction *actionFindNote;
    QAction *actionFineLine;
    QAction *actionFindMark;
    QAction *actionExtractMIDI;
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QLabel *originLabel;
    QLabel *resultLabel;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuProcess;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *ScoreReaderClass)
    {
        if (ScoreReaderClass->objectName().isEmpty())
            ScoreReaderClass->setObjectName(QStringLiteral("ScoreReaderClass"));
        ScoreReaderClass->resize(600, 400);
        ScoreReaderClass->setMaximumSize(QSize(1024, 768));
        actionOpen = new QAction(ScoreReaderClass);
        actionOpen->setObjectName(QStringLiteral("actionOpen"));
        action_2 = new QAction(ScoreReaderClass);
        action_2->setObjectName(QStringLiteral("action_2"));
        actionClose = new QAction(ScoreReaderClass);
        actionClose->setObjectName(QStringLiteral("actionClose"));
        actionFindNote = new QAction(ScoreReaderClass);
        actionFindNote->setObjectName(QStringLiteral("actionFindNote"));
        actionFineLine = new QAction(ScoreReaderClass);
        actionFineLine->setObjectName(QStringLiteral("actionFineLine"));
        actionFindMark = new QAction(ScoreReaderClass);
        actionFindMark->setObjectName(QStringLiteral("actionFindMark"));
        actionExtractMIDI = new QAction(ScoreReaderClass);
        actionExtractMIDI->setObjectName(QStringLiteral("actionExtractMIDI"));
        centralWidget = new QWidget(ScoreReaderClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        originLabel = new QLabel(centralWidget);
        originLabel->setObjectName(QStringLiteral("originLabel"));

        gridLayout->addWidget(originLabel, 0, 0, 1, 1);

        resultLabel = new QLabel(centralWidget);
        resultLabel->setObjectName(QStringLiteral("resultLabel"));

        gridLayout->addWidget(resultLabel, 0, 1, 1, 1);

        ScoreReaderClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(ScoreReaderClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 600, 21));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuProcess = new QMenu(menuBar);
        menuProcess->setObjectName(QStringLiteral("menuProcess"));
        ScoreReaderClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(ScoreReaderClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        ScoreReaderClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(ScoreReaderClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        ScoreReaderClass->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuProcess->menuAction());
        menuFile->addAction(actionOpen);
        menuFile->addSeparator();
        menuFile->addAction(actionClose);
        menuProcess->addAction(actionFindNote);
        menuProcess->addAction(actionFineLine);
        menuProcess->addAction(actionFindMark);
        menuProcess->addAction(actionExtractMIDI);

        retranslateUi(ScoreReaderClass);

        QMetaObject::connectSlotsByName(ScoreReaderClass);
    } // setupUi

    void retranslateUi(QMainWindow *ScoreReaderClass)
    {
        ScoreReaderClass->setWindowTitle(QApplication::translate("ScoreReaderClass", "ScoreReader", 0));
        actionOpen->setText(QApplication::translate("ScoreReaderClass", "Open images", 0));
        action_2->setText(QApplication::translate("ScoreReaderClass", "\354\242\205\353\243\214", 0));
        actionClose->setText(QApplication::translate("ScoreReaderClass", "Close", 0));
        actionFindNote->setText(QApplication::translate("ScoreReaderClass", "Find a notes", 0));
        actionFineLine->setText(QApplication::translate("ScoreReaderClass", "Find a lines", 0));
        actionFindMark->setText(QApplication::translate("ScoreReaderClass", "Find a marks", 0));
        actionExtractMIDI->setText(QApplication::translate("ScoreReaderClass", "Extract to MIDI", 0));
        originLabel->setText(QString());
        resultLabel->setText(QString());
        menuFile->setTitle(QApplication::translate("ScoreReaderClass", "File", 0));
        menuProcess->setTitle(QApplication::translate("ScoreReaderClass", "Process", 0));
    } // retranslateUi

};

namespace Ui {
    class ScoreReaderClass: public Ui_ScoreReaderClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SCOREREADER_H
