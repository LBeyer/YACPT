/********************************************************************************
** Form generated from reading UI file 'qtgui.ui'
**
** Created by: Qt User Interface Compiler version 5.4.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTGUI_H
#define UI_QTGUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QOpenGLWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QTGuiClass
{
public:
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout;
    QOpenGLWidget *openGLWidget;
    QVBoxLayout *verticalLayout;
    QPushButton *startButton;
    QPushButton *saveButton;
    QPushButton *pushButton;
    QSpacerItem *verticalSpacer;
    QLabel *sppDisplay;
    QMenuBar *menuBar;
    QMenu *menuMenu;

    void setupUi(QMainWindow *QTGuiClass)
    {
        if (QTGuiClass->objectName().isEmpty())
            QTGuiClass->setObjectName(QStringLiteral("QTGuiClass"));
        QTGuiClass->resize(689, 519);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(QTGuiClass->sizePolicy().hasHeightForWidth());
        QTGuiClass->setSizePolicy(sizePolicy);
        centralWidget = new QWidget(QTGuiClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(centralWidget->sizePolicy().hasHeightForWidth());
        centralWidget->setSizePolicy(sizePolicy1);
        horizontalLayout = new QHBoxLayout(centralWidget);
        horizontalLayout->setSpacing(0);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setSizeConstraint(QLayout::SetDefaultConstraint);
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        openGLWidget = new QOpenGLWidget(centralWidget);
        openGLWidget->setObjectName(QStringLiteral("openGLWidget"));
        sizePolicy1.setHeightForWidth(openGLWidget->sizePolicy().hasHeightForWidth());
        openGLWidget->setSizePolicy(sizePolicy1);
        openGLWidget->setMinimumSize(QSize(300, 200));

        horizontalLayout->addWidget(openGLWidget);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        startButton = new QPushButton(centralWidget);
        startButton->setObjectName(QStringLiteral("startButton"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(startButton->sizePolicy().hasHeightForWidth());
        startButton->setSizePolicy(sizePolicy2);
        startButton->setLayoutDirection(Qt::LeftToRight);

        verticalLayout->addWidget(startButton);

        saveButton = new QPushButton(centralWidget);
        saveButton->setObjectName(QStringLiteral("saveButton"));
        sizePolicy2.setHeightForWidth(saveButton->sizePolicy().hasHeightForWidth());
        saveButton->setSizePolicy(sizePolicy2);

        verticalLayout->addWidget(saveButton);

        pushButton = new QPushButton(centralWidget);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        sizePolicy2.setHeightForWidth(pushButton->sizePolicy().hasHeightForWidth());
        pushButton->setSizePolicy(sizePolicy2);
        pushButton->setLayoutDirection(Qt::LeftToRight);

        verticalLayout->addWidget(pushButton);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);

        sppDisplay = new QLabel(centralWidget);
        sppDisplay->setObjectName(QStringLiteral("sppDisplay"));
        sppDisplay->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        verticalLayout->addWidget(sppDisplay);


        horizontalLayout->addLayout(verticalLayout);

        QTGuiClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(QTGuiClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 689, 21));
        sizePolicy2.setHeightForWidth(menuBar->sizePolicy().hasHeightForWidth());
        menuBar->setSizePolicy(sizePolicy2);
        menuMenu = new QMenu(menuBar);
        menuMenu->setObjectName(QStringLiteral("menuMenu"));
        QTGuiClass->setMenuBar(menuBar);

        menuBar->addAction(menuMenu->menuAction());

        retranslateUi(QTGuiClass);
        QObject::connect(startButton, SIGNAL(clicked()), QTGuiClass, SLOT(startButtonPressed()));
        QObject::connect(saveButton, SIGNAL(clicked()), QTGuiClass, SLOT(saveButtonPressed()));

        QMetaObject::connectSlotsByName(QTGuiClass);
    } // setupUi

    void retranslateUi(QMainWindow *QTGuiClass)
    {
        QTGuiClass->setWindowTitle(QApplication::translate("QTGuiClass", "QTGui", 0));
        startButton->setText(QApplication::translate("QTGuiClass", "start", 0));
        saveButton->setText(QApplication::translate("QTGuiClass", "save", 0));
        pushButton->setText(QApplication::translate("QTGuiClass", "PushButton", 0));
        sppDisplay->setText(QApplication::translate("QTGuiClass", "0", 0));
        menuMenu->setTitle(QApplication::translate("QTGuiClass", "menu", 0));
    } // retranslateUi

};

namespace Ui {
    class QTGuiClass: public Ui_QTGuiClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTGUI_H
