#pragma once

#include "texture.h"
#include <QtWidgets/QMainWindow>
#include <QtOpenGL/QtOpenGL>
#include "renderer.h"
#include "ui_qtgui.h"

class QTGui : public QMainWindow, protected QOpenGLFunctions
{
	Q_OBJECT

public:
	explicit QTGui(Renderer& renderer, QWidget *parent = nullptr);
	~QTGui();

	void show();
	void render();
	void setTitle(const std::string& title);

public slots:
	void startButtonPressed();
	void saveButtonPressed();

private:
	void initializeGL();

	Ui::QTGuiClass ui;
	std::unique_ptr<Texture> texture;
	Renderer& renderer;
	bool running;
};