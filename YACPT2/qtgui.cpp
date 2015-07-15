#include "qtgui.h"
#include <QFileDialog>

QTGui::QTGui(Renderer& renderer, QWidget *parent)
	: QMainWindow(parent),
	texture(nullptr),
	renderer(renderer),
	running(false)
{
	ui.setupUi(this);
	ui.openGLWidget->setFixedSize(renderer.getResolution().width, renderer.getResolution().height);
	ui.centralWidget->adjustSize();
	setWindowFlags(windowFlags() ^ Qt::WindowMaximizeButtonHint ^ Qt::MSWindowsFixedSizeDialogHint);
}

QTGui::~QTGui()
{
}

void QTGui::show()
{
	QMainWindow::show();
	initializeGL();
}

void QTGui::render()
{
	if(running)
	{
		renderer.renderFrame(texture->map());
		texture->unmap();
		auto resolution = renderer.getResolution();
		ui.openGLWidget->makeCurrent();
		glDrawPixels(resolution.width, resolution.height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		ui.openGLWidget->update();
		ui.sppDisplay->setText(QString::number(renderer.getSpp()));
	}
}

void QTGui::initializeGL()
{
	initializeOpenGLFunctions();
	ui.openGLWidget->makeCurrent();
	glewInit();
	texture.reset(new Texture(renderer.getResolution()));
}

void QTGui::setTitle(const std::string& title)
{
	setWindowTitle(QString::fromStdString(title));
}

void QTGui::startButtonPressed()
{
	running = !running;
	if(running)
	{
		ui.startButton->setText("stop");
	}
	else
	{
		ui.startButton->setText("start");
	}
}

void QTGui::saveButtonPressed()
{
	auto filename = QFileDialog::getSaveFileName(this, "Save Image", "C://", "Bitmap (*.bmp)");
	texture->save(filename.toStdString());
}