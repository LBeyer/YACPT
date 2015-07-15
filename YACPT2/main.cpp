#include "parser.h"
#include "qtgui.h"
#include <QtWidgets/QApplication>
#include "timer.h"


float findBestCIts(Renderer& renderer, QTGui& gui, QApplication& app, float min, float max, float step, unsigned int frameCount)
{
	auto bestCIts = min;
	auto bestTime = std::numeric_limits<long long>::max();
	BVHSettings bvhSettings = {1, 1};
	gui.startButtonPressed();
	app.processEvents();
	for(auto cIts = min; cIts < max; cIts += step)
	{
		renderer.reset();
		bvhSettings.cIts = cIts;
		renderer.buildBVHs(bvhSettings);
		Timer t;
		for(auto i = 0u; i < frameCount; i++)
		{
			gui.render();
			app.processEvents();
		}
		auto time = t.elapsedMs();
		if(time < bestTime)
		{
			bestCIts = cIts;
			bestTime = time;
		}
	}
	app;
	return bestCIts;
}

int main(int argc, char* argv[])
{
	QApplication app(argc, argv);
	auto scene(argc == 2 ? parse(argv[1]) : parse("sampleScene.txt"));
	Renderer renderer(scene);
	QTGui gui(renderer);
	gui.show();
	while(gui.isVisible())
	{
		gui.render();
		app.processEvents();
	}
}
