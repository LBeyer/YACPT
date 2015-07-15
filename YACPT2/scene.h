#pragma once

#include "settings.h"
#include "camera.h"
#include "world.h"

class Scene
{
public:
	Scene() = default;
	Scene(const Scene& other) = delete;
	Scene(Scene&& other);

	World world;
	Camera camera;
	Settings settings;
};