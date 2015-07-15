#include "scene.h"

Scene::Scene(Scene&& other)
	: world(std::move(other.world)),
	camera(std::move(other.camera)),
	settings(std::move(other.settings))
{
}