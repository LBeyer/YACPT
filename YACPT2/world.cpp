#include "world.h"

World::World()
	: materials(),
	meshes()
{
}

World::World(World&& other)
	: materials(std::move(other.materials)),
	meshes(std::move(other.meshes))
{
}

void World::add(const Material& material)
{
	materials.push_back(material);
}

void World::add(const Meshable& meshable)
{
	add(meshable.meshify());
}

void World::add(Mesh&& mesh)
{
	meshes.push_back(std::move(mesh));
}