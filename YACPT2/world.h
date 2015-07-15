#pragma once

#include "meshable.h"
#include "material.h"
#include <vector>

class World
{
public:
	World();
	World(const World& other) = delete;
	World(World&& other);

	void add(const Material& material);
	void add(const Meshable& mershable);
	void add(Mesh&& mesh);
	
	std::vector<Material> materials;
	std::vector<Mesh> meshes;
};