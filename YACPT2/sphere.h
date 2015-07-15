#pragma once

#include "meshable.h"

class Sphere : public Meshable
{
public:
	Sphere(const Vec3& position, float radius, uint32_t materialIndex, unsigned int refinementSteps);
	Mesh meshify() const override;

private:
	Vec3 position;
	float radius;
	uint32_t materialIndex;
	unsigned int refinementSteps;
};