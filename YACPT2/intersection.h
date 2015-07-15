#pragma once

#include "util.h"
#include "vec3.h"
#include <cstdint>

class Intersection
{
public:
	DEVICE inline Intersection();
	DEVICE inline operator bool() const;

	Vec3 point, surfaceNormal;
	uint32_t materialIndex;
	bool doesIntersect;
};

DEVICE inline Intersection::Intersection()
	: point(),
	surfaceNormal(),
	materialIndex(),
	doesIntersect(false)
{
}

DEVICE inline Intersection::operator bool() const
{
	return doesIntersect;
}