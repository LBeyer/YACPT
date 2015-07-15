#include "triangle.h"

Triangle::Triangle(uint32_t a, uint32_t b, uint32_t c, uint32_t materialIndex)
	: a(a),
	b(b),
	c(c),
	materialIndex(materialIndex)
{
}

AABB Triangle::getAABB(const Vec3* vertices) const
{
	return{min(min(vertices[a], vertices[b]), vertices[c]), max(max(vertices[a], vertices[b]), vertices[c])};
}

Vec3 Triangle::getMidPoint(const Vec3* vertices) const
{
	return getAABB(vertices).midPoint();
}