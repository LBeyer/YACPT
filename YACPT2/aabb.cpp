#include "aabb.h"

Vec3 AABB::midPoint() const
{
	return (min + max) / 2;
}

float AABB::surfaceArea() const
{
	auto extent = max - min;
	return 2 * (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z);
}

AABB& AABB::operator+=(const AABB& other)
{
	min = ::min(min, other.min);
	max = ::max(max, other.max);
	return *this;
}