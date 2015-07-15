#pragma once

#include "vec3.h"
#include "ray.h"
#include "cuda_runtime.h"

class AABB
{
public:
	Vec3 midPoint() const;
	float surfaceArea() const;
	DEVICE inline bool intersect(const Ray& ray) const;

	AABB& operator+=(const AABB& other);

	Vec3 min, max;
};

DEVICE inline bool AABB::intersect(const Ray& ray) const
{
	const auto t1 = (min.x - ray.origin.x) / ray.direction.x;
	const auto t2 = (max.x - ray.origin.x) / ray.direction.x;
	const auto t3 = (min.y - ray.origin.y) / ray.direction.y;
	const auto t4 = (max.y - ray.origin.y) / ray.direction.y;
	const auto t5 = (min.z - ray.origin.z) / ray.direction.z;
	const auto t6 = (max.z - ray.origin.z) / ray.direction.z;

	auto tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
	auto tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

	return tmin <= tmax && tmax >= 0;
}