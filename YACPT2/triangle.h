#pragma once

#include "util.h"
#include "ray.h"
#include "intersection.h"
#include "aabb.h"

class Triangle
{
public:
	Triangle(uint32_t a, uint32_t b, uint32_t c, uint32_t materialIndex);
	DEVICE inline Intersection intersect(const Ray& ray, const Vec3* vertices) const;
	AABB getAABB(const Vec3* vertices) const;
	Vec3 getMidPoint(const Vec3* vertices) const;

	uint32_t a, b, c, materialIndex;
};

//http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
DEVICE inline Intersection Triangle::intersect(const Ray& ray, const Vec3* vertices) const
{
	Intersection its;
	auto e1 = vertices[b] - vertices[a];
	auto e2 = vertices[c] - vertices[a];
	auto P = ray.direction.cross(e2);
	auto det = e1.dot(P);
	auto epsilon = 0.00001f;
	if(det > -epsilon && det < epsilon)
	{
		return its;
	}
	auto inv_det = 1.0f / det;
	auto T = ray.origin - vertices[a];
	auto u = T.dot(P) * inv_det;
	if(u < 0 || u > 1)
	{
		return its;
	}
	auto Q = T.cross(e1);
	auto v = ray.direction.dot(Q) * inv_det;
	if(v < 0 || u + v > 1)
	{
		return its;
	}
	auto t = e2.dot(Q) * inv_det;
	if(t > epsilon)
	{
		its.point = ray.origin + t * ray.direction;
		its.surfaceNormal = e1.cross(e2).normalize();
		if(its.surfaceNormal.dot(ray.direction) > 0)
		{
			its.surfaceNormal *= -1;
		}
		its.doesIntersect = true;
		its.materialIndex = materialIndex;
	}
	return its;
}