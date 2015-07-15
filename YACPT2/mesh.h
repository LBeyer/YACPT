#pragma once

#include "bvh.h"
#include "triangle.h"
#include "managedPtr.h"
#include <vector>

class Mesh
{
public:
	Mesh(const std::vector<Vec3>& vertices, const std::vector<Triangle>& faces);
	Mesh(const Mesh& other) = delete;
	Mesh(Mesh&& other);
	Mesh& operator=(Mesh&& other);
	Mesh& operator=(const Mesh& other) = delete;

	DEVICE inline Intersection intersect(const Ray& ray) const;
	AABB getAABB() const;
	Vec3 getMidPoint() const;
	void buildBVH(float cTrav, float cIts);
	size_t polygonCount() const;

private:
	ManagedPtr<Vec3[]> vertices;
	ManagedPtr<Triangle[]> faces;
	ManagedPtr<BVH<Triangle>> bvh;
	uint32_t vertexCount, faceCount;
};

DEVICE inline Intersection Mesh::intersect(const Ray& ray) const
{
	return bvh->intersect(ray, vertices.get());
}