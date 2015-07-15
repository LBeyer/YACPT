#include "mesh.h"

Mesh::Mesh(const std::vector<Vec3>& vertices, const std::vector<Triangle>& faces)
	: vertices(),
	faces(),
	bvh(),
	vertexCount(static_cast<uint32_t>(vertices.size())),
	faceCount(static_cast<uint32_t>(faces.size()))
{
	Vec3* dVertices = nullptr;
	cudaMallocManaged(&dVertices, sizeof(Vec3) * vertexCount);
	cudaMemcpy(dVertices, vertices.data(), sizeof(Vec3) * vertexCount, cudaMemcpyHostToHost);
	this->vertices.reset(dVertices, vertexCount);

	Triangle* dFaces = nullptr;
	cudaMallocManaged(&dFaces, sizeof(Triangle) * faceCount);
	cudaMemcpy(dFaces, faces.data(), sizeof(Triangle) * faceCount, cudaMemcpyHostToHost);
	this->faces.reset(dFaces, faceCount);
}

Mesh::Mesh(Mesh&& other)
	: vertices(std::move(other.vertices)),
	faces(std::move(other.faces)),
	bvh(std::move(other.bvh)),
	vertexCount(other.vertexCount),
	faceCount(other.faceCount)
{
}

Mesh& Mesh::operator=(Mesh&& other)
{
	vertices = std::move(other.vertices);
	faces = std::move(other.faces);
	bvh = std::move(other.bvh);
	vertexCount = other.vertexCount;
	faceCount = other.faceCount;
	return *this;
}

AABB Mesh::getAABB() const
{
	return bvh->getAABB();
}

Vec3 Mesh::getMidPoint() const
{
	return bvh->getMidPoint();
}

void Mesh::buildBVH(float cTrav, float cIts)
{
	bvh = makeManaged<BVH<Triangle>>(faces.get(), faceCount, cTrav, cIts, vertices.get());
}

size_t Mesh::polygonCount() const
{
	return faceCount;
}