#pragma once

#include "scene.h"
#include "cudaPtr.h"
#include <random>

class Renderer
{
public:
	explicit Renderer(Scene& scene);
	~Renderer();
	void buildBVHs(const BVHSettings& bvhSettings);
	void renderFrame(uchar4* devPtr);
	void reset();
	Resolution getResolution() const;
	size_t getSpp() const;

private:
	void createColorBuffer();
	void copyToDevice(const Camera& camera);
	void copyToDevice(const std::vector<Material>& materials);
	void moveToDevice(std::vector<Mesh>& meshes);

	std::mt19937_64 rng;
	Resolution resolution;
	size_t spp, meshCount;
	float gamma;
	CudaPtr<Vec3[]> colorBuffer;
	CudaPtr<Camera> camera;
	CudaPtr<Material[]> materials;
	ManagedPtr<Mesh[]> meshes;
	ManagedPtr<BVH<Mesh>> bvh;
};