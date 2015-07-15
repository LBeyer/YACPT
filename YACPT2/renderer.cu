#include "renderer.h"
#include <gl/glew.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <memory>
#include <cuda_gl_interop.h>
#include <math_constants.h>

void initCuda();
__device__ inline unsigned int wangHash(unsigned int n);
__device__ inline bool outOfBounds(unsigned int x, unsigned int y, const Resolution& resolution);
__device__ inline void applyGamma(Vec3& color, float gamma);
__device__ inline Vec3 trace(Ray ray, const Material* materials, curandState& randomState, const BVH<Mesh>& bvh);
__global__ void kernel(unsigned long long seed, uchar4* devPtr, Resolution resolution, size_t spp, float gamma, Vec3* colorBuffer, Camera& camera, const Material* materials, const BVH<Mesh>& bvh);
__device__ inline Ray diffuseReflection(const Vec3& intersectionPoint, const Vec3& surfaceNormal, curandState& randomState);
__device__ inline Ray specularReflection(const Vec3& intersectionPoint, const Vec3& surfaceNormal, const Vec3& incomingDirection);

__device__ inline unsigned int wangHash(unsigned int n)
{
	n = (n ^ 61) ^ (n >> 16);
	n = n + (n << 3);
	n = n ^ (n >> 4);
	n = n * 0x27d4eb2d;
	n = n ^ (n >> 15);
	return n;
}

__device__ inline bool outOfBounds(unsigned int x, unsigned int y, const Resolution& resolution)
{
	return x >= resolution.width || y >= resolution.height;
}

__device__ inline void applyGamma(Vec3& color, float gamma)
{
	color.r = powf(color.r, 1 / gamma);
	color.g = powf(color.g, 1 / gamma);
	color.b = powf(color.b, 1 / gamma);
}

__global__ void kernel(unsigned long long seed, uchar4* devPtr, Resolution resolution, size_t spp, float gamma, Vec3* colorBuffer, Camera& camera, const Material* materials, const BVH<Mesh>& bvh)
{
	auto x = threadIdx.x + blockIdx.x * blockDim.x;
	auto y = threadIdx.y + blockIdx.y * blockDim.y;

	if(outOfBounds(x, y, resolution))
	{
		return;
	}

	auto offset = x + (resolution.height - y - 1) * resolution.width;	// opengl textures are bottom-up
	curandState randomState;
	curand_init(seed + offset, 0, 0, &randomState);

	auto ray = camera.getRay((x + curand_uniform(&randomState)) / resolution.width, (y + curand_uniform(&randomState)) / resolution.height);
	colorBuffer[offset] += trace(ray, materials, randomState, bvh);
	auto color = (colorBuffer[offset] / spp).clamp();
	applyGamma(color, gamma);

	devPtr[offset].x = color.r * 255;
	devPtr[offset].y = color.g * 255;
	devPtr[offset].z = color.b * 255;
	devPtr[offset].w = 0;
}

__device__ inline Vec3 trace(Ray ray, const Material* materials, curandState& randomState, const BVH<Mesh>& bvh)
{
	auto color = Color::white();
	auto weight = 1.0f;
	while(true)
	{
		auto its = bvh.intersect(ray);

		if(!its)
		{
			return Color::black();
		}

		auto& material = materials[its.materialIndex];
		if(material.reflectance.brdfType == BRDFType::Emissive)
		{
			return color * material.reflectance.emissive * weight;
		}
		else if(material.reflectance.brdfType == BRDFType::Diffuse)
		{
			ray = diffuseReflection(its.point, its.surfaceNormal, randomState);
			color *= material.reflectance.diffuse;
		}
		else if(material.reflectance.brdfType == BRDFType::Mirror)
		{
			ray = specularReflection(its.point, its.surfaceNormal, ray.direction);
			color *= material.reflectance.specular;
		}
		else if(material.reflectance.brdfType == BRDFType::Glossy)
		{

		}
		else if(material.reflectance.brdfType == BRDFType::Dielectric)
		{

		}
		// russian roulette
		auto survival = 0.5f;
		if(curand_uniform(&randomState) > survival)
		{
			return Color::black();
		}
		else
		{
			weight /= survival;
		}
	}
}

__device__ inline Ray diffuseReflection(const Vec3& intersectionPoint, const Vec3& surfaceNormal, curandState& randomState)
{
	// create cosine weighted direction on unit hemisphere facing towards the z-axis
	auto u = curand_uniform(&randomState);
	auto v = curand_uniform(&randomState);
	auto phi = 2 * CUDART_PI_F * u;
	auto cosTheta = sqrtf(v);
	auto sinTheta = sqrtf(1 - v);
	Vec3 dir = {cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta};
	dir.normalize();

	// change of basis towards surfacenormal/tangent/bitangent 
	Vec3 dx0 = {0, surfaceNormal.z, -surfaceNormal.y};
	Vec3 dx1 = {-surfaceNormal.z, 0, surfaceNormal.x};
	auto dx = ((dx0.dot(dx0)) > (dx1.dot(dx1)) ? dx0 : dx1).normalize();
	auto dy = surfaceNormal.cross(dx);
	dir = dir.x * dx + dir.y * dy + dir.z * surfaceNormal;

	auto offset = 0.0001f;
	return{intersectionPoint + offset * surfaceNormal, dir};
}

__device__ inline Ray specularReflection(const Vec3& intersectionPoint, const Vec3& surfaceNormal, const Vec3& incomingDirection)
{
	Vec3 direction = {incomingDirection - (2 * incomingDirection.dot(surfaceNormal) * surfaceNormal)};
	auto offset = 0.0001f;
	return{intersectionPoint + offset * surfaceNormal, direction};
}

Renderer::Renderer(Scene& scene)
	: rng(),
	resolution(scene.settings.resolution),
	spp(0),
	meshCount(scene.world.meshes.size()),
	gamma(scene.settings.gamma),
	colorBuffer(),
	camera(),
	materials(),
	meshes(),
	bvh()
{
	initCuda();
	createColorBuffer();
	copyToDevice(scene.camera);
	copyToDevice(scene.world.materials);
	moveToDevice(scene.world.meshes);
	buildBVHs(scene.settings.bvhSettings);
}

Renderer::~Renderer()
{
	cudaDeviceSynchronize();
}

void Renderer::buildBVHs(const BVHSettings& bvhSettings)
{
	cudaDeviceSynchronize();
	for(auto& mesh : meshes)
	{
		mesh.buildBVH(bvhSettings.cTrav, bvhSettings.cIts);
	}
	bvh = makeManaged<BVH<Mesh>>(meshes.get(), static_cast<uint32_t>(meshCount), bvhSettings.cTrav, bvhSettings.cIts);
}

void Renderer::createColorBuffer()
{
	colorBuffer = makeCudaPtr<Vec3[]>(resolution.width * resolution.height);
}

void Renderer::copyToDevice(const Camera& camera)
{
	this->camera = makeCudaPtr<Camera>(camera);
}

void Renderer::copyToDevice(const std::vector<Material>& materials)
{
	Material* ptr = nullptr;
	cudaMalloc(&ptr, materials.size() * sizeof(Material));
	cudaMemcpy(ptr, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	this->materials.reset(ptr);
}

void Renderer::moveToDevice(std::vector<Mesh>& meshes)
{
	Mesh* ptr = nullptr;
	cudaMallocManaged(&ptr, meshCount * sizeof(Mesh));
	for(auto i = 0; i < meshCount; i++)
	{
		ptr[i] = std::move(meshes[i]);
	}
	this->meshes.reset(ptr, meshCount);
}

void Renderer::renderFrame(uchar4* devPtr)
{
	dim3 threads = {8, 8};
	dim3 grids = {(resolution.width + threads.x - 1) / threads.x, (resolution.height + threads.y - 1) / threads.y};
	spp++;
	std::uniform_int_distribution<unsigned long long> dist;
	auto seed = dist(rng);
	kernel<<<grids, threads>>>(seed, devPtr, resolution, spp, gamma, colorBuffer.get(), *camera.get(), materials.get(), *bvh.get());
}

void Renderer::reset()
{
	cudaDeviceSynchronize();
	createColorBuffer();
	spp = 0;
}

Resolution Renderer::getResolution() const
{
	return resolution;
}

size_t Renderer::getSpp() const
{
	return spp;
}

void initCuda()
{
	cudaDeviceProp prop;
	int dev;
	std::memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 5;
	prop.minor = 2;
	if(cudaChooseDevice(&dev, &prop) != cudaSuccess)
	{
		throw std::exception("cudaChooseDevice");
	}
	if(cudaSetDevice(dev) != cudaSuccess)
	{
		throw std::exception("cudaSetDevice");
	}
}