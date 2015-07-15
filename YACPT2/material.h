#pragma once

#include "vec3.h"

enum class BRDFType
{
	Emissive,
	Diffuse,
	Mirror,
	Glossy,
	Dielectric
};

struct Reflectance
{
	BRDFType brdfType;
	Vec3 emissive, diffuse, specular, refractive;
	float ior;
};

class Material
{
public:
	static Material diffuse(Vec3 color);
	static Material specular(Vec3 color);
	static Material emissive(Vec3 color);

	Reflectance reflectance;

private:
	Material(BRDFType brdfType, const Vec3& emissive, const Vec3& diffuse, const Vec3& specular, const Vec3& refractive, float ior);
};