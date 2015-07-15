#include "material.h"

Material Material::diffuse(Vec3 color)
{
	return{BRDFType::Diffuse, Vec3(), color.clamp(), Vec3(), Vec3(), 0};
}

Material Material::specular(Vec3 color)
{
	return{BRDFType::Mirror, Vec3(), Vec3(), color.clamp(), Vec3(), 0};
}

Material Material::emissive(Vec3 color)
{
	return{BRDFType::Emissive, color.clamp(), Vec3(), Vec3(), Vec3(), 0};
}

Material::Material(BRDFType brdfType, const Vec3& emissive, const Vec3& diffuse, const Vec3& specular, const Vec3& refractive, float ior)
	: reflectance({brdfType, emissive, diffuse, specular, refractive, ior})
{
}