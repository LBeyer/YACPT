#pragma once

#include "ray.h"

class Camera
{
public:
	Camera();
	Camera(const Vec3& pov, const Vec3& lookAt, const Vec3& up, float fovY, float aspectRatio);

	DEVICE inline Ray getRay(float x, float y) const;

private:
	Vec3 pov, viewDir, up, dirX, dirY;
	float fovY, aspectRatio;
};


DEVICE inline Ray Camera::getRay(const float x, const float y) const
{
	return{pov, (viewDir + dirX * aspectRatio * (x - 0.5f) + dirY * (y - 0.5f)).normalize()};
}