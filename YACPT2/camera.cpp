#include "camera.h"

#define _USE_MATH_DEFINES
#include "math.h"

float degToRad(float deg);

Camera::Camera()
	: Camera({0, 0, 0}, {0, 0, 1}, {0, 1, 0}, 30, 4.0f / 3)
{
}

Camera::Camera(const Vec3& pov, const Vec3& lookAt, const Vec3& up, float fovY, float aspectRatio)
	: pov(pov),
	viewDir(lookAt - pov),
	up(up),
	dirX(),
	dirY(),
	fovY(fovY),
	aspectRatio(aspectRatio)
{
	viewDir.normalize();
	this->up.normalize();
	auto u = viewDir.cross(up).normalize();
	auto v = u.cross(viewDir).normalize();
	dirX = -2 * tanf(degToRad(fovY) * 0.5f) * u;
	dirY = -2 * tanf(degToRad(fovY) * 0.5f) * v;
}

float degToRad(float deg)
{
	return deg * static_cast<float>(M_PI) / 180;
}