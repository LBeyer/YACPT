#pragma once

#include "util.h"
#include <assert.h>
#include <limits>

#ifndef __CUDACC__
#include <iosfwd>
#include <algorithm>
#else

#endif

class Vec3
{
public:
	HOST DEVICE inline Vec3();
	HOST DEVICE inline Vec3(float x, float y, float z);

	HOST DEVICE inline Vec3& clamp(float min = 0.0f, float max = 1.0f);
	HOST DEVICE inline Vec3& normalize();
	HOST DEVICE inline Vec3 normalized() const;
	HOST DEVICE inline float length() const;
	HOST DEVICE inline float sqrLength() const;
	HOST DEVICE inline float dot(const Vec3& v) const;
	HOST DEVICE inline Vec3 cross(const Vec3& v) const;
	HOST DEVICE inline float sum() const;
	HOST DEVICE inline float min() const;
	HOST DEVICE inline float max() const;
	HOST DEVICE inline Vec3 operator-() const;
	HOST DEVICE inline Vec3& operator+=(const Vec3& v);
	HOST DEVICE inline Vec3& operator-=(const Vec3& v);
	HOST DEVICE inline Vec3& operator*=(const Vec3& v);
	HOST DEVICE inline Vec3& operator*=(float f);
	HOST DEVICE inline Vec3& operator/=(float f);
	HOST DEVICE inline float& operator[](size_t index);
	HOST DEVICE inline const float& operator[](size_t index) const;
	HOST DEVICE inline float* begin();
	HOST DEVICE inline const float* begin() const;
	HOST DEVICE inline float* end();
	HOST DEVICE inline const float* end() const;

	HOST DEVICE inline static Vec3 black();
	HOST DEVICE inline static Vec3 white();
	HOST DEVICE inline static Vec3 biggest();
	HOST DEVICE inline static Vec3 smallest();

	union
	{
		float x, r;
	};
	union
	{
		float y, g;
	};
	union
	{
		float z, b;
	};
};

HOST DEVICE inline Vec3 operator+(const Vec3& lhs, const Vec3& rhs);
HOST DEVICE inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs);
HOST DEVICE inline Vec3 operator*(const Vec3& lhs, float rhs);
HOST DEVICE inline Vec3 operator*(const Vec3& lhs, const Vec3& rhs);
HOST DEVICE inline Vec3 operator*(float lhs, const Vec3& rhs);
HOST DEVICE inline Vec3 operator/(const Vec3& lhs, float rhs);
HOST DEVICE inline Vec3 min(const Vec3& v1, const Vec3& v2);
HOST DEVICE inline Vec3 max(const Vec3& v1, const Vec3& v2);
#ifndef __CUDACC__
std::ostream& operator<<(std::ostream& ostr, const Vec3& v);
std::istream& operator>>(std::istream& istr, Vec3& v);
#endif

typedef Vec3 Color;

HOST DEVICE inline Vec3::Vec3()
	: x(0),
	y(0),
	z(0)
{
}

HOST DEVICE inline Vec3::Vec3(float x, float y, float z)
	: x(x),
	y(y),
	z(z)
{
}

HOST DEVICE inline Vec3& Vec3::clamp(float min, float max)
{
	assert(min < max);

#ifdef __CUDACC__
	x = fminf(fmaxf(x, min), max);
	y = fminf(fmaxf(y, min), max);
	z = fminf(fmaxf(z, min), max);
#else
	x = std::min(std::max(x, min), max);
	y = std::min(std::max(y, min), max);
	z = std::min(std::max(z, min), max);
#endif
	return *this;
}

HOST DEVICE inline Vec3& Vec3::normalize()
{
	return *this /= length();
}

HOST DEVICE inline Vec3 Vec3::normalized() const
{
	return Vec3(*this) /= length();
}

HOST DEVICE inline float Vec3::length() const
{
#ifdef __CUDACC__
	return sqrtf(x * x + y * y + z * z);
#else
	return std::sqrtf(x * x + y * y + z * z);
#endif
}

HOST DEVICE inline float Vec3::sqrLength() const
{
	return x * x + y * y + z * z;
}

HOST DEVICE inline float Vec3::dot(const Vec3& v) const
{
	return x * v.x + y * v.y + z * v.z;
}

HOST DEVICE inline Vec3 Vec3::cross(const Vec3& v) const
{
	return{y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
}

HOST DEVICE inline float Vec3::sum() const
{
	return x + y + z;
}

HOST DEVICE inline float Vec3::min() const
{
#ifdef __CUDACC__
	return fminf(fminf(x, y), z);
#else
	return std::min(std::min(x, y), z);
#endif
}

HOST DEVICE inline float Vec3::max() const
{
#ifdef __CUDACC__
	return fmaxf(fmaxf(x, y), z);
#else
	return std::max(std::max(x, y), z);
#endif
}

HOST DEVICE inline Vec3 Vec3::operator-() const
{
	return{-x, -y, -z};
}

HOST DEVICE inline Vec3& Vec3::operator+=(const Vec3& v)
{
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

HOST DEVICE inline Vec3& Vec3::operator-=(const Vec3& v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return *this;
}

HOST DEVICE inline Vec3& Vec3::operator*=(const Vec3& v)
{
	x *= v.x;
	y *= v.y;
	z *= v.z;
	return *this;
}

HOST DEVICE inline Vec3& Vec3::operator*=(float f)
{
	x *= f;
	y *= f;
	z *= f;
	return *this;
}

HOST DEVICE inline Vec3& Vec3::operator/=(float f)
{
	return *this *= (1.0f / f);
}

HOST DEVICE inline float& Vec3::operator[](size_t index)
{
	assert(index < 3);

	return (&x)[index];
}

HOST DEVICE inline const float& Vec3::operator[](size_t index) const
{
	assert(index < 3);

	return (&x)[index];
}

HOST DEVICE inline float* Vec3::begin()
{
	return &x;
}

HOST DEVICE inline const float* Vec3::begin() const
{
	return &x;
}

HOST DEVICE inline float* Vec3::end()
{
	return &x + 3;
}

HOST DEVICE inline const float* Vec3::end() const
{
	return &x + 3;
}

HOST DEVICE inline Vec3 Vec3::black()
{
	return{0, 0, 0};
}

HOST DEVICE inline Vec3 Vec3::white()
{
	return{1, 1, 1};
}

HOST DEVICE inline Vec3 Vec3::biggest()
{
#ifndef __CUDACC__
	auto max = std::numeric_limits<float>::max();
	return{max, max, max};
#else
	return{FLT_MAX, FLT_MAX, FLT_MAX};
#endif
}

HOST DEVICE inline Vec3 Vec3::smallest()
{
#ifndef __CUDACC__
	auto min = std::numeric_limits<float>::min();
	return{min, min, min};
#else
	return{FLT_MIN, FLT_MIN, FLT_MIN};
#endif
}

HOST DEVICE inline Vec3 operator+(const Vec3& lhs, const Vec3& rhs)
{
	return{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

HOST DEVICE inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs)
{
	return{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

HOST DEVICE inline Vec3 operator*(const Vec3& lhs, float rhs)
{
	return{lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}

HOST DEVICE inline Vec3 operator*(const Vec3& lhs, const Vec3& rhs)
{
	return{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
}

HOST DEVICE inline Vec3 operator*(float lhs, const Vec3& rhs)
{
	return{lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
}

HOST DEVICE inline Vec3 operator/(const Vec3& lhs, float rhs)
{
	return{lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
}

HOST DEVICE inline Vec3 min(const Vec3& v1, const Vec3& v2)
{
#ifdef __CUDACC__
	return{fminf(v1.x, v2.x), fminf(v1.y, v2.y), fminf(v1.z, v2.z)};
#else
	return{std::min(v1.x, v2.x), std::min(v1.y, v2.y), std::min(v1.z, v2.z)};
#endif
}

HOST DEVICE inline Vec3 max(const Vec3& v1, const Vec3& v2)
{
#ifdef __CUDACC__
	return{fmaxf(v1.x, v2.x), fmaxf(v1.y, v2.y), fmaxf(v1.z, v2.z)};
#else
	return{std::max(v1.x, v2.x), std::max(v1.y, v2.y), std::max(v1.z, v2.z)};
#endif
}