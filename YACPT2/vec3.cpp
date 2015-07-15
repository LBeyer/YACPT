#include "vec3.h"
#include <iostream>

std::ostream& operator<<(std::ostream& ostr, const Vec3& v)
{
	return ostr << v.x << " " << v.y << " " << v.z;
}

std::istream& operator>>(std::istream& istr, Vec3& v)
{
	return istr >> v.x >> v.y >> v.z;
}