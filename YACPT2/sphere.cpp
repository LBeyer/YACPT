#include "sphere.h"
#include <map>
#include <utility>

class MidPointHelper
{
public:
	void add(const Triangle& face, std::vector<Vec3>& vertices);
	uint32_t getMidPoint(uint32_t a, uint32_t b);

private:
	std::map<std::pair<uint32_t, uint32_t>, uint32_t> map;
};


Sphere::Sphere(const Vec3& position, float radius, uint32_t materialIndex, unsigned int refinementSteps)
	: position(position),
	radius(radius),
	materialIndex(materialIndex),
	refinementSteps(refinementSteps)
{
}

Mesh Sphere::meshify() const
{
	std::vector<Vec3> vertices;
	std::vector<Triangle> faces;

	auto t = (1 + std::sqrtf(5)) / 2;

	vertices.push_back({-1, t, 0});
	vertices.push_back({1, t, 0});
	vertices.push_back({-1, -t, 0});
	vertices.push_back({1, -t, 0});

	vertices.push_back({0, -1, t});
	vertices.push_back({0, 1, t});
	vertices.push_back({0, -1, -t});
	vertices.push_back({0, 1, -t});

	vertices.push_back({t, 0, -1});
	vertices.push_back({t, 0, 1});
	vertices.push_back({-t, 0, -1});
	vertices.push_back({-t, 0, 1});

	for(auto& vertex : vertices)
	{
		vertex.normalize();
	}

	faces.push_back({0, 11, 5, materialIndex});
	faces.push_back({0, 5, 1, materialIndex});
	faces.push_back({0, 1, 7, materialIndex});
	faces.push_back({0, 7, 10, materialIndex});
	faces.push_back({0, 10, 11, materialIndex});
	faces.push_back({1, 5, 9, materialIndex});
	faces.push_back({5, 11, 4, materialIndex});
	faces.push_back({11, 10, 2, materialIndex});
	faces.push_back({10, 7, 6, materialIndex});
	faces.push_back({7, 1, 8, materialIndex});
	faces.push_back({3, 9, 4, materialIndex});
	faces.push_back({3, 4, 2, materialIndex});
	faces.push_back({3, 2, 6, materialIndex});
	faces.push_back({3, 6, 8, materialIndex});
	faces.push_back({3, 8, 9, materialIndex});
	faces.push_back({4, 9, 5, materialIndex});
	faces.push_back({2, 4, 11, materialIndex});
	faces.push_back({6, 2, 10, materialIndex});
	faces.push_back({8, 6, 7, materialIndex});
	faces.push_back({9, 8, 1, materialIndex});

	for(auto i = 0u; i < refinementSteps; i++)
	{
		std::vector<Triangle> newFaces;
		newFaces.reserve(faces.size() * 4);
		MidPointHelper helper;
		for(const auto& face : faces)
		{
			helper.add(face, vertices);
			// replace face by 4 faces
			auto ab = helper.getMidPoint(face.a, face.b);
			auto ac = helper.getMidPoint(face.a, face.c);
			auto bc = helper.getMidPoint(face.b, face.c);

			newFaces.push_back({face.a, ab, ac, materialIndex});
			newFaces.push_back({face.b, ab, bc, materialIndex});
			newFaces.push_back({face.c, ac, bc, materialIndex});
			newFaces.push_back({ab, ac, bc, materialIndex});
		}
		faces = newFaces;
	}

	for(auto& vertex : vertices)
	{
		vertex *= radius;
		vertex += position;
	}

	return{vertices, faces};
}

void MidPointHelper::add(const Triangle& face, std::vector<Vec3>& vertices)
{
	uint32_t indices[3] = {face.a, face.b, face.c};
	for(auto i = 0u; i < 3; i++)
	{
		auto a = std::min(indices[i], indices[(i + 1) % 3]);
		auto b = std::max(indices[i], indices[(i + 1) % 3]);
		auto key = std::make_pair(a, b);
		if(map.find(key) == map.end())
		{
			auto midPoint = ((vertices[a] + vertices[b]) / 2).normalize();
			map[key] = static_cast<uint32_t>(vertices.size());
			vertices.push_back(midPoint);
		}
	}
}

uint32_t MidPointHelper::getMidPoint(uint32_t a, uint32_t b)
{
	return map[std::make_pair(std::min(a, b), std::max(a, b))];
}