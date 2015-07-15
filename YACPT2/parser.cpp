#include "parser.h"
#include <iostream>
#include <fstream>
#include <map>
#include <cctype>
#include "sphere.h"

void removeWhitespace(std::string& s);
void toLowerCase(std::string& s);
void parseSettings(std::ifstream& file, Scene& scene);
void parseCamera(std::ifstream& file, Scene& scene);
void parseMaterial(std::ifstream& file, Scene& scene);
void parseSphere(std::ifstream& file, Scene& scene);
void parseMesh(std::ifstream& file, Scene& scene);

Scene parse(const std::string& filename)
{
	std::ifstream file(filename);
	if(!file)
	{
		return Scene();
	}
	Scene scene;
	std::map<std::string, void(*)(std::ifstream&, Scene&)> map;
	map["settings"] = parseSettings;
	map["camera"] = parseCamera;
	map["material"] = parseMaterial;
	map["sphere"] = parseSphere;
	map["mesh"] = parseMesh;
	std::string word;
	while(std::getline(file, word, '{'))
	{
		removeWhitespace(word);
		toLowerCase(word);
		if(map.find(word) == map.end())
		{
			return Scene();
		}
		else
		{
			map[word](file, scene);
		}
	}
	return scene;
}

void removeWhitespace(std::string& s)
{
	s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
}

void toLowerCase(std::string& s)
{
	std::transform(s.begin(), s.end(), s.begin(), ::tolower);
}

void parseSettings(std::ifstream& file, Scene& scene)
{
	std::string word;
	while(word != "}")
	{
		file >> word;
		removeWhitespace(word);
		toLowerCase(word);
		if(word == "resolution")
		{
			unsigned int width, height;
			file >> width >> height;
			scene.settings.resolution = {width, height};
		}
		else if(word == "intersectioncost")
		{
			float cIts;
			file >> cIts;
			scene.settings.bvhSettings.cIts = cIts;
		}
		else if(word == "traversalcost")
		{
			float cTrav;
			file >> cTrav;
			scene.settings.bvhSettings.cTrav = cTrav;
		}
		else if(word == "gamma")
		{
			float gamma;
			file >> gamma;
			scene.settings.gamma = gamma;
		}
		else if(word == "}")
		{
		}
		else
		{
			return;
		}
	}
}

void parseCamera(std::ifstream& file, Scene& scene)
{
	std::string word;
	Vec3 pov;
	Vec3 lookAt = {0, 0, 1};
	Vec3 up = {0, 1, 0};
	auto fovY = 30.0f;
	auto aspectRatio = 4.0f / 3.0f;
	while(word != "}")
	{
		file >> word;
		removeWhitespace(word);
		toLowerCase(word);
		if(word == "pov")
		{
			file >> pov;
		}
		else if(word == "lookat")
		{
			file >> lookAt;
		}
		else if(word == "up")
		{
			file >> up;
		}
		else if(word == "fovy")
		{
			file >> fovY;
		}
		else if(word == "aspectratio")
		{
			float aspectWidth, aspectHeight;
			file >> aspectWidth;
			file.ignore(std::numeric_limits<std::streamsize>::max(), ':');
			file >> aspectHeight;
			aspectRatio = aspectWidth / aspectHeight;
		}
		else if(word == "}")
		{
		}
		else
		{
			return;
		}
	}
	scene.camera = Camera(pov, lookAt, up, fovY, aspectRatio);
}

void parseMaterial(std::ifstream& file, Scene& scene)
{
	std::string word;
	while(word != "}")
	{
		file >> word;
		removeWhitespace(word);
		toLowerCase(word);
		if(word == "diffuse")
		{
			Vec3 color;
			file >> color;
			scene.world.add(Material::diffuse(color));
		}
		else if(word == "specular")
		{
			Vec3 color;
			file >> color;
			scene.world.add(Material::specular(color));
		}
		else if(word == "emissive")
		{
			Vec3 color;
			file >> color;
			scene.world.add(Material::emissive(color));
		}
		else if(word == "}")
		{
		}
		else
		{
			return;
		}
	}
}

void parseSphere(std::ifstream& file, Scene& scene)
{
	std::string word;
	Vec3 position;
	auto radius = 1.0f;
	auto materialIndex = 0u;
	auto refinementSteps = 0u;
	while(word != "}")
	{
		file >> word;
		removeWhitespace(word);
		toLowerCase(word);
		if(word == "position")
		{
			file >> position;
		}
		else if(word == "radius")
		{
			file >> radius;
		}
		else if(word == "material")
		{
			file >> materialIndex;
		}
		else if(word == "refinementsteps")
		{
			file >> refinementSteps;
		}
		else if(word == "}")
		{
		}
		else
		{
			return;
		}
	}
	scene.world.add(Sphere(position, radius, materialIndex, refinementSteps));
}

void parseMesh(std::ifstream& file, Scene& scene)
{
	std::string word;
	std::vector<Vec3> vertices;
	std::vector<Triangle> faces;
	auto materialIndex = 0u;
	auto globalMaterial = false;
	while(word != "}")
	{
		file >> word;
		removeWhitespace(word);
		toLowerCase(word);
		if(word == "material")
		{
			file >> materialIndex;
			globalMaterial = true;
		}
		else if(word == "vertices")
		{
			Vec3 vertex;
			while(file >> vertex)
			{
				vertices.push_back(vertex);
			}
			file.clear();
		}
		else if(word == "faces")
		{
			if(globalMaterial)
			{
				unsigned int a, b, c;
				while(file >> a >> b >> c)
				{
					faces.push_back({a, b, c, materialIndex});
				}
				file.clear();
			}
			else
			{
				unsigned int a, b, c;
				while(file >> a >> b >> c >> materialIndex)
				{
					faces.push_back({a, b, c, materialIndex});
				}
				file.clear();
			}
		}
		else if(word == "import")	// ayyy spaghetti lmao
		{
			std::string filename;
			file >> filename;
			std::ifstream importFile(filename);
			importFile >> word;
			if(word == "vertices")
			{
				Vec3 vertex;
				while(importFile >> vertex)
				{
					vertices.push_back(vertex);
				}
				importFile.clear();
			}
			importFile >> word;
			if(word == "faces")
			{
				unsigned int a, b, c;
				while(importFile >> a >> b >> c)
				{
					faces.push_back({a, b, c, materialIndex});
				}
				importFile.clear();
			}
		}
		else if(word == "}")
		{
		}
		else
		{
			return;
		}
	}
	scene.world.add(Mesh(vertices, faces));
}