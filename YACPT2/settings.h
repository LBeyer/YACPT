#pragma once

#include "resolution.h"

struct BVHSettings
{
	float cTrav, cIts;
};

struct Settings
{
	Resolution resolution;
	BVHSettings bvhSettings;
	float gamma;
};