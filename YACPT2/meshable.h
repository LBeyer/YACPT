#pragma once

#include "mesh.h"

class Meshable
{
public:
	virtual Mesh meshify() const = 0;
};