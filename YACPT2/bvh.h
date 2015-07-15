#pragma once

#include "util.h"
#include "aabb.h"
#include "intersection.h"
#include "managedPtr.h"
#include <utility>		// forward
#include <memory>		// unique_ptr
#include <numeric>		// iota
#include <algorithm>	// generate, sort
#include <cstring>		// memcpy

struct Split
{
	uint32_t pivot;
	uint8_t axis;
	AABB left, right;
};

struct Node
{
	AABB aabb;
	uint32_t hit, noHit;
	uint32_t begin, end;
};

inline Split findBestSplit(const AABB* aabbs, const std::unique_ptr<uint32_t[]> (&sorted)[3], uint32_t begin, uint32_t end, const AABB& aabb, float cTrav, float cIts);
inline AABB accumulateAABB(const AABB* aabbs, const uint32_t* indices, uint32_t begin, uint32_t end);
inline void makeLeaf(Node& node);
HOST DEVICE inline bool isLeaf(const Node& node);

template<class Shape>
class BVH
{
public:
	template<class ...Args>
	BVH(Shape* shapes, uint32_t shapeCount, float cTrav, float cIts, Args&&... args);

	AABB getAABB() const;
	Vec3 getMidPoint() const;

	template<class ...Args>
	DEVICE Intersection intersect(const Ray& ray, Args&&... args) const;

private:
	void shrinkToFit(size_t nodeCount);

	ManagedPtr<Node[]> nodes;
	ManagedPtr<uint32_t[]> indices;
	const Shape* shapes;
};

inline Split findBestSplit(const AABB* aabbs, const std::unique_ptr<uint32_t[]>(&sorted)[3], uint32_t begin, uint32_t end, const AABB& aabb, float cTrav, float cIts)
{
	Split split = {0, 3};
	auto bestCost = cIts * (end - begin);
	auto surfaceArea = aabb.surfaceArea();

	for(uint8_t axis = 0; axis < 3; axis++)
	{
		for(auto pivot = begin + 1; pivot < end - 1; pivot++)
		{
			auto nLeft = pivot - begin;
			auto nRight = end - pivot;
			auto leftAABB = accumulateAABB(aabbs, sorted[axis].get(), begin, pivot);
			auto rightAABB = accumulateAABB(aabbs, sorted[axis].get(), pivot, end);
			auto cost = cTrav + cIts * (leftAABB.surfaceArea() / surfaceArea * nLeft + rightAABB.surfaceArea() / surfaceArea * nRight);
			if(cost < bestCost)
			{
				bestCost = cost;
				split = {pivot, axis, leftAABB, rightAABB};
			}
		}
	}
	return split;
}

inline AABB accumulateAABB(const AABB* aabbs, const uint32_t* indices, uint32_t begin, uint32_t end)
{
	AABB aabb = {Vec3::biggest(), Vec3::smallest()};
	for(auto index = begin; index < end; index++)
	{
		aabb += aabbs[indices[index]];
	}
	return aabb;
}

inline void makeLeaf(Node& node)
{
	node.hit = node.noHit;
}

HOST DEVICE inline bool isLeaf(const Node& node)
{
	return node.hit == node.noHit;
}

template<class Shape>
template<class ...Args>
BVH<Shape>::BVH(Shape* shapes, uint32_t shapeCount, float cTrav, float cIts, Args&&... args)
	: nodes(makeManaged<Node[]>(2 * shapeCount - 1)),
	indices(makeManaged<uint32_t[]>(shapeCount)),
	shapes(shapes)
{
	// we reorder these indices so we dont change the order of shapes
	std::iota(indices.begin(), indices.end(), 0);

	// cache the midpoints of the shapes
	std::unique_ptr<Vec3[]> midPoints(new Vec3[shapeCount]);
	size_t index = 0;
	std::generate(midPoints.get(), midPoints.get() + shapeCount, [&]()
	{
		return shapes[index++].getMidPoint(std::forward<Args>(args)...);
	});

	// cache the aabbs
	std::unique_ptr<AABB[]> aabbs(new AABB[shapeCount]);
	index = 0;
	std::generate(aabbs.get(), aabbs.get() + shapeCount, [&]()
	{
		return shapes[index++].getAABB(std::forward<Args>(args)...);
	});

	// indices sorted by axis
	std::unique_ptr<uint32_t[]> sorted[3];
	for(auto axis = 0; axis < 3; axis++)
	{
		sorted[axis].reset(new uint32_t[shapeCount]);
		std::iota(sorted[axis].get(), sorted[axis].get() + shapeCount, 0);
		std::sort(sorted[axis].get(), sorted[axis].get() + shapeCount, [&](uint32_t lhs, uint32_t rhs)
		{
			return midPoints[lhs][axis] < midPoints[rhs][axis];
		});
	}

	// buffers
	std::unique_ptr<uint32_t[]> temp[2] = {std::unique_ptr<uint32_t[]>(new uint32_t[shapeCount]), std::unique_ptr<uint32_t[]>(new uint32_t[shapeCount])};

	auto& root = nodes[0];
	root.begin = 0;
	root.end = shapeCount;
	root.noHit = 0;
	root.aabb = accumulateAABB(aabbs.get(), indices.get(), root.begin, root.end);

	auto current = 0u;
	auto toDo = 1u;
	auto nodeCount = 1u;
	while(toDo > 0)
	{
		auto& node = nodes[current];
		auto split = findBestSplit(aabbs.get(), sorted, node.begin, node.end, node.aabb, cTrav, cIts);
		if(split.axis == 3)	// no split
		{
			makeLeaf(node);
		}
		else
		{
			// update indices
			std::memcpy(indices.get() + node.begin, sorted[split.axis].get() + node.begin, (node.end - node.begin) * sizeof(uint32_t));
			// splitting point on the axis of pivot
			auto midPoint = (midPoints[indices[split.pivot - 1]][split.axis] + midPoints[indices[split.pivot]][split.axis]) / 2;
			// update sorted indices
			for(auto axis = 0u; axis < 3; axis++)
			{
				if(axis != split.axis)
				{
					auto l = 0u;
					auto r = 0u;
					for(auto i = node.begin; i < node.end; i++)
					{
						// write to buffer, copy to sorted
						if(midPoints[sorted[axis][i]][split.axis] < midPoint) // left side
						{
							temp[0][l] = sorted[axis][i];
							l++;
						}
						else // right side
						{
							temp[1][r] = sorted[axis][i];
							r++;
						}
					}
					std::memcpy(sorted[axis].get() + node.begin, temp[0].get(), l * sizeof(uint32_t));
					std::memcpy(sorted[axis].get() + node.begin + l, temp[1].get(), r * sizeof(uint32_t));
				}
			}
			node.hit = nodeCount;
			auto& leftChild = nodes[nodeCount];
			auto& rightChild = nodes[nodeCount + 1];

			leftChild.begin = node.begin;
			leftChild.end = split.pivot;
			leftChild.noHit = nodeCount + 1;
			leftChild.aabb = split.left;

			rightChild.begin = split.pivot;
			rightChild.end = node.end;
			rightChild.noHit = node.noHit;
			rightChild.aabb = split.right;

			nodeCount += 2;
			toDo += 2;
		}
		current++;
		toDo--;
	}
	shrinkToFit(nodeCount);
}

template<class Shape>
AABB BVH<Shape>::getAABB() const
{
	return nodes->aabb;
}

template<class Shape>
Vec3 BVH<Shape>::getMidPoint() const
{
	return nodes->aabb.midPoint();
}

template<class Shape>
template<class ...Args>
DEVICE Intersection BVH<Shape>::intersect(const Ray& ray, Args&&... args) const
{
	auto sqrDistance = FLT_MAX;
	Intersection closestIntersection;
	uint32_t current = 0;
	do
	{
		auto& node = nodes[current];
		if(!(node.aabb.intersect(ray)))
		{
			current = node.noHit;
		}
		else if(isLeaf(node))
		{
			for(auto i = node.begin; i < node.end; i++)
			{
				auto currentIntersection = shapes[indices[i]].intersect(ray, std::forward<Args>(args)...);
				if(currentIntersection)
				{
					auto currentSqrDistance = (currentIntersection.point - ray.origin).sqrLength();
					if(currentSqrDistance < sqrDistance)
					{
						closestIntersection = currentIntersection;
						sqrDistance = currentSqrDistance;
					}
				}
			}
			current = node.hit;
		}
		else
		{
			current = node.hit;
		}
	} while(current != 0);
	return closestIntersection;
}

template<class Shape>
void BVH<Shape>::shrinkToFit(size_t nodeCount)
{
	auto shrunkNodes = makeManaged<Node[]>(nodeCount);
	std::memcpy(shrunkNodes.begin(), nodes.begin(), nodeCount * sizeof(Node));
	nodes = std::move(shrunkNodes);
}