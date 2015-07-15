#pragma once
#include <chrono>

class Timer
{
public:
	Timer();
	long long elapsedMs();

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> startPoint;
};