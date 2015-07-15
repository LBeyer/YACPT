#include "timer.h"

Timer::Timer()
	: startPoint(std::chrono::high_resolution_clock::now())
{
}

long long Timer::elapsedMs()
{
	auto now = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::milliseconds>(now - startPoint).count();
}