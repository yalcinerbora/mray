#include "Timer.h"

#include <chrono>

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration  = TimePoint::duration;

void Timer::Start()
{
    start = std::bit_cast<uint64_t>(Clock::now());
}

void Timer::Split()
{
    TimePoint end = Clock::now();
    TimePoint startC = std::bit_cast<TimePoint>(start);
    elapsed = std::bit_cast<uint64_t>(end - startC);
}

void Timer::Lap()
{
    TimePoint end = Clock::now();
    TimePoint startC = std::bit_cast<TimePoint>(start);
    elapsed = std::bit_cast<uint64_t>(end - startC);
    start = std::bit_cast<uint64_t>(end);
}

template <>
double Timer::Elapsed<Nanosecond>()
{
    return std::chrono::duration<double, Nanosecond>(elapsed).count();
}

template <>
double Timer::Elapsed<Microsecond>()
{
    return std::chrono::duration<double, Microsecond>(elapsed).count();
}

template <>
double Timer::Elapsed<Millisecond>()
{
    return std::chrono::duration<double, Millisecond>(elapsed).count();
}

template <>
double Timer::Elapsed<Second>()
{
    return std::chrono::duration<double, Second>(elapsed).count();
}

template <>
double Timer::Elapsed<Minue>()
{
    return std::chrono::duration<double, Minue>(elapsed).count();
}