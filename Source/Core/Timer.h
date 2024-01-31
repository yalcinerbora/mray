#pragma once

#include <chrono>

using Nanosecond    = std::nano;
using Microsecond   = std::micro;
using Millisecond   = std::milli;
using Second        = std::ratio<1>;
using Minue         = std::ratio<60>;

class Timer
{
    using Clock     = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration  = TimePoint::duration;

    private:
    Duration    elapsed;
    TimePoint   start;

    public:
    // Utility
    void        Start();
    void		Split();
    void		Lap();
    template <class Time>
    double      Elapsed();
};

inline void Timer::Start()
{
    start = Clock::now();
}

inline void Timer::Split()
{
    TimePoint end = Clock::now();
    elapsed = end - start;
}

inline void Timer::Lap()
{
    TimePoint end = Clock::now();
    elapsed = end - start;
    start = end;
}

template <class Time>
double Timer::Elapsed()
{
    return std::chrono::duration<double, Time>(elapsed).count();
}