#pragma once

#include <ratio>
#include <string>

using Nanosecond    = std::nano;
using Microsecond   = std::micro;
using Millisecond   = std::milli;
using Second        = std::ratio<1>;
using Minute        = std::ratio<60>;

class Timer
{
    private:
    uint64_t    elapsed;
    uint64_t    start;

    //Duration    elapsed;
    //TimePoint   start;

    public:
    // Utility
    void        Start();
    void		Split();
    void		Lap();
    template <class Time>
    double      Elapsed() const;
};

std::string FormatTimeDynamic(const Timer& t);