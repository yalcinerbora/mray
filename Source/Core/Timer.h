#pragma once

#include <ratio>

using Nanosecond    = std::nano;
using Microsecond   = std::micro;
using Millisecond   = std::milli;
using Second        = std::ratio<1>;
using Minue         = std::ratio<60>;

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
    double      Elapsed();
};



//template <class Time>
//double Timer::Elapsed()
//{
//    return std::chrono::duration<double, Time>(elapsed).count();
//}