#include "Timer.h"
#include "Log.h"

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

uint64_t Timer::ElapsedIntMS() const
{
    Duration dur = std::bit_cast<Duration>(elapsed);
    return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(dur).count());
}

template <>
double Timer::Elapsed<Nanosecond>() const
{
    return std::chrono::duration<double, Nanosecond>(std::bit_cast<Duration>(elapsed)).count();
}

template <>
double Timer::Elapsed<Microsecond>() const
{
    return std::chrono::duration<double, Microsecond>(std::bit_cast<Duration>(elapsed)).count();
}

template <>
double Timer::Elapsed<Millisecond>() const
{
    return std::chrono::duration<double, Millisecond>(std::bit_cast<Duration>(elapsed)).count();
}

template <>
double Timer::Elapsed<Second>() const
{
    return std::chrono::duration<double, Second>(std::bit_cast<Duration>(elapsed)).count();
}

template <>
double Timer::Elapsed<Minute>() const
{
    return std::chrono::duration<double, Minute>(std::bit_cast<Duration>(elapsed)).count();
}

std::string FormatTimeDynamic(uint64_t ms)
{
    if(ms == std::numeric_limits<uint64_t>::max())
        return std::string("Eternity");

    // TODO: There is probably smarter way to do this,
    // check later
    static constexpr uint64_t MILLISEC_THRESHOLD = 1000;
    static constexpr double SEC_THRESHOLD = 60.0;
    //
    static constexpr double HOUR_IN_MINS = 60.0;
    static constexpr double DAY_IN_MINS = 24.0 * HOUR_IN_MINS;

    if(ms < MILLISEC_THRESHOLD)
        return MRAY_FORMAT("{:.3f}ms", double(ms));
    else if(auto sec = double(ms) / 1000.0; sec < SEC_THRESHOLD)
        return MRAY_FORMAT("{:.3f}s", sec);

    // Now do it manually
    // Approximate minutes as much as we can
    double totalMinutes = double(ms / 1000) / 60.0;
    if(totalMinutes < HOUR_IN_MINS)
    {
        double min;
        double sec = std::modf(totalMinutes, &min) * 60.0;
        return MRAY_FORMAT("{:d}m_{:.0f}s", static_cast<uint32_t>(min), sec);
    }
    else if(totalMinutes < DAY_IN_MINS)
    {
        double hour;
        double min = std::modf(totalMinutes / HOUR_IN_MINS, &hour) * 60;
        double sec = std::modf(min, &min) * 60;
        return MRAY_FORMAT("{:d}h_{:d}m_{:.0f}s",
                           static_cast<uint32_t>(hour),
                           static_cast<uint32_t>(min),
                           sec);
    }
    else
    {
        double day;
        double hour = std::modf(totalMinutes / DAY_IN_MINS, &day) * 24;
        double min = std::modf(hour, &hour) * 60;
        double sec = std::modf(min, &min) * 60;
        return MRAY_FORMAT("{:d}d_{:d}h_{:d}m_{:.0f}s",
                           static_cast<uint32_t>(day),
                           static_cast<uint32_t>(hour),
                           static_cast<uint32_t>(min),
                           sec);
    }
}

std::string FormatTimeDynamic(const Timer& t)
{
    // TODO: There is probably smarter way to do this,
    // check later
    static constexpr double MICROSEC_THRESHOLD = 1000.0;
    static constexpr double MILLISEC_THRESHOLD = 1000.0;
    static constexpr double SEC_THRESHOLD = 60.0;
    //
    static constexpr double HOUR_IN_MINS = 60.0;
    static constexpr double DAY_IN_MINS = 24.0 * HOUR_IN_MINS;

    if(auto us = t.Elapsed<Microsecond>(); us < MICROSEC_THRESHOLD)
        return MRAY_FORMAT("{:.3f}us", us);
    else if(auto ms = t.Elapsed<Millisecond>(); ms < MILLISEC_THRESHOLD)
        return MRAY_FORMAT("{:.3f}ms", ms);
    else if(auto sec = t.Elapsed<Second>(); sec < SEC_THRESHOLD)
        return MRAY_FORMAT("{:.3f}s", sec);

    // Now do it manually
    double totalMinutes = t.Elapsed<Minute>();
    if(totalMinutes < HOUR_IN_MINS)
    {
        double min;
        double sec = std::modf(totalMinutes, &min) * 60.0;
        return MRAY_FORMAT("{:d}m_{:.2f}s", static_cast<uint32_t>(min), sec);
    }
    else if(totalMinutes < DAY_IN_MINS)
    {
        double hour;
        double min = std::modf(totalMinutes / HOUR_IN_MINS, &hour) * 60;
        double sec = std::modf(min, &min) * 60;
        return MRAY_FORMAT("{:d}h_{:d}m_{:.1f}s",
                           static_cast<uint32_t>(hour),
                           static_cast<uint32_t>(min),
                           sec);
    }
    else
    {
        double day;
        double hour = std::modf(totalMinutes / DAY_IN_MINS, &day) * 24;
        double min = std::modf(hour, &hour) * 60;
        double sec = std::modf(min, &min) * 60;
        return MRAY_FORMAT("{:d}d_{:d}h_{:d}m_{:.0f}s",
                           static_cast<uint32_t>(day),
                           static_cast<uint32_t>(hour),
                           static_cast<uint32_t>(min),
                           sec);
    }
}