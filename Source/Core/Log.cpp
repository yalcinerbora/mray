

#include "Log.h"

#include "Log.h"
#include <fmt/color.h>

#ifdef MRAY_DEBUG

void MRayDebugLogImpl(fmt::string_view fstr,
                      fmt::format_args args)
{
    fmt::print(stdout, "\033[2K\r{:s}: {:s}\n",
               fmt::format(fg(fmt::color::royal_blue), "Debug"),
               fmt::vformat(fstr, args));
}

#endif

void MRayLogImpl(fmt::string_view fstr, fmt::format_args args)
{
    fmt::print(stdout, "\033[2K\r{:s}\n",
               fmt::vformat(fstr, args));
}

void MRayWarningLogImpl(fmt::string_view fstr, fmt::format_args args)
{
    fmt::print(stdout, "\033[2K\r{:s}: {:s}\n",
               fmt::format(fg(fmt::color::golden_rod), "Warning"),
               fmt::vformat(fstr, args));

}

void MRayErrorLogImpl(fmt::string_view fstr, fmt::format_args args)
{
    fmt::print(stderr, "\033[2K\r{:s}: {:s}\n",
               fmt::format(fg(fmt::color::crimson), "Error"),
               fmt::vformat(fstr, args));
}

std::string MRayFormatImpl(fmt::string_view fstr, fmt::format_args args)
{
    return fmt::vformat(fstr, args);
}