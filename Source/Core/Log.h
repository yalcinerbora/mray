#pragma once

#include <string>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>

#ifdef MRAY_DEBUG

    template<class... Args>
    inline void MRAY_DEBUG_LOG(fmt::format_string<Args...> fstr, Args&&... args)
    {
        std::string s = fmt::format(fstr, std::forward<Args>(args)...);
        fmt::print(stdout, "{:s}: {:s}\n",
                   fmt::format(fg(fmt::color::royal_blue), "Debug"),
                   s);
    }
#else
    template<class... Args>
    inline constexpr void MRAY_DEBUG_LOG(Args&&...) {}
#endif

template<class... Args>
inline void MRAY_LOG(fmt::format_string<Args...> fstr, Args&&... args)
{
    std::string s = fmt::format(fstr, std::forward<Args>(args)...);
    fmt::print(stdout, "{:s}\n", s);
}

template<class... Args>
inline void MRAY_WARNING_LOG(fmt::format_string<Args...> fstr, Args&&... args)
{
    std::string s = fmt::format(fstr, std::forward<Args>(args)...);
    fmt::print(stdout, "{:s}: {:s}\n",
               fmt::format(fg(fmt::color::golden_rod), "Warning"),
               s);
}

template<class... Args>
inline void MRAY_ERROR_LOG(fmt::format_string<Args...> fstr, Args&&... args)
{
    std::string s = fmt::format(fstr, std::forward<Args>(args)...);
    fmt::print(stderr, "{:s}: {:s}\n",
               fmt::format(fg(fmt::color::crimson), "Error"),
               s);
}

template<class... Args>
inline std::string MRAY_FORMAT(fmt::format_string<Args...> fstr, Args&&... args)
{
    return fmt::format(fstr, std::forward<Args>(args)...);
}