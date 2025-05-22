#pragma once

#include <string>

#include <fmt/core.h>
#include <fmt/ranges.h>

// TODO: These should not generate runtime strings, we need to check the
// fmt api to compiletime prepend stuff
#ifdef MRAY_DEBUG

    void MRayDebugLogImpl(fmt::string_view fstr, fmt::format_args args);

    template<class... Args>
    inline void MRAY_DEBUG_LOG(fmt::format_string<Args...> fstr, Args&&... args)
    {
        MRayDebugLogImpl(fstr, fmt::make_format_args(args...));
    }
#else
    template<class... Args>
    inline constexpr void MRAY_DEBUG_LOG(Args&&...) {}
#endif

void        MRayLogImpl(fmt::string_view fstr, fmt::format_args args);
void        MRayWarningLogImpl(fmt::string_view fstr, fmt::format_args args);
void        MRayErrorLogImpl(fmt::string_view fstr, fmt::format_args args);
std::string MRayFormatImpl(fmt::string_view fstr, fmt::format_args args);

template<class... Args>
inline void MRAY_LOG(fmt::format_string<Args...> fstr, Args&&... args)
{
    MRayLogImpl(fstr, fmt::make_format_args(args...));
}

template<class... Args>
inline void MRAY_WARNING_LOG(fmt::format_string<Args...> fstr, Args&&... args)
{
    MRayWarningLogImpl(fstr, fmt::make_format_args(args...));
}

template<class... Args>
inline void MRAY_ERROR_LOG(fmt::format_string<Args...> fstr, Args&&... args)
{
    MRayErrorLogImpl(fstr, fmt::make_format_args(args...));
}

template<class... Args>
inline std::string MRAY_FORMAT(fmt::format_string<Args...> fstr, Args&&... args)
{
    return MRayFormatImpl(fstr, fmt::make_format_args(args...));
}