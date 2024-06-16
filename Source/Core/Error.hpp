#pragma once

#include "Log.h"

template<class... Args>
inline MRayError::MRayError(fmt::format_string<Args...> fstr, Args&&... args)
    : type(MRayError::HAS_ERROR)
    , customInfo(MRAY_FORMAT(fstr, std::forward<Args>(args)...))
{}

inline void MRayError::AppendInfo(const std::string& s)
{
    if(customInfo.empty())
        customInfo = MRAY_FORMAT("{:s}", s);
    else
        customInfo += MRAY_FORMAT("|| {:s}", s);
}
