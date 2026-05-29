#include "Core/Log.h"


#pragma once

namespace mray::host::print
{
    template<class... Args>
    MR_GF_DEF void Print(fmt::format_string<Args...>, Args&&...);
}

namespace mray::host::print
{

template<class... Args>
MR_GF_DECL
void Print(fmt::format_string<Args...> fmtString, Args&&... args)
{
    MRAY_LOG(fmtString, std::forward<Args>(args)...);
}

}