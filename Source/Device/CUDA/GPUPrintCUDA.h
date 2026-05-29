#pragma once

namespace mray::cuda::print
{
    template<uint32_t N, class... Args>
    MR_HF_DEF void Print(const char(&)[N], Args&&...);
}

namespace mray::cuda::print
{

template<uint32_t N, class... Args>
MR_HF_DECL
void Print(const char (&fmtString)[N], Args&&... args)
{
    printf(fmtString, std::forward<Args>(args)...);
}

}