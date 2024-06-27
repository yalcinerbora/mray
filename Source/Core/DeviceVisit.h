#pragma once

#include "Types.h"

namespace DeviceVisitDetail
{
    template<uint32_t I, class VariantT, class Func>
    requires(I < std::variant_size_v<std::remove_reference_t<VariantT>>)
    MRAY_HYBRID
    constexpr auto LoopAndInvoke(VariantT&& v, Func&& f) -> decltype(auto);

}

template<uint32_t I, class VariantT, class Func>
requires(I < std::variant_size_v<std::remove_reference_t<VariantT>>)
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr auto DeviceVisitDetail::LoopAndInvoke(VariantT&& v, Func&& f) -> decltype(auto)
{
    using CurrentType = decltype(std::get<I>(v));

    if(I == v.index())
        return std::invoke(f, std::forward<CurrentType>(std::get<I>(v)));
    else if constexpr(I < std::variant_size_v<std::remove_reference_t<VariantT>> -1)
        return LoopAndInvoke<I + 1>(std::forward<VariantT>(v), std::forward<Func>(f));
    else
    {
        #ifdef __CUDA_ARCH__
            if constexpr (MRAY_IS_DEBUG)
                printf("Invalid variant access on device!\n");
            __trap();
        #else
            throw MRayError("Invalid variant access on device!");
        #endif
    }
}

template<class VariantT, class Func>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr auto DeviceVisit(VariantT&& v, Func&& f) -> decltype(auto)
{
    return DeviceVisitDetail::LoopAndInvoke<0>(std::forward<VariantT>(v), std::forward<Func>(f));
}