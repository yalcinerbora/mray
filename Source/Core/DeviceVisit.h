#pragma once

#include "Definitions.h"

#include <cstdio>
#include <functional>
#include <variant>

// First failure related to the concerns described above
// nvcc did not like it (see: https://godbolt.org/z/fM811b4cx)
// Thankfully it was not the actual variant implementation, I dunno how to
// implement it :(
// Some basic implementation of std::visit (recursive invocation)
// (Above link shows the compiled down result, it decays to nothing when all variables
// are known at compile time.)
namespace DeviceVisitDetail
{
    template<uint32_t I, class VariantT, class Func>
    requires(I < std::variant_size_v<std::remove_reference_t<VariantT>>)
    MRAY_GPU
    constexpr auto LoopAndInvoke(VariantT&& v, Func&& f) -> decltype(auto);

}

template<uint32_t I, class VariantT, class Func>
requires(I < std::variant_size_v<std::remove_reference_t<VariantT>>)
MRAY_GPU MRAY_GPU_INLINE
constexpr auto DeviceVisitDetail::LoopAndInvoke(VariantT&& v, Func&& f) -> decltype(auto)
{
    using CurrentType = decltype(std::get<I>(v));

    if(I == v.index())
        return std::invoke(f, std::forward<CurrentType>(std::get<I>(v)));
    else if constexpr(I < std::variant_size_v<std::remove_reference_t<VariantT>> -1)
        return LoopAndInvoke<I + 1>(std::forward<VariantT>(v), std::forward<Func>(f));
    else
    {
        if constexpr(MRAY_IS_DEBUG)
            printf("Invalid variant access on device!\n");

        //
        #ifdef MRAY_DEVICE_CODE_PATH_CUDA
            __trap();
        #else
            abort();
        #endif
    }
}

template<class VariantT, class Func>
MRAY_GPU MRAY_GPU_INLINE
constexpr auto DeviceVisit(VariantT&& v, Func&& f) -> decltype(auto)
{
    return DeviceVisitDetail::LoopAndInvoke<0>(std::forward<VariantT>(v), std::forward<Func>(f));
}