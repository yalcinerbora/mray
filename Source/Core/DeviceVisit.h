#pragma once

#include "Definitions.h"

#include <cstdio>
#include <functional>
#include <variant>
#include <utility>

// First failure related to the concerns described above
// nvcc did not like it (see: https://godbolt.org/z/fM811b4cx)
// Thankfully it was not the actual variant implementation, I dunno how to
// implement it :(
// Some basic implementation of std::visit (recursive invocation)
// (Above link shows the compiled down result, it decays to nothing when all variables
// are known at compile time.)
namespace DeviceVisitDetail
{
    template<uint32_t I>
    using UIntTConst = std::integral_constant<uint32_t, I>;

    template<uint32_t I, class VariantT, class Func>
    requires(I < std::variant_size_v<std::remove_reference_t<VariantT>>)
    MR_GF_DECL constexpr
    auto RecurseVisitImpl(VariantT&& v, Func&& f) -> decltype(auto);

    template<uint32_t O, class VariantT, class Func>
    MR_GF_DECL constexpr
    auto IfElseVisitImpl(UIntTConst<O>, VariantT&& v, Func&& f) -> decltype(auto);
}

template<uint32_t I, class VariantT, class Func>
requires(I < std::variant_size_v<std::remove_reference_t<VariantT>>)
MR_GF_DEF constexpr
auto DeviceVisitDetail::RecurseVisitImpl(VariantT&& v, Func&& f) -> decltype(auto)
{
    using CurrentType = decltype(std::get<I>(v));

    if(I == v.index())
        return std::invoke(f, std::forward<CurrentType>(std::get<I>(v)));
    else if constexpr(I < std::variant_size_v<std::remove_reference_t<VariantT>> -1)
        return RecurseVisitImpl<I + 1>(std::forward<VariantT>(v), std::forward<Func>(f));
    MRAY_UNREACHABLE;
}

template<uint32_t O, class VariantT, class Func>
MR_GF_DEF constexpr
auto DeviceVisitDetail::IfElseVisitImpl(UIntTConst<O>, VariantT&& v, Func&& f) -> decltype(auto)
{
    using V = std::remove_cvref_t<VariantT>;
    constexpr uint32_t STAMP_COUNT = 16;
    constexpr uint32_t VSize = uint32_t(std::variant_size_v<V>);
    uint32_t index = uint32_t(v.index());
    // I dunno how to make this compile time
    // so we check it runtime
    [[maybe_unused]] int invokeCount = 0;
    #define COND_INVOKE(I)                                      \
        if constexpr(MRAY_IS_DEBUG) invokeCount++;              \
        if constexpr(VSize > (I + O)) if(index == O + I)        \
        {                                                       \
            using CurrentType = decltype(std::get<O + I>(v));   \
            return std::invoke                                  \
            (                                                   \
                std::forward<Func>(f),                          \
                std::forward<CurrentType>(std::get<O + I>(v))   \
            );                                                  \
        }
    // End COND_INVOKE
    COND_INVOKE(0)
    COND_INVOKE(1)
    COND_INVOKE(2)
    COND_INVOKE(3)
    COND_INVOKE(4)
    COND_INVOKE(5)
    COND_INVOKE(6)
    COND_INVOKE(7)
    COND_INVOKE(8)
    COND_INVOKE(9)
    COND_INVOKE(10)
    COND_INVOKE(11)
    COND_INVOKE(12)
    COND_INVOKE(13)
    COND_INVOKE(14)
    COND_INVOKE(15)
    #undef COND_INVOKE
    assert(invokeCount == STAMP_COUNT && "Invalid Visit implementation, "
           "add more\"COND_INVOKE\"s");
    if constexpr(VSize > O + STAMP_COUNT)
        return DeviceVisitDetail::IfElseVisitImpl(UIntTConst<O + STAMP_COUNT>{},
                                                  std::forward<VariantT>(v),
                                                  std::forward<Func>(f));
    MRAY_UNREACHABLE;
}

template<class VariantT, class Func>
MR_GF_DECL constexpr
auto DeviceVisit(VariantT&& v, Func&& f) -> decltype(auto)
{
    //return DeviceVisitDetail::RecurseVisitImpl<0>(std::forward<VariantT>(v), std::forward<Func>(f));
    return DeviceVisitDetail::IfElseVisitImpl
    (
        DeviceVisitDetail::UIntTConst<0>{},
        std::forward<VariantT>(v),
        std::forward<Func>(f)
    );
}