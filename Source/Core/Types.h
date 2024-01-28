#pragma once

#include <span>
#include <optional>
#include <variant>
#include <vector>

#include "MathFunctions.h"
#include "Error.h"

// Untill c++23, we custom define this
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2674r0.pdf
// Directly from the above paper
template <class T>
concept ImplicitLifetimeC = requires()
{
    std::disjunction
    <
        std::is_scalar<T>,
        std::is_array<T>,
        std::is_aggregate<T>,
        std::conjunction
        <
            std::is_trivially_destructible<T>,
            std::disjunction
            <
                std::is_trivially_default_constructible<T>,
                std::is_trivially_copy_constructible<T>,
                std::is_trivially_move_constructible<T>
            >
        >
    >::value;
};

// Rename the std::optional, gpu may not like it
// most(all after c++20) of optional is constexpr
// so the "relaxed-constexpr" flag of nvcc will be able to compile it
// Just to be sure, aliasing here to ease refactoring
template <class T>
using Optional = std::optional<T>;

template <class T, std::size_t Extent = std::dynamic_extent>
using Span = std::span<T, Extent>;

template <class... Types>
using Variant = std::variant<Types...>;

template <class T0, class T1>
using Pair = std::pair<T0, T1>;

template <class... Args>
using Tuple = std::tuple<Args...>;

// TODO: reference_wrapper<T> vs. span<T,1> which is better?
template <class T>
using Ref = std::reference_wrapper<T>;

// First failure related to the concerns described above
// nvcc did not like it (see: https://godbolt.org/z/fM811b4cx)
// Thankfully it was not the actual variant implementation, I dunno how to
// implement it :(
// Some basic implementation of std::visit (recursive invocation)
// (Above link shows the compiled down result, it decays to nothing when all variables
// are known at compile time.)
namespace detail
{
    template<uint32_t I, class VariantT, class Func>
    requires(I < std::variant_size_v<std::remove_reference_t<VariantT>>)
    MRAY_HYBRID MRAY_CGPU_INLINE
    constexpr auto LoopAndInvoke(VariantT&& v, Func&& f) -> decltype(auto);
}

template<class VariantT, class Func>
MRAY_HYBRID
constexpr auto DeviceVisit(VariantT&& v, Func&& f) -> decltype(auto);


// Some span wrappers for convenience
template<class T, std::size_t Extent = std::dynamic_extent>
constexpr Span<const T, Extent> ToConstSpan(Span<T, Extent> s);

//
template<ImplicitLifetimeC T>
constexpr Span<T> ToSpan(std::vector<Byte> v);

//
template<class T0, std::size_t E0,
         class T1, std::size_t E1>
requires std::is_same_v<std::decay_t<T0>, std::decay_t<T1>>
constexpr bool IsSubspan(Span<T0, E0> checkedSpan, Span<T1, E1> bigSpan);

// This is definately a thing that i cannot even comprahend
// Thanks to:
// https://stackoverflow.com/questions/55941964/how-to-filter-duplicate-types-from-tuple-c
namespace UniqueVariantDetail
{
template <typename T, typename... Ts>
struct Unique : std::type_identity<T> {};

template <typename... Ts, typename U, typename... Us>
struct Unique<Variant<Ts...>, U, Us...>
    : std::conditional_t<(std::is_same_v<U, Ts> || ...)
    , Unique<Variant<Ts...>, Us...>
    , Unique<Variant<Ts..., U>, Us...>> {};

}

template <typename... Ts>
using UniqueVariant = typename UniqueVariantDetail::Unique<Variant<>, Ts...>::type;

template <class T>
struct SampleT
{
    T           sampledResult;
    Float       pdf;
};

// Bitspan
// Not exact match of a std::span but suits the needs
// and only dynamic extent
template <std::unsigned_integral T>
class Bitspan
{
    private:
    T*                  data;
    uint32_t            size;

    public:
    constexpr           Bitspan();
    constexpr           Bitspan(T*, uint32_t bitcount);
    constexpr           Bitspan(Span<T>);

    constexpr bool      operator[](uint32_t index) const;
    // Hard to return reference for modification
    constexpr void      SetBit(uint32_t index, bool) requires (!std::is_const_v<T>);
    constexpr uint32_t  Size() const;
    constexpr uint32_t  ByteSize() const;
};

template<uint32_t I, class VariantT, class Func>
requires(I < std::variant_size_v<std::remove_reference_t<VariantT>>)
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr auto detail::LoopAndInvoke(VariantT&& v, Func&& f) -> decltype(auto)
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
MRAY_HYBRID
constexpr auto DeviceVisit(VariantT&& v, Func&& f) -> decltype(auto)
{
    return detail::LoopAndInvoke<0>(std::forward<VariantT>(v), std::forward<Func>(f));
}

template<class T, std::size_t Extent>
constexpr Span<const T, Extent> ToConstSpan(Span<T, Extent> s)
{
    return Span<const T, Extent>(s);
}

// TODO add arrays maybe? (decay changes c arrays to ptrs)
template<class T0, std::size_t E0,
         class T1, std::size_t E1>
requires std::is_same_v<std::decay_t<T0>, std::decay_t<T1>>
constexpr bool IsSubspan(Span<T0, E0> checkedSpan, Span<T1, E1> bigSpan)
{
    ptrdiff_t diff = checkedSpan.data() - bigSpan.data();
    if(diff >= 0)
    {
        size_t diffS = static_cast<size_t>(diff);
        bool ptrInRange = diffS < bigSpan.size();
        bool backInRange = (diffS + checkedSpan.size()) <= bigSpan.size();
        return (ptrInRange && backInRange);
    }
    else return false;
}

template <std::unsigned_integral T>
constexpr Bitspan<T>::Bitspan()
    : data(nullptr)
    , size(0u)
{}

template <std::unsigned_integral T>
constexpr Bitspan<T>::Bitspan(T* ptr, uint32_t bitcount)
    : data(ptr)
    , size(bitcount)
{}

template <std::unsigned_integral T>
constexpr Bitspan<T>::Bitspan(Span<T> s)
    : data(s.data())
    , size(MathFunctions::NextMultiple(s.size(), sizeof(T)))
{}

template <std::unsigned_integral T>
constexpr bool Bitspan<T>::operator[](uint32_t index) const
{
    assert(index < size && "Out of range access on bitspan!");

    size_t wordIndex = index / sizeof(T);
    size_t wordLocalIndex = index % sizeof(T);

    return data[wordIndex] >> wordLocalIndex;
}

template <std::unsigned_integral T>
constexpr void Bitspan<T>::SetBit(uint32_t index, bool v) requires (!std::is_const_v<T>)
{
    assert(index < size && "Out of range access on bitspan!");

    size_t wordIndex = index / sizeof(T);
    size_t wordLocalIndex = index % sizeof(T);

    T localMask = std::numeric_limits<T>::max();
    localMask &= (static_cast<T>(v) << wordLocalIndex);

    data[wordIndex] &= localMask;
}

template <std::unsigned_integral T>
constexpr uint32_t Bitspan<T>::Size() const
{
    return size;
}

template <std::unsigned_integral T>
constexpr uint32_t Bitspan<T>::ByteSize() const
{
    return MathFunctions::NextMultiple(size, sizeof(T));
}