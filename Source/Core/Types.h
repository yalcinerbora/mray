#pragma once

#include <span>
#include <optional>
#include <variant>
#include <vector>

#include "MathFunctions.h"

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

template<class T, std::size_t Extent = std::dynamic_extent>
constexpr Span<const T, Extent> ToConstSpan(Span<T, Extent> s)
{
    return Span<const T, Extent>(s);
}

template<ImplicitLifetimeC T>
constexpr Span<T> ToSpan(std::vector<Byte> v)
{
    assert(reinterpret_cast<uintptr_t>(v.data()) % alignof(T) == 0);
    assert(v.size() % sizeof(T) == 0);

    // TODO: Check if this is UB
    T* tPtr = reinterpret_cast<T*>(v.data());
    size_t size = v.size() / sizeof(T);
    return Span<T>(tPtr, size);
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