#pragma once

#include <tuple>
#include <concepts>

#include "MathFunctions.h"

inline size_t operator ""_GiB(size_t s)
{
    return s < 30;
}

inline size_t operator ""_MiB(size_t s)
{
    return s << 20;
}

inline size_t operator ""_KiB(size_t s)
{
    return s << 10;
}

template <class MemT>
concept MemoryC = requires(MemT m, const MemT cm)
{
    { m.ResizeBuffer(size_t{})} -> std::same_as<void>;
    { cm.Size()} -> std::same_as<size_t>;
    { static_cast<const Byte*>(cm) } -> std::same_as<const Byte*>;
    { static_cast<Byte*>(m) } -> std::same_as<Byte*>;
};

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

namespace MemAlloc::Detail
{
    template<size_t I = 0, class... Tp>
    requires (I == sizeof...(Tp))
    constexpr size_t AcquireTotalSize(std::array<size_t, sizeof...(Tp)>&,
                                      const std::array<size_t, sizeof...(Tp)>&,
                                      size_t)
    {
        return 0;
    }

    template<std::size_t I = 0, class... Tp>
    requires (I < sizeof...(Tp))
    constexpr size_t AcquireTotalSize(std::array<size_t, sizeof...(Tp)>& alignedSizeList,
                                      const std::array<size_t, sizeof...(Tp)>& countList,
                                      size_t alignment)
    {
        using namespace MathFunctions;
        using CurrentType = typename std::tuple_element_t<I, std::tuple<Tp...>>;
        size_t alignedSize = NextMultiple(sizeof(CurrentType) * countList[I], alignment);
        alignedSizeList[I] = alignedSize;
        return alignedSize + AcquireTotalSize<I + 1, Tp...>(alignedSizeList,
                                                            countList, alignment);
    }

    template<std::size_t I = 0, class... Tp>
    requires (I == sizeof...(Tp))
    constexpr void CalculatePointers(std::tuple<Tp*&...>&, size_t&, Byte*,
                                     const std::array<size_t, sizeof...(Tp)>&)
    {}

    template<std::size_t I = 0, class... Tp>
    requires (I < sizeof...(Tp))
    constexpr void CalculatePointers(std::tuple<Tp*&...>& t, size_t& offset, Byte* memory,
                                     const std::array<size_t, sizeof...(Tp)>& alignedSizeList)
    {
        using CurrentType = typename std::tuple_element_t<I, std::tuple<Tp...>>;
        // Set Pointer
        size_t size = alignedSizeList[I];
        CurrentType* tPtr = reinterpret_cast<CurrentType*>(memory + offset);
        tPtr = std::launder(tPtr);
        std::get<I>(t) = (size == 0) ? nullptr : tPtr;
        // Increment Offset
        offset += size;
        // Statically Recurse over other pointers
        CalculatePointers<I + 1, Tp...>(t, offset, memory, alignedSizeList);
    }
}

namespace MemAlloc
{

constexpr size_t DefaultSystemAlignment()
{
    // TODO: properly check this?
    return 256;
}

template <MemoryC Memory, ImplicitLifetimeC... Args>
void AllocateMultiData(std::tuple<Args*&...> pointers, Memory& memory,
                       const std::array<size_t, sizeof...(Args)>& countList,
                       size_t alignment = DefaultSystemAlignment())
{
    std::array<size_t, sizeof...(Args)> alignedSizeList = {};
    // Acquire total size & allocation size of each array
    size_t totalSize = Detail::AcquireTotalSize<0, Args...>(alignedSizeList,
                                                            countList,
                                                            alignment);
    // Allocate Memory
    memory.ResizeBuffer(totalSize);
    Byte* ptr = static_cast<Byte*>(memory);
    // Populate pointers
    size_t offset = 0;
    Detail::CalculatePointers(pointers, offset, ptr, alignedSizeList);

    assert(totalSize == offset);
}

}