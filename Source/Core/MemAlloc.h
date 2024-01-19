#pragma once

#include <tuple>
#include <concepts>

#include "MathFunctions.h"
#include "Types.h"

constexpr inline size_t operator ""_GiB(size_t s)
{
    return s < 30;
}

constexpr inline size_t operator ""_MiB(size_t s)
{
    return s << 20;
}

constexpr inline size_t operator ""_KiB(size_t s)
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

namespace MemAlloc
{

constexpr size_t DefaultSystemAlignment();

template <MemoryC Memory, ImplicitLifetimeC... Args>
void AllocateMultiData(std::tuple<Span<Args>&...> spans, Memory& memory,
                       const std::array<size_t, sizeof...(Args)>& countList,
                       size_t alignment = DefaultSystemAlignment());

template <class Memory>
requires requires(Memory m) { {m.ResizeBuffer(size_t{})} -> std::same_as<void>; }
void AllocateTextureSpace(std::vector<size_t>& offsets,
                          Memory& memory,
                          const std::vector<size_t>& sizes,
                          const std::vector<size_t>& alignments);

template <ImplicitLifetimeC Left, ImplicitLifetimeC Right>
constexpr Span<Left> RepurposeAlloc(Span<Right> rhs);

}


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
    constexpr void CalculateSpans(std::tuple<Span<Tp>&...>&, size_t&, Byte*,
                                     const std::array<size_t, sizeof...(Tp)>&)
    {}

    template<std::size_t I = 0, class... Tp>
    requires (I < sizeof...(Tp))
    constexpr void CalculateSpans(std::tuple<Span<Tp>&...>& t, size_t& offset, Byte* memory,
                                  const std::array<size_t, sizeof...(Tp)>& alignedSizeList)
    {
        using CurrentType = typename std::tuple_element_t<I, std::tuple<Tp...>>;
        // Set Pointer
        size_t size = alignedSizeList[I];
        CurrentType* tPtr = reinterpret_cast<CurrentType*>(memory + offset);
        tPtr = std::launder(tPtr);
        std::get<I>(t) = Span<CurrentType>((size == 0) ? nullptr : tPtr,
                                           size / sizeof(CurrentType));
        // Increment Offset
        offset += size;
        // Statically Recurse over other pointers
        CalculateSpans<I + 1, Tp...>(t, offset, memory, alignedSizeList);
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
void AllocateMultiData(std::tuple<Span<Args>&...> spans, Memory& memory,
                       const std::array<size_t, sizeof...(Args)>& countList,
                       size_t alignment)
{
    std::array<size_t, sizeof...(Args)> alignedSizeList;
    // Acquire total size & allocation size of each array
    size_t totalSize = Detail::AcquireTotalSize<0, Args...>(alignedSizeList,
                                                            countList,
                                                            alignment);
    // Allocate Memory
    memory.ResizeBuffer(totalSize);
    Byte* ptr = static_cast<Byte*>(memory);
    // Populate pointers
    size_t offset = 0;
    Detail::CalculateSpans(spans, offset, ptr, alignedSizeList);

    assert(totalSize == offset);
}

template <class Memory>
requires requires(Memory m) { {m.ResizeBuffer(size_t{})} -> std::same_as<void>; }
void AllocateTextureSpace(std::vector<size_t>& offsets,
                          Memory& memory,
                          const std::vector<size_t>& sizes,
                          const std::vector<size_t>& alignments)
{
    assert(sizes.size() == alignments.size());
    offsets.resize(sizes.size());

    size_t totalSize = 0;
    for(size_t i = 0; i < sizes.size(); i++)
    {
        size_t alignedSize = MathFunctions::NextMultiple(sizes[i], alignments[i]);
        offsets[i] = totalSize;
        totalSize += alignedSize;
    }

    // Allocate Memory
    memory.ResizeBuffer(totalSize);
}


template <ImplicitLifetimeC Left, ImplicitLifetimeC Right>
constexpr Span<Left> RepurposeAlloc(Span<Right> rhs)
{
    static_assert(alignof(Right) == alignof(Left));
    size_t elementCount = rhs.size_bytes() / sizeof(Left);

    // TODO: Check if this is UB (probably is)
    Byte* rawPtr = reinterpret_cast<Byte*>(rhs.data());
    Left* leftPtr = std::launder(reinterpret_cast<Left*>(rawPtr));
    return Span<Left>(leftPtr, elementCount);
}

}