#pragma once

#include <tuple>
#include <concepts>

#include "Math.h"
#include "Types.h"

constexpr inline size_t operator ""_TiB(unsigned long long int s)
{
    return s << 40;
}

constexpr inline size_t operator ""_GiB(unsigned long long int s)
{
    return s << 30;
}

constexpr inline size_t operator ""_MiB(unsigned long long int s)
{
    return s << 20;
}

constexpr inline size_t operator ""_KiB(unsigned long long int s)
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

template<ImplicitLifetimeC Left, ImplicitLifetimeC Right>
static constexpr bool RepurposeAllocRequirements =
(
    !std::is_same_v<std::remove_cvref_t<Right>, Byte> && (alignof(Right) >= alignof(Left)) ||
    std::is_same_v<std::remove_cvref_t<Right>, Byte>
);

constexpr size_t DefaultSystemAlignment();

template <MemoryC Memory, ImplicitLifetimeC... Args>
void AllocateMultiData(Tuple<Span<Args>&...> spans, Memory& memory,
                       const std::array<size_t, sizeof...(Args)>& countList,
                       size_t alignment = DefaultSystemAlignment());

template <MemoryC Memory, ImplicitLifetimeC... Args>
Tuple<Span<Args>...> AllocateMultiData(Memory& memory,
                                       const std::array<size_t, sizeof...(Args)>& countList,
                                       size_t alignment = DefaultSystemAlignment());

template<ImplicitLifetimeC T, MemoryC Memory>
std::vector<Span<T>>
AllocateSegmentedData(Memory& memory, const std::vector<size_t>& counts,
                      size_t alignment = DefaultSystemAlignment());

template <class Memory>
requires requires(Memory m) { {m.ResizeBuffer(size_t{})} -> std::same_as<void>; }
std::vector<size_t> AllocateTextureSpace(Memory& memory,
                                         const std::vector<size_t>& sizes,
                                         const std::vector<size_t>& alignments);

template<size_t N>
size_t RequiredAllocation(const std::array<size_t, N>& byteSizeList,
                          size_t alignment = DefaultSystemAlignment());

template <ImplicitLifetimeC Left, ImplicitLifetimeC Right>
requires RepurposeAllocRequirements<Left, Right>
constexpr Span<Left> RepurposeAlloc(Span<Right> rhs);

// Simple wrapper to utilize "MultiData" allocation scheme
struct VectorBackedMemory
{
    std::vector<uint8_t>    v;
    //
    void                    ResizeBuffer(size_t s);
    size_t                  Size() const;
    explicit operator const Byte*() const;
    explicit operator       Byte*();
};

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
        using namespace Math;
        using CurrentType = typename std::tuple_element_t<I, Tuple<Tp...>>;
        size_t alignedSize = NextMultiple(sizeof(CurrentType) * countList[I], alignment);
        alignedSizeList[I] = alignedSize;
        return alignedSize + AcquireTotalSize<I + 1, Tp...>(alignedSizeList,
                                                            countList, alignment);
    }

    template<std::size_t I = 0, class... Tp>
    requires (I == sizeof...(Tp))
    constexpr void CalculateSpans(Tuple<Span<Tp>&...>&, size_t&, Byte*,
                                  const std::array<size_t, sizeof...(Tp)>&,
                                  const std::array<size_t, sizeof...(Tp)>&)
    {}

    template<std::size_t I = 0, class... Tp>
    requires (I < sizeof...(Tp))
    constexpr void CalculateSpans(Tuple<Span<Tp>&...>& t, size_t& offset, Byte* memory,
                                  const std::array<size_t, sizeof...(Tp)>& alignedSizeList,
                                  const std::array<size_t, sizeof...(Tp)>& countList)
    {
        using CurrentType = typename std::tuple_element_t<I, Tuple<Tp...>>;
        // Set Pointer
        size_t size = alignedSizeList[I];
        CurrentType* tPtr = reinterpret_cast<CurrentType*>(memory + offset);
        tPtr = std::launder(tPtr);
        std::get<I>(t) = Span<CurrentType>((size == 0) ? nullptr : tPtr, countList[I]);
        assert(size / sizeof(CurrentType) >= countList[I]);

        // Increment Offset
        offset += size;
        // Statically Recurse over other pointers
        CalculateSpans<I + 1, Tp...>(t, offset, memory, alignedSizeList, countList);
    }
}

namespace MemAlloc
{

constexpr size_t DefaultSystemAlignment()
{
    // Each device has multiple alignment expositions,
    // (CUDA malloc provides 256byte aligned mem)
    // Did not checked other HW vendors, but
    // "AllocateMultiData" should behave as if multiple allocations
    // occurs contiguously, so this alignment should match it
    // Since this is in core library there is no proper way to
    // programaticcaly check this so it is given as a constant.
    //
    // TODO: Programatically check this for each HW vendor that this
    // code is compiled
    return 256;
}

template <MemoryC Memory, ImplicitLifetimeC... Args>
void AllocateMultiData(Tuple<Span<Args>&...> spans, Memory& memory,
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
    Detail::CalculateSpans(spans, offset, ptr, alignedSizeList, countList);

    assert(totalSize == offset);
}

template <MemoryC Memory, ImplicitLifetimeC... Args>
Tuple<Span<Args>...> AllocateMultiData(Memory& memory,
                                       const std::array<size_t, sizeof...(Args)>& countList,
                                       size_t alignment)
{
    Tuple<Span<Args>...> result;
    Tuple<Span<Args>&...> resultRef = ToTupleRef(result);
    AllocateMultiData(resultRef, memory, countList, alignment);
    return result;
}

template<ImplicitLifetimeC T, MemoryC Memory>
std::vector<Span<T>>
AllocateSegmentedData(Memory& memory, const std::vector<size_t>& counts,
                      size_t alignment)
{
    std::vector<size_t> alignedByteOffsets(counts.size() + 1);
    alignedByteOffsets[0] = 0;

    std::transform_inclusive_scan
    (
        counts.cbegin(), counts.cend(), alignedByteOffsets.begin() + 1,
        std::plus{},
        [alignment](size_t s)
        {
            return Math::NextMultiple(s * sizeof(T), alignment);
        }
    );
    //
    size_t totalSize = alignedByteOffsets.back();
    memory.ResizeBuffer(totalSize);
    Span<Byte> allBytes = Span<Byte>(static_cast<Byte*>(memory), memory.Size());
    //
    std::vector<Span<T>> result;
    result.reserve(counts.size());
    for(size_t i = 0; i < counts.size(); i++)
    {
        size_t byteStart = alignedByteOffsets[i];
        size_t byteEnd = alignedByteOffsets[i + 1];
        size_t byteCount = byteEnd - byteStart;
        assert(byteStart % sizeof(T) == 0);
        assert(byteCount % sizeof(T) == 0);

        Span<Byte> mem = allBytes.subspan(byteStart, byteCount);
        Span<T> resultingSpan = RepurposeAlloc<T>(mem);
        result.push_back(resultingSpan);
    }
    return result;
}


template <class Memory>
requires requires(Memory m) { {m.ResizeBuffer(size_t{})} -> std::same_as<void>; }
std::vector<size_t> AllocateTextureSpace(Memory& memory,
                                         const std::vector<size_t>& sizes,
                                         const std::vector<size_t>& alignments)
{
    assert(sizes.size() == alignments.size());
    std::vector<size_t> offsets(sizes.size());

    size_t totalSize = 0;
    for(size_t i = 0; i < sizes.size(); i++)
    {
        size_t alignedSize = Math::NextMultiple(sizes[i], alignments[i]);
        offsets[i] = totalSize;
        totalSize += alignedSize;
    }

    // Allocate Memory
    memory.ResizeBuffer(totalSize);
    return offsets;
}

template<size_t N>
size_t RequiredAllocation(const std::array<size_t, N>& byteSizeList,
                          size_t alignment)
{
    size_t result = 0;
    for(size_t i = 0; i < N; i++)
        result += Math::NextMultiple(byteSizeList[i], alignment);
    return result;
}
template <ImplicitLifetimeC Left, ImplicitLifetimeC Right>
requires RepurposeAllocRequirements<Left, Right>
constexpr Span<Left> RepurposeAlloc(Span<Right> rhs)
{
    using ByteT = std::conditional_t<std::is_const_v<Right>, const Byte, Byte>;
    size_t elementCount = rhs.size_bytes() / sizeof(Left);
    // TODO: Check if this is UB (probably is)
    ByteT* rawPtr = reinterpret_cast<ByteT*>(rhs.data());
    assert(std::uintptr_t(rawPtr) % alignof(Left) == 0);
    Left* leftPtr = std::launder(reinterpret_cast<Left*>(rawPtr));
    return Span<Left>(leftPtr, elementCount);
}

inline void VectorBackedMemory::ResizeBuffer(size_t s)
{
    v.resize(s);
}

inline size_t VectorBackedMemory::Size() const
{
    return v.size();
}

inline VectorBackedMemory::operator const Byte*() const
{
    return reinterpret_cast<const Byte*>(v.data());
}

inline VectorBackedMemory::operator Byte*()
{
    return reinterpret_cast<Byte*>(v.data());
}

static_assert(MemoryC<VectorBackedMemory>,
              "\"VectorBackedMemory\" does not "
              "satisfy MemoryC concept!");

}