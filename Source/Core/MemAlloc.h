#pragma once

#include <concepts>
#include <type_traits>

#include "Math.h"
#include "Definitions.h"

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
    (!std::is_same_v<std::remove_cvref_t<Right>, Byte> &&
      alignof(std::remove_cvref_t<Right>) >= alignof(std::remove_cvref_t<Left>)) ||
    std::is_same_v<std::remove_cvref_t<Right>, Byte>
);

constexpr size_t DefaultSystemAlignment();

template <MemoryC Memory, RelaxedLifetimeC... Args>
void AllocateMultiData(Tuple<Span<Args>&...> spans, Memory& memory,
                       const std::array<size_t, sizeof...(Args)>& countList,
                       size_t alignment = DefaultSystemAlignment());

template <MemoryC Memory, RelaxedLifetimeC... Args>
Tuple<Span<Args>...> AllocateMultiData(Memory& memory,
                                       const std::array<size_t, sizeof...(Args)>& countList,
                                       size_t alignment = DefaultSystemAlignment());

template<RelaxedLifetimeC T, MemoryC Memory>
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

// Simple wrappers to utilize "MultiData" allocation scheme
struct AlignedMemory
{
    void*   mem;
    size_t  alignment;
    size_t  size;
    size_t  allocSize;
    bool    neverDecrease;

    public:
    // Constructors & Destructor
    // TODO: Implement Move/Copy etc.
    // Currently, we do not need it.
                    AlignedMemory(size_t alignment = DefaultSystemAlignment(),
                                  bool neverDecrease = true);
                    AlignedMemory(size_t size,
                                  size_t alignment = DefaultSystemAlignment(),
                                  bool neverDecrease = true);
                    AlignedMemory(const AlignedMemory&) = delete;
                    AlignedMemory(AlignedMemory&&) = delete;
    AlignedMemory&  operator=(const AlignedMemory&) = delete;
    AlignedMemory&  operator=(AlignedMemory&&) = delete;
                    ~AlignedMemory();

    void            ResizeBuffer(size_t newSize);
    size_t          Size() const;

    explicit operator Byte*();
    explicit operator const Byte*() const;

};

static_assert(MemoryC<AlignedMemory>,
              "\"MemAlloc::AlignedMemory\" does not "
              "satisfy MemoryC concept!");

}

namespace MemAlloc::Detail
{

    template<class... Tp>
    constexpr size_t AcquireTotalSize(std::array<size_t, sizeof...(Tp)>& alignedSizeList,
                                      const std::array<size_t, sizeof...(Tp)>& countList,
                                      size_t alignment)
    {
        constexpr size_t N = sizeof...(Tp);
        constexpr std::array<size_t, N> sizeList = {sizeof(Tp)...};

        size_t totalSize = 0;
        for(uint32_t i = 0; i < N; i++)
        {
            using Math::NextMultiple;
            alignedSizeList[i] = NextMultiple(sizeList[i] * countList[i], alignment);
            totalSize += alignedSizeList[i];
        }
        return totalSize;
    }

    template<std::size_t... Is, class... Tp>
    constexpr size_t CalculateSpans(Tuple<Span<Tp>&...>& t, Byte* memory,
                                    const std::array<size_t, sizeof...(Tp)>& alignedSizeList,
                                    const std::array<size_t, sizeof...(Tp)>& countList,
                                    std::index_sequence<Is...>)
    {
        constexpr size_t N = sizeof...(Tp);
        std::array<size_t, N> byteOffsets = {};
        size_t offset = 0;
        for(uint32_t i = 0; i < N; i++)
        {
            byteOffsets[i] = offset;
            offset += alignedSizeList[i];
        }
        if constexpr(MRAY_IS_DEBUG)
        {
            std::array<size_t, N> sizes = {sizeof(Tp)...};
            for(uint32_t i = 0; i < N; i++)
                assert(alignedSizeList[i] / sizes[i] >= countList[i]);
        }
        // Param pack expansion
        (
            // Expanding statement...
            (get<Is>(t) = Span<Tp>(std::launder(reinterpret_cast<Tp*>(memory + byteOffsets[Is])),
                                                countList[Is])),
            // Expand
            ...
        );
        return alignedSizeList.back() + byteOffsets.back();
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
    // programmatically check this so it is given as a constant.
    //
    // TODO: Programmatically check this for each HW vendor that this
    // code is compiled
    return 256;
}

template <MemoryC Memory, RelaxedLifetimeC... Args>
void AllocateMultiData(Tuple<Span<Args>&...> spans, Memory& memory,
                       const std::array<size_t, sizeof...(Args)>& countList,
                       size_t alignment)
{
    std::array<size_t, sizeof...(Args)> alignedSizeList;
    // Acquire total size & allocation size of each array
    size_t totalSize = Detail::AcquireTotalSize<Args...>(alignedSizeList,
                                                         countList,
                                                         alignment);
    // Allocate Memory
    memory.ResizeBuffer(totalSize);
    Byte* ptr = static_cast<Byte*>(memory);
    // Populate pointers
    [[maybe_unused]]
    size_t offset = Detail::CalculateSpans(spans, ptr, alignedSizeList, countList,
                                           std::make_index_sequence<sizeof...(Args)>{});
    assert(totalSize == offset);
}

template <MemoryC Memory, RelaxedLifetimeC... Args>
Tuple<Span<Args>...> AllocateMultiData(Memory& memory,
                                       const std::array<size_t, sizeof...(Args)>& countList,
                                       size_t alignment)
{
    Tuple<Span<Args>...> result;
    Tuple<Span<Args>&...> resultRef = ToTupleRef(result);
    AllocateMultiData(resultRef, memory, countList, alignment);
    return result;
}

template<RelaxedLifetimeC T, MemoryC Memory>
std::vector<Span<T>>
AllocateSegmentedData(Memory& memory, const std::vector<size_t>& counts,
                      size_t alignment)
{
    std::vector<size_t> alignedByteOffsets(counts.size() + 1u);
    alignedByteOffsets[0] = 0;

    for(size_t i = 1; i < alignedByteOffsets.size(); i++)
    {
        size_t v = Math::NextMultiple(counts[i - 1] * sizeof(T), alignment);
        alignedByteOffsets[i] = alignedByteOffsets[i - 1] + v;
    }

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

Pair<double, std::string_view> ConvertMemSizeToString(size_t size);

}