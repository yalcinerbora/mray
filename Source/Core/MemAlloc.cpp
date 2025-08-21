#include "MemAlloc.h"
#include "System.h"

MemAlloc::AlignedMemory::AlignedMemory(size_t alignmentIn,
                                       bool neverDecreaseIn)
    : AlignedMemory(0, alignmentIn, neverDecreaseIn)
{}

MemAlloc::AlignedMemory::AlignedMemory(size_t sizeIn,
                                       size_t alignmentIn,
                                       bool neverDecreaseIn)
    : mem(nullptr)
    , alignment(alignmentIn)
    , size(sizeIn)
    , allocSize(size)
    , neverDecrease(neverDecreaseIn)
{
    ResizeBuffer(size);
}

MemAlloc::AlignedMemory::~AlignedMemory()
{
    #ifdef MRAY_WINDOWS
        _aligned_free(mem);
    #elif defined MRAY_LINUX
        std::free(mem);
    #endif
}

void MemAlloc::AlignedMemory::ResizeBuffer(size_t newSize)
{
    size_t oldSize = size;
    if(newSize == 0)
    {
        size = newSize;
        return;
    }
    if(neverDecrease && newSize <= allocSize)
    {
        size = newSize;
        return;
    }

    // Actually allocate
    allocSize = Math::NextMultiple(newSize, alignment);
    // TODO: This should be mmap, VirtualAlloc change these later
    AlignedFree(mem, oldSize, alignment);
    mem = AlignedAlloc(allocSize, alignment);
    size = newSize;
}
size_t MemAlloc::AlignedMemory::Size() const
{
    return size;
}

MemAlloc::AlignedMemory::operator Byte* ()
{
    return reinterpret_cast<Byte*>(mem);
}

MemAlloc::AlignedMemory::operator const Byte* () const
{
    return reinterpret_cast<Byte*>(mem);
}

Pair<double, std::string_view>
MemAlloc::ConvertMemSizeToString(size_t size)
{
    // This function is overengineered for a GUI operation.
    // This probably has better precision? (probably not)
    // has high amount memory (TiB++ of memory).
    Pair<double, std::string_view> result;
    using namespace std::string_view_literals;
    size_t shiftVal = 0;
    if(size >= 1_TiB)
    {
        result.second = "TiB"sv;
        shiftVal = 40;
    }
    else if(size >= 1_GiB)
    {
        result.second = "GiB"sv;
        shiftVal = 30;
    }
    else if(size >= 1_MiB)
    {
        result.second = "MiB"sv;
        shiftVal = 20;
    }
    else if(size >= 1_KiB)
    {
        result.second = "KiB"sv;
        shiftVal = 10;
    }
    else
    {
        result.second = "Bytes"sv;
        shiftVal = 0;
    }

    size_t mask = ((size_t(1) << shiftVal) - 1);
    size_t integer = size >> shiftVal;
    size_t decimal = mask & size;
    // Sanity check
    static_assert(std::numeric_limits<double>::is_iec559,
                  "This overengineered function requires "
                  "IEEE-754 floats.");
    static constexpr size_t DOUBLE_MANTISSA = 52;
    static constexpr size_t MANTISSA_MASK = (size_t(1) << DOUBLE_MANTISSA) - 1;
    size_t bitCount = Bit::RequiredBitsToRepresent(decimal);
    if(bitCount > DOUBLE_MANTISSA)
        decimal >>= (bitCount - DOUBLE_MANTISSA);
    else
        decimal <<= (DOUBLE_MANTISSA - bitCount);


    uint64_t dblFrac = std::bit_cast<uint64_t>(1.0);
    dblFrac |= decimal & MANTISSA_MASK;
    result.first = std::bit_cast<double>(dblFrac);
    result.first += static_cast<double>(integer) - 1.0;
    return result;
}