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