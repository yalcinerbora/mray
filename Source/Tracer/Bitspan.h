#pragma once

#include "Core/Types.h"
#include "Device/GPUAtomic.h"
#include <concepts>

// Bitspan
// Not exact match of a std::span but suits the needs
// and only dynamic extent
template<std::unsigned_integral T>
class Bitspan
{
    private:
    T* data;
    uint32_t            size;

    public:
    MRAY_HYBRID constexpr           Bitspan();
    MRAY_HYBRID constexpr           Bitspan(T*, uint32_t bitcount);
    MRAY_HYBRID constexpr           Bitspan(Span<T>);

    MRAY_HYBRID constexpr bool      operator[](uint32_t index) const;
    // Hard to return reference for modification
    MRAY_GPU    constexpr void      SetBitParallel(uint32_t index, bool) const requires (!std::is_const_v<T>);
    MRAY_HYBRID constexpr void      SetBit(uint32_t index, bool) const requires (!std::is_const_v<T>);
    MRAY_HYBRID constexpr uint32_t  Size() const;
    MRAY_HYBRID constexpr uint32_t  ByteSize() const;

    MRAY_HYBRID constexpr uint32_t* Data();
    MRAY_HYBRID constexpr
    const uint32_t*                 Data() const;

    MRAY_HYBRID constexpr Span<T>   AsSpan() const;

    static  MRAY_HYBRID T           CountT(uint32_t bitCount);
};

template<class T>
constexpr Bitspan<const T> ToConstSpan(Bitspan<T> s)
{
    return Bitspan<const T>(s.Data(), s.Size());
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitspan<T>::Bitspan()
    : data(nullptr)
    , size(0u)
{}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitspan<T>::Bitspan(T* ptr, uint32_t bitcount)
    : data(ptr)
    , size(bitcount)
{}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Bitspan<T>::Bitspan(Span<T> s)
    : data(s.data())
    , size(static_cast<uint32_t>(s.size() * sizeof(T)))
{}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Bitspan<T>::operator[](uint32_t index) const
{
    assert(index < size && "Out of range access on bitspan!");
    T wordIndex = static_cast<T>(index / sizeof(T));
    T wordLocalIndex = static_cast<T>(index % sizeof(T));

    return (data[wordIndex] >> wordLocalIndex) & T(0x1);
}

template <std::unsigned_integral T>
MRAY_GPU MRAY_CGPU_INLINE
constexpr void Bitspan<T>::SetBitParallel(uint32_t index, bool v) const requires (!std::is_const_v<T>)
{
    assert(index < size && "Out of range access on bitspan!");
    T wordIndex = static_cast<T>(index / sizeof(T));
    T wordLocalIndex = static_cast<T>(index % sizeof(T));
    if(v)
    {
        // If we are setting a bit
        // we use atomic or, identity of or is zero (for unused parts)
        T localMask = (T(1) << wordLocalIndex);
        DeviceAtomic::AtomicOr(data[wordIndex], localMask);
    }
    else
    {
        // If we are clearing a bit
        // we use atomic and, identity of and is one
        T localMask = ~(T(1) << wordLocalIndex);
        DeviceAtomic::AtomicAnd(data[wordIndex], localMask);
    }
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Bitspan<T>::SetBit(uint32_t index, bool v) const requires (!std::is_const_v<T>)
{
    assert(index < size && "Out of range access on bitspan!");
    T wordIndex = static_cast<T>(index / sizeof(T));
    T wordLocalIndex = static_cast<T>(index % sizeof(T));
    if(v)
    {
        // If we are setting a bit
        // we use atomic or, identity of or is zero (for unused parts)
        T localMask = (T(1) << wordLocalIndex);
        data[wordIndex] |= localMask;
    }
    else
    {
        // If we are clearing a bit
        // we use atomic and, identity of and is one
        T localMask = ~(T(1) << wordLocalIndex);
        data[wordIndex] &= localMask;
    }
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr uint32_t Bitspan<T>::Size() const
{
    return size;
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr uint32_t Bitspan<T>::ByteSize() const
{
    return Math::NextMultiple<uint32_t>(size, static_cast<uint32_t>(sizeof(T)));
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr uint32_t* Bitspan<T>::Data()
{
    return data;
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const uint32_t* Bitspan<T>::Data() const
{
    return data;
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Span<T> Bitspan<T>::AsSpan() const
{
    return Span<T>(data, Math::DivideUp<uint32_t>(size, sizeof(T)));
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
T Bitspan<T>::CountT(uint32_t bitCount)
{
    return Math::DivideUp<uint32_t>(bitCount, sizeof(T));
}