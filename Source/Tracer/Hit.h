#pragma once

#include "Core/Types.h"

// Type ereased HitData
// For triangles and spheres it is Vector2
// For other things it may be other sized floats so this data
// is held via (void*)
// TODO: check std::variant<...>, and CUDA compatiblity

template<class... Args>
class MetaHitPtrT
{
    private:
    Variant<Args...>*               ptr;

    public:
    // Constructors & Destructor
    MRAY_HYBRID constexpr           MetaHitPtrT();
    MRAY_HYBRID constexpr           MetaHitPtrT(Byte* dPtr, uint32_t combinedSize);
    // Methods
    template<class T>
    MRAY_HYBRID constexpr T&        Ref(uint32_t i);
    template<class T>
    MRAY_HYBRID constexpr const T&  Ref(uint32_t i) const;
};

template<class... Args>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr MetaHitPtrT<Args...>::MetaHitPtrT()
    : ptr(nullptr)
{}

template<class... Args>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr MetaHitPtrT<Args...>::MetaHitPtrT(Byte* dPtr, uint32_t)
    : ptr(static_cast<std::variant<Args...>>(dPtr))
{}

template<class... Args>
template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T& MetaHitPtrT<Args...>::Ref(uint32_t i)
{
    return std::get<T>(ptr[i]);
}

template<class... Args>
template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const T& MetaHitPtrT<Args...>::Ref(uint32_t i) const
{
    return std::get<T>(ptr[i]);
}