#pragma once

#include "Core/Types.h"
#include "Core/Vector.h"

// Size ereased vector of floats
// For triangles and spheres it is Vector2 (N=2)
// For other things it may be other sized floats
// N is compile time constant at most can be 3 maybe?
template<uint32_t N>
class MetaHitT
{
    public:
    static constexpr uint32_t MaxDim = N;
    private:
    Vector<N, Float>        vec;

    public:
    // Constructors & Destructor
    template<uint32_t M>
    requires(M <= N)
    MRAY_HYBRID constexpr   MetaHitT(const Vector<M, Float>&);

    // Methods
    template<uint32_t M>
    requires(M <= N)
    MRAY_HYBRID constexpr
    Vector<M, Float> AsVector() const;
};

template<uint32_t N>
template<uint32_t M>
requires(M <= N)
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr MetaHitT<N>::MetaHitT(const Vector<M, Float>& v)
{
    UNROLL_LOOP
    for(uint32_t i = 0; i < N; i++)
    {
        vec[i] = v[i];
    }
}

template<uint32_t N>
template<uint32_t M>
requires(M <= N)
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<M, Float>
MetaHitT<N>::AsVector() const
{
    return vec;
}
