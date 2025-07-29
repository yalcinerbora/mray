#pragma once
/**

Arbitrary sized axis aligned bounding box.

N should be 2, 3 or 4 at most.

These are convenience register classes for GPU.

*/

#include "Vector.h"

template<unsigned int N, FloatC T>
class AABB
{
    static_assert(N == 2 || N == 3 || N == 4, "AABB size should be 2, 3 or 4");

    public:
    using InnerType = T;

    public:
    static constexpr int            AABBVertexCount = 8;
    friend struct                   AABBOffsetChecker;

    private:
    Vector<N, T>                    min;
    Vector<N, T>                    max;

    public:
    // Constructors & Destructor
    constexpr           AABB() = default;
    MR_PF_DECL explicit AABB(const Vector<N, T>& min,
                             const Vector<N, T>& max) noexcept;
    MR_PF_DECL explicit AABB(Span<const T, N> dataMin,
                             Span<const T, N> dataMax) noexcept;

    // Accessors
    MR_PF_DECL const Vector<N, T>&   Min() const noexcept;
    MR_PF_DECL const Vector<N, T>&   Max() const noexcept;
    MR_PF_DECL Vector<N, T>          Min() noexcept;
    MR_PF_DECL Vector<N, T>          Max() noexcept;

    // Mutators
    MR_PF_DECL_V void SetMin(const Vector<N, T>&) noexcept;
    MR_PF_DECL_V void SetMax(const Vector<N, T>&) noexcept;

    // Functionality
    MR_PF_DECL Vector<N, T> GeomSpan() const noexcept;
    MR_PF_DECL Vector<N, T> Centroid() const noexcept;
    MR_PF_DECL AABB         Union(const AABB&) const noexcept;
    MR_PF_DECL bool         IsInside(const Vector<N, T>&) const noexcept;
    MR_PF_DECL bool         IsOutside(const Vector<N, T>&) const noexcept;
    MR_PF_DECL Vector<N,T>  FurthestCorner(const Vector<N, T>&) const noexcept;

    // Intersection
    MR_PF_DECL bool         IntersectsSphere(const Vector<N, T>& sphrPos,
                                             float sphrRadius) const noexcept;
    // Constants
    MR_PF_DECL static AABB  Zero() noexcept;
    MR_PF_DECL static AABB  Covering() noexcept;
    MR_PF_DECL static AABB  Negative() noexcept;
};

// Requirements of AABB
static_assert(std::is_trivially_default_constructible_v<AABB3> == true, "AABB3 has to be trivially destructible");
static_assert(std::is_trivially_destructible_v<AABB3> == true, "AABB3 has to be trivially destructible");
static_assert(std::is_trivially_copyable_v<AABB3> == true, "AABB3 has to be trivially copyable");
static_assert(std::is_polymorphic_v<AABB3> == false, "AABB3 should not be polymorphic");

// Implementation
#include "AABB.hpp" // CPU & GPU