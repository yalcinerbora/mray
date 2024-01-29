#pragma once

#pragma once

/**

Arbitrary sized axis aligned bounding box.

N should be 2, 3 or 4 at most.

These are convenience register classes for GPU.

*/

#include "Vector.h"

template<unsigned int N, FloatingPointC T>
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
    constexpr                   AABB() = default;
    MRAY_HYBRID constexpr       AABB(const Vector<N, T>& min,
                                     const Vector<N, T>& max);
    MRAY_HYBRID constexpr       AABB(const T* dataMin,
                                     const T* dataMax);

    template <class... ArgsMin, class... ArgsMax>
    requires (sizeof...(ArgsMin) == N) && (std::convertible_to<T, ArgsMin> && ...) &&
             (sizeof...(ArgsMax) == N) && (std::convertible_to<T, ArgsMax> && ...)
    MRAY_HYBRID constexpr       AABB(const ArgsMin... dataMin,
                                     const ArgsMax... dataMax);

    // Accessors
    MRAY_HYBRID constexpr const Vector<N, T>&   Min() const;
    MRAY_HYBRID constexpr const Vector<N, T>&   Max() const;
    MRAY_HYBRID constexpr Vector<N, T>          Min();
    MRAY_HYBRID constexpr Vector<N, T>          Max();

    // Mutators
    MRAY_HYBRID constexpr void                  SetMin(const Vector<N, T>&);
    MRAY_HYBRID constexpr void                  SetMax(const Vector<N, T>&);

    // Functionality
    MRAY_HYBRID constexpr Vector<N, T>          Span() const;
    MRAY_HYBRID constexpr Vector<N, T>          Centroid() const;
    MRAY_HYBRID NO_DISCARD constexpr AABB       Union(const AABB&) const;
    MRAY_HYBRID constexpr AABB&                 UnionSelf(const AABB&);
    MRAY_HYBRID constexpr bool                  IsInside(const Vector<N, T>&) const;
    MRAY_HYBRID constexpr bool                  IsOutside(const Vector<N, T>&) const;
    MRAY_HYBRID constexpr Vector<N,T>           FurthestCorner(const Vector<N, T>&) const;

    // Intersection
    MRAY_HYBRID constexpr bool                  IntersectsSphere(const Vector<N, T>& sphrPos,
                                                                 float sphrRadius) const;

    static MRAY_HYBRID constexpr AABB           Zero();
    static MRAY_HYBRID constexpr AABB           Covering();
    static MRAY_HYBRID constexpr AABB           Negative();
};

// Requirements of Vectors
static_assert(std::is_trivially_default_constructible_v<AABB3> == true, "AABB3 has to be trivially destructible");
static_assert(std::is_trivially_destructible_v<AABB3> == true, "AABB3 has to be trivially destructible");
static_assert(std::is_trivially_copyable_v<AABB3> == true, "AABB3 has to be trivially copyable");
static_assert(std::is_polymorphic_v<AABB3> == false, "AABB3 should not be polymorphic");

struct AABBOffsetChecker
{
    // Some Sanity Traits
    static_assert(offsetof(AABB2f, min) == 0,
                  "AABB2f::min is not properly aligned for contiguous mem read/write");
    static_assert(offsetof(AABB2f, max) == sizeof(Vector<2, float>),
                  "AABB2f:: max is not properly aligned for contiguous mem read/write");
    static_assert(offsetof(AABB3f, min) == 0,
                  "AABB3f::min is not properly aligned for contiguous mem read/write");
    static_assert(offsetof(AABB3f, max) == sizeof(Vector<3, float>),
                  "AABB3f:: max is not properly aligned for contiguous mem read/write");
    static_assert(offsetof(AABB4f, min) == 0,
                  "AABB4f::min is not properly aligned for contiguous mem read/write");
    static_assert(offsetof(AABB4f, max) == sizeof(Vector<4, float>),
                  "AABB4f:: max is not properly aligned for contiguous mem read/write");
    //
    static_assert(offsetof(AABB2d, min) == 0,
                  "AABB2d::min is not properly aligned for contiguous mem read/write");
    static_assert(offsetof(AABB2d, max) == sizeof(Vector<2, double>),
                  "AABB2d:: max is not properly aligned for contiguous mem read/write");
    static_assert(offsetof(AABB3d, min) == 0,
                  "AABB3d::min is not properly aligned for contiguous mem read/write");
    static_assert(offsetof(AABB3d, max) == sizeof(Vector<3, double>),
                  "AABB3d:: max is not properly aligned for contiguous mem read/write");
    static_assert(offsetof(AABB4d, min) == 0,
                  "AABB4d::min is not properly aligned for contiguous mem read/write");
    static_assert(offsetof(AABB4d, max) == sizeof(Vector<4, double>),
                  "AABB4d:: max is not properly aligned for contiguous mem read/write");
};

// Implementation
#include "AABB.hpp" // CPU & GPU