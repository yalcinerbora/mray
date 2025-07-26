#pragma once
/**

Ray struct for convenient usability.

*/
#include "Vector.h"
#include "BitFunctions.h"

#include <cstring>

template<FloatC T>
class RayT
{
    static constexpr auto CoveringTMM = Vector<2, T>(-std::numeric_limits<T>::infinity(),
                                                     std::numeric_limits<T>::infinity());
    public:
    using InnerType = T;

    private:
    Vector<3,T>                     direction;
    Vector<3,T>                     position;
    public:
    // Constructors & Destructor
    constexpr           RayT() = default;
    MR_PF_DECL explicit RayT(const Vector<3, T>& direction,
                             const Vector<3, T>& position) noexcept;
    MR_PF_DECL explicit RayT(const Vector<3, T>[2]) noexcept;

    // Intersections
    MR_PF_DECL bool  IntersectsSphere(Vector<3, T>& pos, T& t,
                                      const Vector<3, T>& sphereCenter,
                                      T sphereRadius) const noexcept;
    MR_PF_DECL bool  IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                        const Vector<3, T> triVertex[3],
                                        bool cullFace = true) const noexcept;
    MR_PF_DECL bool  IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                        const Vector<3, T>& t0,
                                        const Vector<3, T>& t1,
                                        const Vector<3, T>& t2,
                                        bool cullFace = true) const noexcept;
    MR_PF_DECL bool  IntersectsPlane(Vector<3, T>& position, T& t,
                                     const Vector<3, T>& planePos,
                                     const Vector<3, T>& normal) const noexcept;

    MR_PF_DECL bool  IntersectsAABB(Vector<2, T>& tOut,
                                    const Vector<3, T>& aabbMin,
                                    const Vector<3, T>& aabbMax,
                                    const Vector<2, T>& tMinMax = CoveringTMM) const noexcept;
    MR_PF_DECL bool  IntersectsAABB(const Vector<3, T>& min,
                                    const Vector<3, T>& max,
                                    const Vector<2, T>& tMinMax = CoveringTMM) const noexcept;
    MR_PF_DECL bool  IntersectsAABB(Vector<3, T>& pos, T& t,
                                    const Vector<3, T>& min,
                                    const Vector<3, T>& max,
                                    const Vector<2, T>& tMinMax = CoveringTMM) const noexcept;

    MR_PF_DECL RayT           NormalizeDir() const noexcept;
    MR_PF_DECL RayT&          NormalizeDirSelf() noexcept;
    MR_PF_DECL RayT           Advance(T) const noexcept;
    MR_PF_DECL RayT           Advance(T t, const Vector<3,T>& dir) const noexcept;
    MR_PF_DECL RayT&          AdvanceSelf(T) noexcept;
    MR_PF_DECL RayT&          AdvanceSelf(T t, const Vector<3, T>& dir) noexcept;
    MR_PF_DECL Vector<3,T>    AdvancedPos(T t) const noexcept;

    MR_PF_DECL RayT     Nudge(const Vector<3, T>& dir) const noexcept;
    MR_PF_DECL RayT&    NudgeSelf(const Vector<3, T>& dir) noexcept;

    MR_PF_DECL const Vector<3, T>& Dir() const noexcept;
    MR_PF_DECL const Vector<3, T>& Pos() const noexcept;
};

// Requirements of IERay
static_assert(std::is_trivially_default_constructible_v<Ray> == true, "Ray has to be trivially destructible");
static_assert(std::is_trivially_destructible_v<Ray> == true, "Ray has to be trivially destructible");
static_assert(std::is_trivially_copyable_v<Ray> == true, "Ray has to be trivially copyable");
static_assert(std::is_polymorphic_v<Ray> == false, "Ray should not be polymorphic");
static_assert(ImplicitLifetimeC<Ray>, "Rays should have implicit lifetime");

#include "Ray.hpp"
