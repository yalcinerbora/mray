#pragma once
/**

Ray struct for convenient usability.

*/
#include "Vector.h"
#include "MathConstants.h"

template<FloatingPointC T>
class RayT
{
    public:
    using InnerType = T;

    private:
    Vector<3,T>                     direction;
    Vector<3,T>                     position;
    public:
    // Constructors & Destructor
    constexpr                       RayT() = default;
    MRAY_HYBRID constexpr explicit  RayT(const Vector<3, T>& direction,
                                         const Vector<3, T>& position);
    MRAY_HYBRID constexpr explicit  RayT(const Vector<3, T>[2]);

    // Intersections
    MRAY_HYBRID constexpr bool     IntersectsSphere(Vector<3, T>& pos, T& t,
                                                    const Vector<3, T>& sphereCenter,
                                                    T sphereRadius) const;
    MRAY_HYBRID constexpr bool     IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                                      const Vector<3, T> triVertex[3],
                                                      bool cullFace = true) const;
    MRAY_HYBRID constexpr bool     IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                                      const Vector<3, T>& t0,
                                                      const Vector<3, T>& t1,
                                                      const Vector<3, T>& t2,
                                                      bool cullFace = true) const;
    MRAY_HYBRID constexpr bool     IntersectsPlane(Vector<3, T>& position, T& t,
                                                   const Vector<3, T>& planePos,
                                                   const Vector<3, T>& normal);

    MRAY_HYBRID constexpr bool     IntersectsAABB(Vector<2, T>& tOut,
                                                  const Vector<3, T>& aabbMin,
                                                  const Vector<3, T>& aabbMax,
                                                  const Vector<2, T>& tMinMax = Vector<2, T>(-std::numeric_limits<T>::infinity(),
                                                                                             std::numeric_limits<T>::infinity())) const;
    MRAY_HYBRID constexpr bool     IntersectsAABB(const Vector<3, T>& min,
                                                  const Vector<3, T>& max,
                                                  const Vector<2, T>& tMinMax = Vector<2, T>(-std::numeric_limits<T>::infinity(),
                                                                                             std::numeric_limits<T>::infinity())) const;
    MRAY_HYBRID constexpr bool     IntersectsAABB(Vector<3, T>& pos, T& t,
                                                  const Vector<3, T>& min,
                                                  const Vector<3, T>& max,
                                                  const Vector<2, T>& tMinMax = Vector<2, T>(-std::numeric_limits<T>::infinity(),
                                                                                             std::numeric_limits<T>::infinity())) const;

    NO_DISCARD MRAY_HYBRID constexpr RayT           NormalizeDir() const;
               MRAY_HYBRID constexpr RayT&          NormalizeDirSelf();
    NO_DISCARD MRAY_HYBRID constexpr RayT           Advance(T) const;
    NO_DISCARD MRAY_HYBRID constexpr RayT           Advance(T t, const Vector<3,T>& dir) const;
               MRAY_HYBRID constexpr RayT&          AdvanceSelf(T);
               MRAY_HYBRID constexpr RayT&          AdvanceSelf(T t, const Vector<3, T>& dir);
    NO_DISCARD MRAY_HYBRID constexpr Vector<3,T>    AdvancedPos(T t) const;

    // TODO: Make these constexpr later (resolve memcpy pattern etc)
    MRAY_HYBRID NO_DISCARD RayT                     Nudge(const Vector<3, T>& dir) const;
    MRAY_HYBRID RayT&                               NudgeSelf(const Vector<3, T>& dir);

    MRAY_HYBRID constexpr const Vector<3, T>&       Dir() const;
    MRAY_HYBRID constexpr const Vector<3, T>&       Pos() const;
};

// Requirements of IERay
static_assert(std::is_trivially_default_constructible_v<Ray> == true, "Ray has to be trivially destructible");
static_assert(std::is_trivially_destructible_v<Ray> == true, "Ray has to be trivially destructible");
static_assert(std::is_trivially_copyable_v<Ray> == true, "Ray has to be trivially copyable");
static_assert(std::is_polymorphic_v<Ray> == false, "Ray should not be polymorphic");
static_assert(ImplicitLifetimeC<Ray>, "Rays should have implicit lifetime");

#include "Ray.hpp"
