#pragma once

#include "Vector.h"

// Quaternion, layout is (w, x, y, z)
// where; v[0] = w, v[1] = x, v[2] = y, v[3] = z
// No need to dictate alignment here vector will handle it
template<FloatingPointC T>
class Quat
{
    public:
    using InnerType = T;

    private:
    Vector<4, T>            v;

    public:
    // Constructors & Destructor
    constexpr               Quat() = default;
    MRAY_HYBRID constexpr   Quat(T w, T x, T y, T z);
    MRAY_HYBRID constexpr   Quat(const T*);
    MRAY_HYBRID constexpr   Quat(const Vector<4, T>& vec);
    MRAY_HYBRID constexpr   Quat(T angle, const Vector<3, T>& axis);

    // Type casting
    MRAY_HYBRID explicit    operator Vector<4, T>& ();
    MRAY_HYBRID explicit    operator const Vector<4, T>& () const;
    MRAY_HYBRID explicit    operator T* ();
    MRAY_HYBRID explicit    operator const T* () const;

    // Access
    MRAY_HYBRID constexpr T&         operator[](unsigned int);
    MRAY_HYBRID constexpr const T&   operator[](unsigned int) const;

    // Operators
    MRAY_HYBRID constexpr Quat  operator*(const Quat&) const;
    MRAY_HYBRID constexpr Quat  operator*(T) const;
    MRAY_HYBRID constexpr Quat  operator+(const Quat&) const;
    MRAY_HYBRID constexpr Quat  operator-(const Quat&) const;
    MRAY_HYBRID constexpr Quat  operator-() const;
    MRAY_HYBRID constexpr Quat  operator/(T) const;

    MRAY_HYBRID constexpr void  operator*=(const Quat&);
    MRAY_HYBRID constexpr void  operator*=(T);
    MRAY_HYBRID constexpr void  operator+=(const Quat&);
    MRAY_HYBRID constexpr void  operator-=(const Quat&);
    MRAY_HYBRID constexpr void  operator/=(T);

    // Logic
    MRAY_HYBRID constexpr bool  operator==(const Quat&) const;
    MRAY_HYBRID constexpr bool  operator!=(const Quat&) const;

    // Utility
    MRAY_HYBRID NO_DISCARD constexpr Quat   Normalize() const;
    MRAY_HYBRID constexpr Quat&             NormalizeSelf();
    MRAY_HYBRID constexpr T                 Length() const;
    MRAY_HYBRID constexpr T                 LengthSqr() const;
    MRAY_HYBRID NO_DISCARD constexpr Quat   Conjugate() const;
    MRAY_HYBRID constexpr Quat&             ConjugateSelf();
    MRAY_HYBRID constexpr T                 Dot(const Quat&) const;
    MRAY_HYBRID constexpr Vector<3, T>      ApplyRotation(const Vector<3, T>&) const;
    MRAY_HYBRID constexpr Vector<3, T>      ApplyInvRotation(const Vector<3, T>&) const;
    MRAY_HYBRID constexpr bool              HasNaN() const;

    // Optimized version of "ApplyInvRotation(Vector3::_Axis())"
    // Where '_' X, Y, Z. By definition rotation defines a orthonormal
    // basis so we can fetch the axes via these functions.
    // These are implemented since MSVC did not optimize the above
    // code fully. It is an obvious optimization. We just ignore
    // most of the data since they are  zero.
    MRAY_HYBRID constexpr Vector<3, T>      OrthoBasisX() const;
    MRAY_HYBRID constexpr Vector<3, T>      OrthoBasisY() const;
    MRAY_HYBRID constexpr Vector<3, T>      OrthoBasisZ() const;

    static MRAY_HYBRID constexpr Quat<T>    NLerp(const Quat<T>& start,
                                                  const Quat<T>& end, T t);
    static MRAY_HYBRID constexpr Quat<T>    SLerp(const Quat<T>& start,
                                                  const Quat<T>& end, T t);
    static MRAY_HYBRID constexpr Quat<T>    BarySLerp(const Quat<T>& q0,
                                                      const Quat<T>& q1,
                                                      const Quat<T>& q2,
                                                      T a, T b);
    static MRAY_HYBRID constexpr Quat<T>    RotationBetween(const Vector<3, T>& a,
                                                            const Vector<3, T>& b);
    static MRAY_HYBRID constexpr Quat<T>    RotationBetweenZAxis(const Vector<3, T>& b);
    static MRAY_HYBRID constexpr Quat<T>    Identity();
};

// Sanity Checks for Quaternion
static_assert(std::is_trivially_default_constructible_v<Quaternion> == true, "Quaternion has to be trivially destructible");
static_assert(std::is_trivially_destructible_v<Quaternion> == true, "Quaternion has to be trivially destructible");
static_assert(std::is_trivially_copyable_v<Quaternion> == true, "Quaternion has to be trivially copyable");
static_assert(std::is_polymorphic_v<Quaternion> == false, "Quaternion should not be polymorphic");
static_assert(ImplicitLifetimeC<Quaternion>, "Quaternions should have implicit lifetime");

// Left Scalar operators
template<class T>
MRAY_HYBRID Quat<T> operator*(T, const Quat<T>&);

// Static Utility
namespace TransformGen
{
    template<std::floating_point T>
    MRAY_HYBRID Quat<T> ToSpaceQuat(const Vector<3, T>& x,
                                    const Vector<3, T>& y,
                                    const Vector<3, T>& z);
    template<std::floating_point T>
    MRAY_HYBRID Quat<T> ToInvSpaceQuat(const Vector<3, T>& x,
                                       const Vector<3, T>& y,
                                       const Vector<3, T>& z);
}

// Implementation
#include "Quaternion.hpp"

// Quaternion Concept
template<class T>
concept QuaternionC = requires()
{
    std::is_same_v<T, QuaternionF>  ||
    std::is_same_v<T, QuaternionD>  ||
    std::is_same_v<T, Quaternion>;
};