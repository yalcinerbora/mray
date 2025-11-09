#pragma once

#include "Vector.h"

// Quaternion, layout is (w, x, y, z)
// where; v[0] = w, v[1] = x, v[2] = y, v[3] = z
// No need to dictate alignment here vector will handle it
template<FloatC T>
class Quat
{
    public:
    using InnerType = T;

    private:
    Vector<4, T>            v;

    public:
    // Constructors & Destructor
    constexpr             Quat() = default;
    MR_PF_DECL_V explicit Quat(T w, T x, T y, T z) noexcept;
    MR_PF_DECL_V explicit Quat(const T*) noexcept;
    MR_PF_DECL_V explicit Quat(const Vector<4, T>& vec) noexcept;
    MR_PF_DECL_V explicit Quat(T angle, const Vector<3, T>& axis) noexcept;

    // Type casting
    MR_PF_DECL explicit operator Vector<4, T>&() noexcept;
    MR_PF_DECL explicit operator const Vector<4, T>&() const noexcept;
    MR_PF_DECL explicit operator T*() noexcept;
    MR_PF_DECL explicit operator const T*() const noexcept;

    // Access
    MR_PF_DECL T&         operator[](unsigned int) noexcept;
    MR_PF_DECL const T&   operator[](unsigned int) const noexcept;

    // Operators
    MR_PF_DECL Quat  operator*(const Quat&) const noexcept;
    MR_PF_DECL Quat  operator*(T) const noexcept;
    MR_PF_DECL Quat  operator+(const Quat&) const noexcept;
    MR_PF_DECL Quat  operator-(const Quat&) const noexcept;
    MR_PF_DECL Quat  operator-() const noexcept;
    MR_PF_DECL Quat  operator/(T) const noexcept;

    MR_PF_DECL_V void  operator*=(const Quat&) noexcept;
    MR_PF_DECL_V void  operator*=(T) noexcept;
    MR_PF_DECL_V void  operator+=(const Quat&) noexcept;
    MR_PF_DECL_V void  operator-=(const Quat&) noexcept;
    MR_PF_DECL_V void  operator/=(T) noexcept;
    // Logic
    MR_PF_DECL bool  operator==(const Quat&) const noexcept;
    MR_PF_DECL bool  operator!=(const Quat&) const noexcept;

    // Utility
    MR_PF_DECL Quat         Normalize() const noexcept;
    MR_PF_DECL T            Length() const noexcept;
    MR_PF_DECL T            LengthSqr() const noexcept;
    MR_PF_DECL Quat         Conjugate() const noexcept;
    MR_PF_DECL T            Dot(const Quat&) const noexcept;
    MR_PF_DECL Vector<3, T> ApplyRotation(const Vector<3, T>&) const noexcept;
    MR_PF_DECL Vector<3, T> ApplyInvRotation(const Vector<3, T>&) const noexcept;
    MR_PF_DECL bool         HasNaN() const noexcept;

    // Optimized version of "ApplyInvRotation(Vector3::_Axis())"
    // Where '_' X, Y, Z. By definition rotation defines a orthonormal
    // basis so we can fetch the axes via these functions.
    // These are implemented since MSVC did not optimize the above
    // code fully. It is an obvious optimization. We just ignore
    // most of the data since they are zero.
    MR_PF_DECL Vector<3, T>     OrthoBasisX() const noexcept;
    MR_PF_DECL Vector<3, T>     OrthoBasisY() const noexcept;
    MR_PF_DECL Vector<3, T>     OrthoBasisZ() const noexcept;

    MR_PF_DECL static Quat<T>   NLerp(const Quat<T>& start,
                                      const Quat<T>& end, T t) noexcept;
    MR_PF_DECL static Quat<T>   SLerp(const Quat<T>& start,
                                      const Quat<T>& end, T t) noexcept;
    MR_PF_DECL static Quat<T>   BarySLerp(const Quat<T>& q0,
                                          const Quat<T>& q1,
                                          const Quat<T>& q2,
                                          T a, T b) noexcept;
    MR_PF_DECL static Quat<T>    Identity() noexcept;
};

// Sanity Checks for Quaternion
static_assert(std::is_trivially_default_constructible_v<Quaternion> == true, "Quaternion has to be trivially destructible");
static_assert(std::is_trivially_destructible_v<Quaternion> == true, "Quaternion has to be trivially destructible");
static_assert(std::is_trivially_copyable_v<Quaternion> == true, "Quaternion has to be trivially copyable");
static_assert(std::is_polymorphic_v<Quaternion> == false, "Quaternion should not be polymorphic");
static_assert(ImplicitLifetimeC<Quaternion>, "Quaternions should have implicit lifetime");

// Left Scalar operators
template<class T>
MR_PF_DECL Quat<T> operator*(T, const Quat<T>&) noexcept;

// Static Utility
namespace TransformGen
{
    template<std::floating_point T>
    MR_PF_DECL Quat<T> RotationBetween(const Vector<3, T>& a,
                                       const Vector<3, T>& b) noexcept;

    template<std::floating_point T>
    MR_PF_DECL Quat<T> RotationBetweenZAxis(const Vector<3, T>& b) noexcept;

    template<std::floating_point T>
    MR_PF_DECL Quat<T> ToSpaceQuat(const Vector<3, T>& x,
                                   const Vector<3, T>& y,
                                   const Vector<3, T>& z) noexcept;
    template<std::floating_point T>
    MR_PF_DECL Quat<T> ToInvSpaceQuat(const Vector<3, T>& x,
                                      const Vector<3, T>& y,
                                      const Vector<3, T>& z) noexcept;
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