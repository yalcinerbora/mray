#pragma once

#include <concepts>
#include <utility>
#include "Definitions.h"
#include "Types.h"
#include "Span.h"

#ifdef MRAY_HETEROGENEOUS
    #include <cuda/half.h>
#endif

// Floating point type extension for half precision if available
#ifdef MRAY_HETEROGENEOUS

    template<typename T>
    concept FloatC = std::floating_point<T> || std::same_as<T, half>;

#else

    template<typename T>
    concept FloatC = std::floating_point<T>;

#endif

// We do not need this but, for consistency we define it
template<typename T>
concept IntegralC = std::integral<T>;

template<typename T>
concept SignedIntegralC = std::signed_integral<T>;


template<typename T>
concept ArithmeticC = std::integral<T> || FloatC<T>;

template<typename T>
concept SignedC = std::signed_integral<T> || FloatC<T>;

template<unsigned int N, ArithmeticC T>
class Vector;

template<unsigned int N, ArithmeticC T>
class Matrix;

template<FloatC T>
class Quat;

template<FloatC T>
class RayT;

template<unsigned int N, FloatC T>
class AABB;

// Vector Alias
// Typeless vectors are defaulted to float
using Vector2   = Vector<2, Float>;
using Vector3   = Vector<3, Float>;
using Vector4   = Vector<4, Float>;
// Float Type
using Vector2f  = Vector<2, float>;
using Vector3f  = Vector<3, float>;
using Vector4f  = Vector<4, float>;
// Double Type
using Vector2d  = Vector<2, double>;
using Vector3d  = Vector<3, double>;
using Vector4d  = Vector<4, double>;
// Integer Type
using Vector2i  = Vector<2, int32_t>;
using Vector3i  = Vector<3, int32_t>;
using Vector4i  = Vector<4, int32_t>;
// Unsigned Integer Type
using Vector2ui = Vector<2, uint32_t>;
using Vector3ui = Vector<3, uint32_t>;
using Vector4ui = Vector<4, uint32_t>;
// Long Types
using Vector2l  = Vector<2, int64_t>;
using Vector3l  = Vector<3, int64_t>;
using Vector4l  = Vector<4, int64_t>;

using Vector2ul = Vector<2, uint64_t>;
using Vector3ul = Vector<3, uint64_t>;
using Vector4ul = Vector<4, uint64_t>;
// Short
using Vector2s  = Vector<2, int16_t>;
using Vector2us = Vector<2, uint16_t>;
using Vector3s  = Vector<3, int16_t>;
using Vector3us = Vector<3, uint16_t>;
using Vector4s  = Vector<4, int16_t>;
using Vector4us = Vector<4, uint16_t>;
// Byte
using Vector2c  = Vector<2, int8_t>;
using Vector2uc = Vector<2, uint8_t>;
using Vector3c  = Vector<3, int8_t>;
using Vector3uc = Vector<3, uint8_t>;
using Vector4c  = Vector<4, int8_t>;
using Vector4uc = Vector<4, uint8_t>;

// Matrix Alias
// Typeless matrices are defaulted to float
using Matrix2x2 = Matrix<2, Float>;
using Matrix3x3 = Matrix<3, Float>;
using Matrix4x4 = Matrix<4, Float>;
// Float Type
using Matrix2x2f = Matrix<2, float>;
using Matrix3x3f = Matrix<3, float>;
using Matrix4x4f = Matrix<4, float>;
// Double Type
using Matrix2x2d = Matrix<2, double>;
using Matrix3x3d = Matrix<3, double>;
using Matrix4x4d = Matrix<4, double>;
// Integer Type
using Matrix2x2i = Matrix<2, int>;
using Matrix3x3i = Matrix<3, int>;
using Matrix4x4i = Matrix<4, int>;
// Unsigned Integer Type
using Matrix2x2ui = Matrix<2, unsigned int>;
using Matrix3x3ui = Matrix<3, unsigned int>;
using Matrix4x4ui = Matrix<4, unsigned int>;

// Quaternion Alias
using Quaternion = Quat<Float>;
using QuaternionF = Quat<float>;
using QuaternionD = Quat<double>;

// Ray Alias
using Ray = RayT<Float>;
using RayF = RayT<float>;
using RayD = RayT<double>;

// Typeless AABBs are defaulted to float
using AABB2 = AABB<2, Float>;
using AABB3 = AABB<3, Float>;
using AABB4 = AABB<4, Float>;
// Float Type
using AABB2f = AABB<2, float>;
using AABB3f = AABB<3, float>;
using AABB4f = AABB<4, float>;
// Double Type
using AABB2d = AABB<2, double>;
using AABB3d = AABB<3, double>;
using AABB4d = AABB<4, double>;

// Half types if available
#ifdef MRAY_HETEROGENEOUS
    using Vector2h = Vector<2, half>;
    using Vector3h = Vector<3, half>;
    using Vector4h = Vector<4, half>;

    using Matrix2x2h = Matrix2x2<2, half>;
    using Matrix3x3h = Matrix3x3<3, half>;
    using Matrix4x4h = Matrix4x4<4, half>;

    using QuaternionH = Quat<half>;

    using RayH = RayT<half>;

    using AABB2h = AABB<2, half>;
    using AABB3h = AABB<3, half>;
    using AABB4h = AABB<4, half>;
#endif

template<FloatC T>
using IntegralSister = std::conditional_t<std::is_same_v<T, float>, int32_t, int64_t>;

template<class T>
concept ArrayLikeC = requires(T t, Span<const typename T::InnerType, T::Dims> span)
{
    typename T::InnerType;
    T::Dims;
    T(span);
    { t.AsArray() } -> std::same_as<std::array<typename T::InnerType, T::Dims>&>;
    { std::as_const(t).AsArray() } -> std::same_as<const std::array<typename T::InnerType, T::Dims>&>;
};

// Vector Concept
// https://stackoverflow.com/a/54182690
template<class T>
concept VectorC = requires(T x)
{
    // Abuse CTAD
    //
    // Important! If you want to use this on your class, this is not a generic
    // implementation. We abuse CTAD, and if this class had deduction guide(s),
    // this would fail to filter some of the constructors i.e.:
    //
    //  template<class T>
    //  Vector(T) -> Vector<3, T>;
    //
    // (Weird example for the vector but you get the idea)
    //
    { Vector{x} } -> std::same_as<T>;
};
static_assert(!VectorC<Float>, "This should not work");

template<class T>
concept FloatVectorC = (VectorC<T> && std::floating_point<typename T::InnerType>);

template<class T>
concept IntegralVectorC = (VectorC<T> && !FloatVectorC<T>);

template<class T>
concept FloatVectorOrFloatC = (FloatVectorC<T> || std::floating_point<T>);

// Vector, AABB print helpers
template <ArrayLikeC V>
auto format_as(const V& v) { return v.AsArray(); }

template <unsigned int N, FloatC T>
auto format_as(const AABB<N, T>& v)
{
    std::array<std::array<T, N>, 2> result;
    result[0] = v.Min().AsArray();
    result[1] = v.Max().AsArray();
    return result;
}