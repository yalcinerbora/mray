#pragma once

#pragma once

/**

Arbitrary sized vector. Vector is column vector (N x 1 matrix)
which means that it can only be multiplied with matrices from right.

N should be 2, 3 or 4 at most.

*/

#include <cmath>
#include <type_traits>
#include <tuple>
#include <concepts>
#include <array>

#include "MathForward.h"
#include "MathFunctions.h"
#include "NormTypes.h"
#include "Types.h"
#include "BitFunctions.h"

static consteval unsigned int ChooseVectorAlignment(unsigned int totalSize)
{
    if(totalSize <= 2)
        return 2;
    else if(totalSize <= 4)
        return 4;           // 1byte Vector Types
    else if(totalSize <= 8)
        return 8;           // 4byte Vector2 Types
    else if(totalSize < 16)
        return 4;           // 4byte Vector3 Types
    else
        return 16;          // 4byte Vector4 Types
}

template<unsigned int N, ArithmeticC T>
class alignas(ChooseVectorAlignment(N * sizeof(T))) Vector
{
    static_assert(N == 2 || N == 3 || N == 4, "Vector size should be 2, 3 or 4");

    public:
    using InnerType                     = T;
    static constexpr unsigned int Dims  = N;

    private:
    std::array<T, N>                vector;

    public:
    // Constructors & Destructor
    constexpr                       Vector() = default;
    template<std::convertible_to<T> C>
    MRAY_HYBRID constexpr explicit  Vector(C);
    template<std::convertible_to<T> C>
    MRAY_HYBRID constexpr explicit  Vector(Span<const C , N> data);
    template <std::convertible_to<T>... Args>
    MRAY_HYBRID constexpr explicit  Vector(const Args... dataList); //requires (, T> && ...) && (sizeof...(Args) == N);
    template <class... Args>
    MRAY_HYBRID constexpr           Vector(const Vector<N - sizeof...(Args), T>&,
                                           const Args... dataList) requires (N - sizeof...(Args) > 1);
    template<std::convertible_to<T> C>
    MRAY_HYBRID constexpr explicit  Vector(std::array<C, N>&& data);
    template<std::convertible_to<T> C>
    MRAY_HYBRID constexpr explicit  Vector(const Vector<N, C>&);
    template <unsigned int M>
    MRAY_HYBRID constexpr explicit  Vector(const Vector<M, T>&) requires (M > N);
    // NormTypes Related Constructors
    template <std::unsigned_integral IT>
    MRAY_HYBRID constexpr explicit  Vector(const UNorm<N, IT>&) requires (std::floating_point<T>);
    template <std::signed_integral IT>
    MRAY_HYBRID constexpr explicit  Vector(const SNorm<N, IT>&) requires (std::floating_point<T>);

    // Accessors
    MRAY_HYBRID constexpr T&            operator[](unsigned int);
    MRAY_HYBRID constexpr const T&      operator[](unsigned int) const;
    // Structured Binding Helper
    MRAY_HYBRID
    constexpr const std::array<T, N>&   AsArray() const;
    MRAY_HYBRID
    constexpr std::array<T, N>&         AsArray();

    // Type cast
    template<unsigned int M, class C>
    MRAY_HYBRID explicit            operator Vector<M, C>() const requires (M <= N) && std::convertible_to<C, T>;

    // Modify
    MRAY_HYBRID constexpr void      operator+=(const Vector&);
    MRAY_HYBRID constexpr void      operator-=(const Vector&);
    MRAY_HYBRID constexpr void      operator*=(const Vector&);
    MRAY_HYBRID constexpr void      operator*=(T);
    MRAY_HYBRID constexpr void      operator/=(const Vector&);
    MRAY_HYBRID constexpr void      operator/=(T);

    MRAY_HYBRID constexpr Vector    operator+(const Vector&) const;
    MRAY_HYBRID constexpr Vector    operator+(T) const;
    MRAY_HYBRID constexpr Vector    operator-(const Vector&) const;
    MRAY_HYBRID constexpr Vector    operator-(T) const;
    MRAY_HYBRID constexpr Vector    operator-() const requires SignedC<T>;
    MRAY_HYBRID constexpr Vector    operator*(const Vector&) const;
    MRAY_HYBRID constexpr Vector    operator*(T) const;
    MRAY_HYBRID constexpr Vector    operator/(const Vector&) const;
    MRAY_HYBRID constexpr Vector    operator/(T) const;

    MRAY_HYBRID constexpr Vector    operator%(const Vector&) const requires std::floating_point<T>;
    MRAY_HYBRID constexpr Vector    operator%(T) const requires std::floating_point<T>;
    MRAY_HYBRID constexpr Vector    operator%(const Vector&) const  requires std::integral<T>;
    MRAY_HYBRID constexpr Vector    operator%(T) const  requires std::integral<T>;

    // Logic
    MRAY_HYBRID constexpr bool      operator==(const Vector&) const;
    MRAY_HYBRID constexpr bool      operator!=(const Vector&) const;
    MRAY_HYBRID constexpr bool      operator<(const Vector&) const;
    MRAY_HYBRID constexpr bool      operator<=(const Vector&) const;
    MRAY_HYBRID constexpr bool      operator>(const Vector&) const;
    MRAY_HYBRID constexpr bool      operator>=(const Vector&) const;

    // Utility
    MRAY_HYBRID constexpr T         Dot(const Vector&) const;

    // Reduction
    MRAY_HYBRID constexpr T         Sum() const;
    MRAY_HYBRID constexpr T         Multiply() const;
    // Max Min Reduction functions are selections instead
    // since it sometimes useful to fetch the which index
    // (axis) is maximum/minimum so that you can do other stuff with it.
    MRAY_HYBRID constexpr unsigned int  Maximum() const;
    MRAY_HYBRID constexpr unsigned int  Minimum() const;

    MRAY_HYBRID constexpr T         Length() const requires std::floating_point<T>;
    MRAY_HYBRID constexpr T         LengthSqr() const;


    MRAY_HYBRID NO_DISCARD constexpr Vector Normalize() const requires std::floating_point<T>;
    MRAY_HYBRID constexpr Vector&           NormalizeSelf() requires std::floating_point<T>;

    MRAY_HYBRID constexpr Vector            Clamp(const Vector&, const Vector&) const;
    MRAY_HYBRID NO_DISCARD constexpr Vector Clamp(T min, T max) const;
    MRAY_HYBRID constexpr Vector&           ClampSelf(const Vector&, const Vector&);
    MRAY_HYBRID constexpr Vector&           ClampSelf(T min, T max);
    MRAY_HYBRID constexpr bool              HasNaN() const requires std::floating_point<T>;

    MRAY_HYBRID NO_DISCARD constexpr Vector Abs() const requires SignedC<T>;
    MRAY_HYBRID constexpr Vector&           AbsSelf() requires SignedC<T>;
    MRAY_HYBRID NO_DISCARD constexpr Vector Round() const requires std::floating_point<T>;
    MRAY_HYBRID constexpr Vector&           RoundSelf() requires std::floating_point<T>;
    MRAY_HYBRID NO_DISCARD constexpr Vector Floor() const requires std::floating_point<T>;
    MRAY_HYBRID constexpr Vector&           FloorSelf() requires std::floating_point<T>;
    MRAY_HYBRID NO_DISCARD constexpr Vector Ceil() const requires std::floating_point<T>;
    MRAY_HYBRID constexpr Vector&           CeilSelf() requires std::floating_point<T>;

    static MRAY_HYBRID constexpr Vector     Min(const Vector&, const Vector&);
    static MRAY_HYBRID constexpr Vector     Min(const Vector&, T);
    static MRAY_HYBRID constexpr Vector     Max(const Vector&, const Vector&);
    static MRAY_HYBRID constexpr Vector     Max(const Vector&, T);

    static MRAY_HYBRID constexpr Vector     Lerp(const Vector&, const Vector&, T) requires std::floating_point<T>;
    static MRAY_HYBRID constexpr Vector     Smoothstep(const Vector&, const Vector&, T) requires std::floating_point<T>;
    static MRAY_HYBRID constexpr Vector     Sqrt(const Vector&) requires std::floating_point<T>;

    // Literals
    static MRAY_HYBRID constexpr Vector     Zero();

    // Vector 3 Only
    static MRAY_HYBRID constexpr Vector     XAxis() requires (N == 3);
    static MRAY_HYBRID constexpr Vector     YAxis() requires (N == 3);
    static MRAY_HYBRID constexpr Vector     ZAxis() requires (N == 3);
    static MRAY_HYBRID constexpr Vector     Cross(const Vector&, const Vector&) requires std::floating_point<T> && (N == 3);
    static MRAY_HYBRID constexpr Vector     OrthogonalVector(const Vector&) requires std::floating_point<T> && (N == 3);
};

// Left scalar multiplication
template<unsigned int N, ArithmeticC T>
MRAY_HYBRID constexpr Vector<N, T> operator*(T, const Vector<N, T>&);

// Sanity Checks for Vectors
static_assert(std::is_trivially_default_constructible_v<Vector3> == true, "Vectors has to be trivially destructible");
static_assert(std::is_trivially_destructible_v<Vector3> == true, "Vectors has to be trivially destructible");
static_assert(std::is_trivially_copyable_v<Vector3> == true, "Vectors has to be trivially copyable");
static_assert(std::is_polymorphic_v<Vector3> == false, "Vectors should not be polymorphic");
static_assert(ImplicitLifetimeC<Vector3>, "Vectors should have implicit lifetime");

// Alignment Checks
static_assert(sizeof(Vector2) == 8, "Vector2 should be tightly packed");
static_assert(sizeof(Vector3) == 12, "Vector3 should be tightly packed");
static_assert(sizeof(Vector4) == 16, "Vector4 should be tightly packed");

// Implementation
#include "Vector.hpp"

// Vector Concept
// TODO: Check a better way to do this
template<class T>
concept VectorC = requires()
{
    std::is_same_v<T, Vector2>    ||
    std::is_same_v<T, Vector3>    ||
    std::is_same_v<T, Vector4>    ||
    std::is_same_v<T, Vector2f>   ||
    std::is_same_v<T, Vector3f>   ||
    std::is_same_v<T, Vector4f>   ||
    std::is_same_v<T, Vector2d>   ||
    std::is_same_v<T, Vector3d>   ||
    std::is_same_v<T, Vector4d>   ||
    std::is_same_v<T, Vector2i>   ||
    std::is_same_v<T, Vector3i>   ||
    std::is_same_v<T, Vector4i>   ||
    std::is_same_v<T, Vector2ui>  ||
    std::is_same_v<T, Vector3ui>  ||
    std::is_same_v<T, Vector4ui>  ||
    std::is_same_v<T, Vector2l>   ||
    std::is_same_v<T, Vector2ul>  ||
    std::is_same_v<T, Vector3ul>  ||
    std::is_same_v<T, Vector2s>   ||
    std::is_same_v<T, Vector2us>  ||
    std::is_same_v<T, Vector3s>   ||
    std::is_same_v<T, Vector3us>  ||
    std::is_same_v<T, Vector4s>   ||
    std::is_same_v<T, Vector4us>  ||
    std::is_same_v<T, Vector2c>   ||
    std::is_same_v<T, Vector2uc>  ||
    std::is_same_v<T, Vector3c>   ||
    std::is_same_v<T, Vector3uc>  ||
    std::is_same_v<T, Vector4c>   ||
    std::is_same_v<T, Vector4uc>;
};

// TODO: Add more?
static_assert(ArrayLikeC<Vector2>, "Vec2 is not ArrayLike!");
static_assert(ArrayLikeC<Vector3>, "Vec3 is not ArrayLike!");
static_assert(ArrayLikeC<Vector4>, "Vec4 is not ArrayLike!");
