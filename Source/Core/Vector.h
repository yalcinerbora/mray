#pragma once
/**

Arbitrary sized vector. Vector is column vector (N x 1 matrix)
which means that it can only be multiplied with matrices from right.

N should be 2, 3 or 4 at most.

*/

#include <type_traits>
#include <concepts>
#include <array>

#include "MathForward.h"
#include "NormTypes.h"
#include "Types.h"

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
    constexpr           Vector() = default;
    template<std::convertible_to<T> C>
    MR_PF_DECL_V explicit Vector(C);
    template<std::convertible_to<T> C>
    MR_PF_DECL_V explicit Vector(Span<const C , N> data);
    template <std::convertible_to<T>... Args>
    MR_PF_DECL_V explicit Vector(const Args... dataList);
    template <class... Args>
    MR_PF_DECL_V explicit Vector(const Vector<N - sizeof...(Args), T>&,
                               const Args... dataList) requires (N - sizeof...(Args) > 1);
    template<std::convertible_to<T> C>
    MR_PF_DECL_V explicit Vector(std::array<C, N>&& data);
    template<std::convertible_to<T> C>
    MR_PF_DECL_V explicit Vector(const Vector<N, C>&);
    template <unsigned int M>
    MR_PF_DECL_V explicit Vector(const Vector<M, T>&) requires (M > N);
    // NormTypes Related Constructors
    template <std::unsigned_integral IT>
    MR_PF_DECL_V explicit Vector(const UNorm<N, IT>&) requires (std::floating_point<T>);
    template <std::signed_integral IT>
    MR_PF_DECL_V explicit Vector(const SNorm<N, IT>&) requires (std::floating_point<T>);

    // Accessors
    MR_PF_DECL T&       operator[](unsigned int);
    MR_PF_DECL const T& operator[](unsigned int) const;
    // Structured Binding Helper
    MR_PF_DECL  const std::array<T, N>&   AsArray() const;
    MR_PF_DECL  std::array<T, N>&         AsArray();

    // Type cast
    template<unsigned int M, class C>
    MR_PF_DECL explicit operator Vector<M, C>() const requires (M <= N) && std::convertible_to<C, T>;

    // Modify
    MR_PF_DECL_V void   operator+=(const Vector&);
    MR_PF_DECL_V void   operator-=(const Vector&);
    MR_PF_DECL_V void   operator*=(const Vector&);
    MR_PF_DECL_V void   operator*=(T);
    MR_PF_DECL_V void   operator/=(const Vector&);
    MR_PF_DECL_V void   operator/=(T);

    MR_PF_DECL Vector   operator+(const Vector&) const;
    MR_PF_DECL Vector   operator+(T) const;
    MR_PF_DECL Vector   operator-(const Vector&) const;
    MR_PF_DECL Vector   operator-(T) const;
    MR_PF_DECL Vector   operator-() const               requires SignedC<T>;
    MR_PF_DECL Vector   operator*(const Vector&) const;
    MR_PF_DECL Vector   operator*(T) const;
    MR_PF_DECL Vector   operator/(const Vector&) const;
    MR_PF_DECL Vector   operator/(T) const;

    MR_PF_DECL Vector   operator%(const Vector&) const  requires FloatC<T>;
    MR_PF_DECL Vector   operator%(T) const              requires FloatC<T>;
    MR_PF_DECL Vector   operator%(const Vector&) const  requires IntegralC<T>;
    MR_PF_DECL Vector   operator%(T) const              requires IntegralC<T>;

    // Logic
    MR_PF_DECL bool     operator==(const Vector&) const;
    MR_PF_DECL bool     operator!=(const Vector&) const;
    MR_PF_DECL bool     operator<(const Vector&) const;
    MR_PF_DECL bool     operator<=(const Vector&) const;
    MR_PF_DECL bool     operator>(const Vector&) const;
    MR_PF_DECL bool     operator>=(const Vector&) const;
    // Reduction
    MR_PF_DECL T        Sum() const;
    MR_PF_DECL T        Multiply() const;
    // Max Min Reduction functions are selections instead
    // since it sometimes useful to fetch the which index
    // (axis) is maximum/minimum so that you can do other stuff with it.
    MR_PF_DECL unsigned int  Maximum() const;
    MR_PF_DECL unsigned int  Minimum() const;
    // Literals
    MR_PF_DECL static Vector Zero();
    MR_PF_DECL static Vector XAxis() requires (N == 3);
    MR_PF_DECL static Vector YAxis() requires (N == 3);
    MR_PF_DECL static Vector ZAxis() requires (N == 3);
};

// Left scalar multiplication
template<unsigned int N, ArithmeticC T>
MR_PF_DECL Vector<N, T> operator*(T, const Vector<N, T>&);

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

// TODO: Add more?
static_assert(ArrayLikeC<Vector2>, "Vec2 is not ArrayLike!");
static_assert(ArrayLikeC<Vector3>, "Vec3 is not ArrayLike!");
static_assert(ArrayLikeC<Vector4>, "Vec4 is not ArrayLike!");

// Implementation
#include "Vector.hpp"
