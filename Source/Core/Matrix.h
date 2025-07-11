#pragma once

#include "Vector.h"
#include "Quaternion.h"

// Matrices are row-major
// Matrix vector multiplications (m * v) only
// Assumes vector is column vector
template<unsigned int N, ArithmeticC T>
class alignas(ChooseVectorAlignment(N * sizeof(T))) Matrix
{
    static_assert(N == 2 || N == 3 || N == 4, "Matrix size should be 2x2, 3x3 or 4x4");

    public:
    using InnerType                      = T;
    static constexpr unsigned int Dims   = N * N;

    private:
    std::array<T, N*N>              matrix;

    public:
    // Constructors & Destructor
    constexpr                       Matrix() = default;
    template<std::convertible_to<T> C>
    MRAY_HYBRID constexpr explicit  Matrix(C);
    template<std::convertible_to<T> C>
    MRAY_HYBRID constexpr explicit  Matrix(Span<const C, N*N> data);
    template <class... Args>
    MRAY_HYBRID constexpr explicit  Matrix(const Args... dataList) requires (std::convertible_to<Args, T> && ...) &&
                                                                            (sizeof...(Args) == N*N);
    template <class... Rows>
    MRAY_HYBRID constexpr explicit  Matrix(const Rows&... rows) requires (std::is_same_v<Rows, Vector<N, T>> && ...) &&
                                                                         (sizeof...(Rows) == N);
    template <unsigned int M>
    MRAY_HYBRID constexpr explicit  Matrix(const Matrix<M, T>&) requires (M > N);

    // Accessors
    MRAY_HYBRID constexpr T&            operator[](unsigned int);
    MRAY_HYBRID constexpr const T&      operator[](unsigned int) const;
    MRAY_HYBRID constexpr T&            operator()(unsigned int row, unsigned int column);
    MRAY_HYBRID constexpr const T&      operator()(unsigned int row, unsigned int column) const;
    // Structured Binding Helper
    MRAY_HYBRID
    constexpr const std::array<T, N*N>& AsArray() const;
    MRAY_HYBRID
    constexpr std::array<T, N*N>&       AsArray();

    // Modify
    MRAY_HYBRID constexpr void      operator+=(const Matrix&);
    MRAY_HYBRID constexpr void      operator-=(const Matrix&);
    MRAY_HYBRID constexpr void      operator*=(const Matrix&);
    MRAY_HYBRID constexpr void      operator*=(T);
    MRAY_HYBRID constexpr void      operator/=(const Matrix&);
    MRAY_HYBRID constexpr void      operator/=(T);

    MRAY_HYBRID constexpr Matrix        operator+(const Matrix&) const;
    MRAY_HYBRID constexpr Matrix        operator-(const Matrix&) const;
    MRAY_HYBRID constexpr Matrix        operator-() const requires SignedC<T>;
    MRAY_HYBRID constexpr Matrix        operator/(const Matrix&) const;
    MRAY_HYBRID constexpr Matrix        operator/(T) const;
    MRAY_HYBRID constexpr Matrix        operator*(const Matrix&) const;
    template<unsigned int M>
    MRAY_HYBRID constexpr Vector<M, T>  operator*(const Vector<M, T>&) const requires (M == N) || ((M + 1) == N);
    MRAY_HYBRID constexpr Matrix        operator*(T) const;

    // Logic
    MRAY_HYBRID constexpr bool         operator==(const Matrix&) const;
    MRAY_HYBRID constexpr bool         operator!=(const Matrix&) const;

    // Utility
    MRAY_HYBRID NO_DISCARD constexpr T      Determinant() const requires (N == 2);
    MRAY_HYBRID NO_DISCARD constexpr T      Determinant() const requires (N == 3);
    MRAY_HYBRID NO_DISCARD constexpr T      Determinant() const requires (N == 4);
    MRAY_HYBRID NO_DISCARD constexpr Matrix Inverse() const requires std::floating_point<T> && (N == 2);
    MRAY_HYBRID NO_DISCARD constexpr Matrix Inverse() const requires std::floating_point<T> && (N == 3);
    MRAY_HYBRID NO_DISCARD constexpr Matrix Inverse() const requires std::floating_point<T> && (N == 4);
    MRAY_HYBRID constexpr Matrix&           InverseSelf() requires std::floating_point<T>;
    MRAY_HYBRID NO_DISCARD constexpr Matrix Transpose() const;
    MRAY_HYBRID constexpr Matrix&           TransposeSelf();

    MRAY_HYBRID NO_DISCARD constexpr Matrix Clamp(const Matrix&, const Matrix&) const;
    MRAY_HYBRID NO_DISCARD constexpr Matrix Clamp(T min, T max) const;
    MRAY_HYBRID constexpr Matrix&           ClampSelf(const Matrix&, const Matrix&);
    MRAY_HYBRID constexpr Matrix&           ClampSelf(T min, T max);

    MRAY_HYBRID NO_DISCARD constexpr Matrix Abs() const requires SignedC<T>;
    MRAY_HYBRID constexpr Matrix&           AbsSelf() requires SignedC<T>;

    MRAY_HYBRID NO_DISCARD constexpr Matrix Round() const requires std::floating_point<T>;
    MRAY_HYBRID constexpr Matrix&           RoundSelf() requires std::floating_point<T>;
    MRAY_HYBRID NO_DISCARD constexpr Matrix Floor() const requires std::floating_point<T>;
    MRAY_HYBRID constexpr Matrix&           FloorSelf() requires std::floating_point<T>;
    MRAY_HYBRID NO_DISCARD constexpr Matrix Ceil() const requires std::floating_point<T>;
    MRAY_HYBRID constexpr Matrix&           CeilSelf() requires std::floating_point<T>;

    MRAY_HYBRID constexpr RayT<T>           TransformRay(const RayT<T>&) const requires (N == 3);
    MRAY_HYBRID constexpr RayT<T>           TransformRay(const RayT<T>&) const requires (N == 4);
    template <unsigned int M>
    MRAY_HYBRID constexpr AABB<M, T>        TransformAABB(const AABB<M, T>&) const requires((M+1) == N);
    template <unsigned int M>
    MRAY_HYBRID constexpr Vector<M, T>      LeftMultiply(const Vector<M, T>&) const requires (M <= N);

    static MRAY_HYBRID constexpr Matrix     Lerp(const Matrix&, const Matrix&, T) requires std::floating_point<T>;
    static MRAY_HYBRID constexpr Matrix     Min(const Matrix&, const Matrix&);
    static MRAY_HYBRID constexpr Matrix     Min(const Matrix&, T);
    static MRAY_HYBRID constexpr Matrix     Max(const Matrix&, const Matrix&);
    static MRAY_HYBRID constexpr Matrix     Max(const Matrix&, T);

    static MRAY_HYBRID constexpr Matrix     Identity();
    static MRAY_HYBRID constexpr Matrix     Zero();
};

// Left Scalar operators
template<unsigned int N, ArithmeticC T>
MRAY_HYBRID constexpr  Matrix<N, T> operator*(T, const Matrix<N, T>&);

// Spacial Matrix4x4 -> Matrix3x3
template<ArithmeticC T>
MRAY_HYBRID constexpr Matrix<4, T> ToMatrix4x4(const Matrix<3, T>&);

// Sanity Checks for Matrices
static_assert(std::is_trivially_default_constructible_v<Matrix4x4> == true, "Matrices has to be trivially destructible");
static_assert(std::is_trivially_destructible_v<Matrix4x4> == true, "Matrices has to be trivially destructible");
static_assert(std::is_trivially_copyable_v<Matrix4x4> == true, "Matrices has to be trivially copyable");
static_assert(std::is_polymorphic_v<Matrix4x4> == false, "Matrices should not be polymorphic");
static_assert(ImplicitLifetimeC<Matrix4x4>, "Matrices should have implicit lifetime");

// Transformation Matrix Generation
namespace TransformGen
{
    // Extraction Functions
    template<std::floating_point T>
    MRAY_HYBRID constexpr Vector<3, T>  ExtractScale(const Matrix<4, T>&);
    template<std::floating_point T>
    MRAY_HYBRID constexpr Vector<3, T>  ExtractTranslation(const Matrix<4, T>&);
    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<4, T>  Translate(const Vector<3, T>&);
    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<4, T>  Scale(T);
    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix <4, T> Scale(T x, T y, T z);
    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<4, T>  Rotate(T angle, const Vector<3, T>&);
    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<4, T>  Rotate(const Quat<T>&);
    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<4, T>  Perspective(T fovXRadians, T aspectRatio,
                                                    T nearPlane, T farPlane);
    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<4, T>  Orthogonal(T left, T right,
                                                   T top, T bottom,
                                                   T nearPlane, T farPlane);
    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<4, T>  Orthogonal(T width, T height,
                                                   T nearPlane, T farPlane);

    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<4, T>  LookAt(const Vector<3, T>& eyePos,
                                               const Vector<3, T>& at,
                                               const Vector<3, T>& up);
    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<3, T>  ToSpaceMat(const Vector<3, T>& x,
                                                   const Vector<3, T>& y,
                                                   const Vector<3, T>& z);

    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<3, T>  ToInvSpaceMat(const Vector<3, T>& x,
                                                      const Vector<3, T>& y,
                                                      const Vector<3, T>& z);

    template<std::floating_point T>
    MRAY_HYBRID constexpr Vector<3, T> YUpToZUp(const Vector<3, T>& vec);

    template<std::floating_point T>
    MRAY_HYBRID constexpr Vector<3, T> ZUpToYUp(const Vector<3, T>& vec);

    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<4, T> YUpToZUpMat();

    template<std::floating_point T>
    MRAY_HYBRID constexpr Matrix<4, T> ZUpToYUpMat();
}

// Implementation
#include "Matrix.hpp"   // CPU & GPU

// Matrix Traits
template<class T>
concept MatrixC = requires()
{
    std::is_same_v<T, Matrix2x2f>   ||
    std::is_same_v<T, Matrix2x2d>   ||
    std::is_same_v<T, Matrix2x2i>   ||
    std::is_same_v<T, Matrix2x2ui>  ||
    std::is_same_v<T, Matrix3x3f>   ||
    std::is_same_v<T, Matrix3x3d>   ||
    std::is_same_v<T, Matrix3x3i>   ||
    std::is_same_v<T, Matrix3x3ui>  ||
    std::is_same_v<T, Matrix4x4f>   ||
    std::is_same_v<T, Matrix4x4d>   ||
    std::is_same_v<T, Matrix4x4i>   ||
    std::is_same_v<T, Matrix4x4ui>;
};

// TODO: Add more?
static_assert(ArrayLikeC<Matrix2x2>, "Matrix2x2 is not ArrayLike!");
static_assert(ArrayLikeC<Matrix3x3>, "Matrix3x3 is not ArrayLike!");
static_assert(ArrayLikeC<Matrix4x4>, "Matrix4x4 is not ArrayLike!");