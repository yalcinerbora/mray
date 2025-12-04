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
    constexpr             Matrix() = default;
    template<std::convertible_to<T> C>
    MR_PF_DECL_V explicit Matrix(C) noexcept;
    template<std::convertible_to<T> C>
    MR_PF_DECL_V explicit Matrix(Span<const C, N*N> data) noexcept;
    template <class... Args>
    MR_PF_DECL_V explicit Matrix(const Args... dataList) noexcept
    requires (std::convertible_to<Args, T> && ...) && (sizeof...(Args) == N*N);
    template <class... Rows>
    MR_PF_DECL_V explicit Matrix(const Rows&... rows) noexcept
    requires (std::is_same_v<Rows, Vector<N, T>> && ...) && (sizeof...(Rows) == N);
    template <unsigned int M>
    MR_PF_DECL_V explicit Matrix(const Matrix<M, T>&) noexcept requires (M > N);
    template<std::convertible_to<T> C>
    MR_PF_DECL_V explicit Matrix(const Matrix<N, C>&);

    // Accessors
    MR_PF_DECL T&        operator[](unsigned int) noexcept;
    MR_PF_DECL const T&  operator[](unsigned int) const noexcept;
    MR_PF_DECL T&        operator()(unsigned int row, unsigned int column) noexcept;
    MR_PF_DECL const T&  operator()(unsigned int row, unsigned int column) const noexcept;
    // Structured Binding Helper
    MR_PF_DECL const std::array<T, N*N>& AsArray() const noexcept;
    MR_PF_DECL std::array<T, N*N>&       AsArray() noexcept;

    // Modify
    MR_PF_DECL_V void   operator+=(const Matrix&) noexcept;
    MR_PF_DECL_V void   operator-=(const Matrix&) noexcept;
    MR_PF_DECL_V void   operator*=(const Matrix&) noexcept;
    MR_PF_DECL_V void   operator*=(T) noexcept;
    MR_PF_DECL_V void   operator/=(const Matrix&) noexcept;
    MR_PF_DECL_V void   operator/=(T) noexcept;
    MR_PF_DECL Matrix   operator+(const Matrix&) const noexcept;
    MR_PF_DECL Matrix   operator-(const Matrix&) const noexcept;
    MR_PF_DECL Matrix   operator-() const noexcept requires SignedC<T>;
    MR_PF_DECL Matrix   operator/(const Matrix&) const noexcept;
    MR_PF_DECL Matrix   operator/(T) const noexcept;
    MR_PF_DECL Matrix   operator*(const Matrix&) const noexcept;
    template<unsigned int M>
    MR_PF_DECL Vector<M, T> operator*(const Vector<M, T>&) const noexcept requires (M == N) || ((M + 1) == N);
    MR_PF_DECL Matrix       operator*(T) const noexcept;

    // Logic
    MR_PF_DECL bool operator==(const Matrix&) const noexcept;
    MR_PF_DECL bool operator!=(const Matrix&) const noexcept;

    // Utility
    MR_PF_DECL T         Determinant() const noexcept requires (N == 2);
    MR_PF_DECL T         Determinant() const noexcept requires (N == 3);
    MR_PF_DECL T         Determinant() const noexcept requires (N == 4);
    MR_PF_DECL Matrix    Inverse() const noexcept requires FloatC<T> && (N == 2);
    MR_PF_DECL Matrix    Inverse() const noexcept requires FloatC<T> && (N == 3);
    MR_PF_DECL Matrix    Inverse() const noexcept requires FloatC<T> && (N == 4);
    MR_PF_DECL Matrix    Transpose() const noexcept;

    MR_PF_DECL RayT<T>          TransformRay(const RayT<T>&) const noexcept requires (N == 3);
    MR_PF_DECL RayT<T>          TransformRay(const RayT<T>&) const noexcept requires (N == 4);
    template <unsigned int M>
    MR_PF_DECL AABB<M, T>       TransformAABB(const AABB<M, T>&) const noexcept requires((M+1) == N);
    template <unsigned int M>
    MR_PF_DECL Vector<M, T>     LeftMultiply(const Vector<M, T>&) const noexcept requires (M <= N);

    MR_PF_DECL static Matrix    Identity() noexcept;
    MR_PF_DECL static Matrix    Zero() noexcept;
};

template<ArithmeticC T>
class alignas(ChooseVectorAlignment(4 * sizeof(T))) Matrix3x4T
{
    public:
    using InnerType = T;
    static constexpr unsigned int Dims = 12;

    private:
    std::array<T, 12>   matrix;

    public:
    // Constructors & Destructor
    constexpr             Matrix3x4T() = default;
    template<std::convertible_to<T> C>
    MR_PF_DECL_V explicit Matrix3x4T(C) noexcept;
    template<std::convertible_to<T> C>
    MR_PF_DECL_V explicit Matrix3x4T(Span<const C, 4 * 3> data) noexcept;
    template <class... Args>
    MR_PF_DECL_V explicit Matrix3x4T(const Args... dataList) noexcept
        requires (std::convertible_to<Args, T> && ...) && (sizeof...(Args) == 12);
    MR_PF_DECL_V explicit Matrix3x4T(const Vector<4, T>& row0,
                                     const Vector<4, T>& row1,
                                     const Vector<4, T>& row2) noexcept;
    MR_PF_DECL_V explicit Matrix3x4T(const Matrix<4, T>&) noexcept;
    template<std::convertible_to<T> C>
    MR_PF_DECL_V explicit Matrix3x4T(const Matrix3x4T<C>&);

    // Accessors
    MR_PF_DECL T&       operator[](unsigned int) noexcept;
    MR_PF_DECL const T& operator[](unsigned int) const noexcept;
    MR_PF_DECL T&       operator()(unsigned int row, unsigned int column) noexcept;
    MR_PF_DECL const T& operator()(unsigned int row, unsigned int column) const noexcept;
    // Structured Binding Helper
    MR_PF_DECL const std::array<T, 12>& AsArray() const noexcept;
    MR_PF_DECL std::array<T, 12>&       AsArray() noexcept;
    // Modify
    MR_PF_DECL_V void       operator+=(const Matrix3x4T&) noexcept;
    MR_PF_DECL_V void       operator*=(const Matrix3x4T&) noexcept;
    MR_PF_DECL_V void       operator*=(T) noexcept;
    MR_PF_DECL Matrix3x4T   operator+(const Matrix3x4T&) const noexcept;
    MR_PF_DECL Matrix3x4T   operator*(const Matrix3x4T&) const noexcept;
    MR_PF_DECL Vector<3, T> operator*(const Vector<3, T>&) const noexcept;
    MR_PF_DECL Vector<4, T> operator*(const Vector<4, T>&) const noexcept;
    MR_PF_DECL Matrix3x4T   operator*(T) const noexcept;

    // Logic
    MR_PF_DECL bool operator==(const Matrix3x4T&) const noexcept;
    MR_PF_DECL bool operator!=(const Matrix3x4T&) const noexcept;

    // Utility
    MR_PF_DECL T                 Determinant() const noexcept requires FloatC<T>;
    MR_PF_DECL Matrix3x4T        Inverse() const noexcept requires FloatC<T>;
    MR_PF_DECL RayT<T>           TransformRay(const RayT<T>&) const noexcept;
    MR_PF_DECL AABB<3, T>        TransformAABB(const AABB<3, T>&) const noexcept;
    MR_PF_DECL Vector<3, T>      LeftMultiply(const Vector<3, T>&) const noexcept;
    MR_PF_DECL static Matrix3x4T Identity() noexcept;
    MR_PF_DECL static Matrix3x4T Zero() noexcept;
};


// Left Scalar operators
template<unsigned int N, ArithmeticC T>
MR_PF_DECL Matrix<N, T> operator*(T, const Matrix<N, T>&) noexcept;

template<ArithmeticC T>
MR_PF_DECL Matrix3x4T<T> operator*(T, const Matrix3x4T<T>&) noexcept;

// Spacial Matrix4x4 -> Matrix3x3
template<ArithmeticC T>
MR_PF_DECL Matrix<4, T> ToMatrix4x4(const Matrix<3, T>&) noexcept;

// Sanity Checks for Matrices
static_assert(std::is_trivially_default_constructible_v<Matrix4x4> == true, "Matrices has to be trivially destructible");
static_assert(std::is_trivially_destructible_v<Matrix4x4> == true, "Matrices has to be trivially destructible");
static_assert(std::is_trivially_copyable_v<Matrix4x4> == true, "Matrices has to be trivially copyable");
static_assert(std::is_polymorphic_v<Matrix4x4> == false, "Matrices should not be polymorphic");
static_assert(ImplicitLifetimeC<Matrix4x4>, "Matrices should have implicit lifetime");

// Sanity Checks for Matrices
static_assert(std::is_trivially_default_constructible_v<Matrix3x4> == true, "Matrices has to be trivially destructible");
static_assert(std::is_trivially_destructible_v<Matrix3x4> == true, "Matrices has to be trivially destructible");
static_assert(std::is_trivially_copyable_v<Matrix3x4> == true, "Matrices has to be trivially copyable");
static_assert(std::is_polymorphic_v<Matrix3x4> == false, "Matrices should not be polymorphic");
static_assert(ImplicitLifetimeC<Matrix3x4>, "Matrices should have implicit lifetime");

// Transformation Matrix Generation
namespace TransformGen
{
    // Extraction Functions
    template<FloatC T>
    MR_PF_DEF Vector<3, T>  ExtractScale(const Matrix<4, T>&) noexcept;
    template<FloatC T>
    MR_PF_DEF Vector<3, T>  ExtractScale(const Matrix3x4T<T>&) noexcept;
    template<FloatC T>
    MR_PF_DEF Vector<3, T>  ExtractTranslation(const Matrix<4, T>&) noexcept;
    template<FloatC T>
    MR_PF_DEF Matrix<4, T>  Translate(const Vector<3, T>&) noexcept;
    template<FloatC T>
    MR_PF_DEF Matrix<4, T>  Scale(T) noexcept;
    template<FloatC T>
    MR_PF_DEF Matrix <4, T> Scale(T x, T y, T z) noexcept;
    template<FloatC T>
    MR_PF_DEF Matrix<4, T>  Rotate(T angle, const Vector<3, T>&) noexcept;
    template<FloatC T>
    MR_PF_DEF Matrix<4, T>  Rotate(const Quat<T>&) noexcept;
    template<FloatC T>
    MR_PF_DEF Matrix<4, T>  Perspective(T fovXRadians, T aspectRatio,
                                        T nearPlane, T farPlane) noexcept;
    template<FloatC T>
    MR_PF_DEF Matrix<4, T>  Orthogonal(T left, T right,
                                       T top, T bottom,
                                       T nearPlane, T farPlane) noexcept;
    template<FloatC T>
    MR_PF_DEF Matrix<4, T>  Orthogonal(T width, T height,
                                       T nearPlane, T farPlane) noexcept;

    template<FloatC T>
    MR_PF_DEF Matrix<4, T>  LookAt(const Vector<3, T>& eyePos,
                                   const Vector<3, T>& at,
                                   const Vector<3, T>& up) noexcept;
    template<FloatC T>
    MR_PF_DEF Matrix<3, T>  ToSpaceMat(const Vector<3, T>& x,
                                       const Vector<3, T>& y,
                                       const Vector<3, T>& z) noexcept;

    template<FloatC T>
    MR_PF_DEF Matrix<3, T>  ToInvSpaceMat(const Vector<3, T>& x,
                                          const Vector<3, T>& y,
                                          const Vector<3, T>& z) noexcept;

    template<FloatC T>
    MR_PF_DEF Vector<3, T> YUpToZUp(const Vector<3, T>& vec) noexcept;

    template<FloatC T>
    MR_PF_DEF Vector<3, T> ZUpToYUp(const Vector<3, T>& vec) noexcept;

    template<FloatC T>
    MR_PF_DEF Matrix<4, T> YUpToZUpMat() noexcept;

    template<FloatC T>
    MR_PF_DEF Matrix<4, T> ZUpToYUpMat() noexcept;
}

// Implementation
#include "Matrix.hpp"   // CPU & GPU

// Matrix Traits
template<class T>
concept MatrixC = std::is_same_v<T, Matrix2x2f> ||
                  std::is_same_v<T, Matrix2x2d> ||
                  std::is_same_v<T, Matrix3x3f> ||
                  std::is_same_v<T, Matrix3x3d> ||
                  std::is_same_v<T, Matrix4x4f> ||
                  std::is_same_v<T, Matrix4x4d>;

// TODO: Add more?
static_assert(ArrayLikeC<Matrix2x2>, "Matrix2x2 is not ArrayLike!");
static_assert(ArrayLikeC<Matrix3x3>, "Matrix3x3 is not ArrayLike!");
static_assert(ArrayLikeC<Matrix4x4>, "Matrix4x4 is not ArrayLike!");