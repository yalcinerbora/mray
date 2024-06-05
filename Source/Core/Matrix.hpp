#pragma once

template <unsigned int N, ArithmeticC T>
template <std::convertible_to<T> C>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>::Matrix(C t)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] = static_cast<T>(t);
    }
}

template <unsigned int N, ArithmeticC T>
template <std::convertible_to<T> C>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>::Matrix(Span<const C, N*N> data)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] = static_cast<T>(data[i]);
    }
}

template <unsigned int N, ArithmeticC T>
template <class... Args>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>::Matrix(const Args... dataList) requires (std::convertible_to<Args, T> && ...) && (sizeof...(Args) == N * N)
    : matrix{static_cast<T>(dataList) ...}
{}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>::Matrix(const Vector<N, T> rows[N])
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        UNROLL_LOOP
        for(unsigned int j = 0; j < N; j++)
        {
            matrix[i * N + j] = rows[i][j];
        }
    }
}

template <unsigned int N, ArithmeticC T>
template <unsigned int M>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>::Matrix(const Matrix<M, T>& other) requires (M > N)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        Vector<N, T> v = Vector<N, T>(other.matrix + i * M);
        UNROLL_LOOP
        for(int j = 0; j < N; j++)
        {
            matrix[i * N + j] = v[j];
        }
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T& Matrix<N, T>::operator[](unsigned int i)
{
    return matrix[i];
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const T& Matrix<N, T>::operator[](unsigned int i) const
{
    return matrix[i];
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T& Matrix<N, T>::operator()(unsigned int row, unsigned int column)
{
    return matrix[row * N + column];
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const T& Matrix<N, T>::operator()(unsigned int row, unsigned int column) const
{
    return matrix[row * N + column];
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const std::array<T, N*N>& Matrix<N, T>::AsArray() const
{
    return matrix;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr std::array<T, N*N>& Matrix<N, T>::AsArray()
{
    return matrix;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Matrix<N, T>::operator+=(const Matrix& right)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] += right.matrix[i];
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Matrix<N, T>::operator-=(const Matrix& right)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] -= right.matrix[i];
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Matrix<N, T>::operator*=(const Matrix& right)
{
    Matrix m = (*this) * right;
    *this = m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Matrix<N, T>::operator*=(T right)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] *= right;
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Matrix<N, T>::operator/=(const Matrix& right)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] /= right.matrix[i];
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Matrix<N, T>::operator/=(T right)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] /= right;
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::operator+(const Matrix& right) const
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] + right.matrix[i];
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::operator-(const Matrix& right) const
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] - right.matrix[i];
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::operator-() const requires SignedC<T>
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = -matrix[i];
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::operator/(const Matrix& right) const
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] / right.matrix[i];
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::operator/(T right) const
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] / right;
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::operator*(const Matrix& right) const
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        // Load the right column vector for this iteration
        // This is strided access unfortunately
        Vector<N, T> col;
        UNROLL_LOOP
        for(unsigned int j = 0; j < N; j++)
        {
            col[j] = right.matrix[i + j * N];
        }
        // Dot product with each row, write is strided again
        UNROLL_LOOP
        for(unsigned int j = 0; j < N; j++)
        {
            auto leftRow = Vector<N, T>(Span<const T, N>(matrix.data() + j * N, N));
            m(j, i) = leftRow.Dot(col);
        }
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
template <unsigned int M>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<M, T> Matrix<N, T>::operator*(const Vector<M, T>& right) const requires (M == N) || ((M + 1) == N)
{
    Vector<M, T> v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < M; i++)
    {
        auto leftRow = Vector<M, T>(Span<const T, M>(matrix.data() + i * N, M));
        v[i] = leftRow.Dot(right);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::operator*(T right) const
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] * right;
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Matrix<N, T>::operator==(const Matrix& right) const
{
    bool eq = true;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        eq &= matrix[i] == right.matrix[i];
    }
    return eq;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Matrix<N, T>::operator!=(const Matrix& right) const
{
    return !(*this == right);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Matrix<N, T>::Determinant() const requires (N == 2)
{
    const T* m = matrix;
    return m[0] * m[3] - m[1] * m[2];
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Matrix<N, T>::Determinant() const requires (N == 3)
{
    const T* m = matrix;
    T det1 = m[0] * (m[4] * m[8] - m[7] * m[5]);
    T det2 = m[1] * (m[3] * m[8] - m[5] * m[6]);
    T det3 = m[2] * (m[3] * m[7] - m[4] * m[6]);
    return det1 - det2 + det3;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Matrix<N, T>::Determinant() const requires (N == 4)
{
    const T* m = matrix;
    // Changing this to the PBRT-v4 version
    // https://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
    // I did not know you can expand non-row/column way.
    // Implementation of Equation 15.
    auto Det2x2 = [](T m00, T m01, T m10, T m11)
    {
        return m00 * m11 - m01 * m10;
    };

    // Notation here (x,y) show the row-major index of the matrix
    T s0 = Det2x2(m[0], m[1], m[4], m[5]);
    T s1 = Det2x2(m[0], m[2], m[4], m[6]);
    T s2 = Det2x2(m[0], m[3], m[4], m[7]);
    T s3 = Det2x2(m[1], m[2], m[5], m[6]);
    T s4 = Det2x2(m[1], m[3], m[5], m[7]);
    T s5 = Det2x2(m[2], m[3], m[6], m[7]);

    T c5 = Det2x2(m[10], m[11], m[14], m[15]);
    T c4 = Det2x2(m[ 9], m[11], m[13], m[15]);
    T c3 = Det2x2(m[ 9], m[10], m[13], m[14]);
    T c2 = Det2x2(m[ 8], m[11], m[12], m[15]);
    T c1 = Det2x2(m[ 8], m[10], m[12], m[14]);
    T c0 = Det2x2(m[ 8], m[ 9], m[12], m[13]);

    T det = ((s0 * c5) - (s1 * c4) + (s2 * c3) +
             (s3 * c2) - (s4 * c1) + (s5 * c0));
    return det;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Inverse() const requires std::floating_point<T> && (N == 2)
{
    const T* m = matrix;
    Matrix<2, T> result;
    T detRecip = 1 / Determinant(m);

    result[0] = +detRecip * m[3];
    result[1] = -detRecip * m[2];
    result[2] = -detRecip * m[1];
    result[3] = +detRecip * m[0];
    return result;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Inverse() const requires std::floating_point<T> && (N == 3)
{
    // Do not use determinant function here hand craft it
    // Some data is used on the matrix itself
    auto Det2x2 = [](T m00, T m01, T m10, T m11)
    {
        return m00 * m11 - m01 * m10;
    };

    const auto& m = matrix;
    T m00 = Det2x2(m[4], m[5], m[7], m[8]);  //
    T m01 = Det2x2(m[3], m[5], m[6], m[8]);  // Det Portion
    T m02 = Det2x2(m[3], m[4], m[6], m[7]);  //

    T m10 = Det2x2(m[1], m[2], m[7], m[8]);
    T m11 = Det2x2(m[0], m[2], m[6], m[8]);
    T m12 = Det2x2(m[0], m[1], m[6], m[7]);

    T m20 = Det2x2(m[1], m[2], m[4], m[5]);
    T m21 = Det2x2(m[0], m[2], m[3], m[5]);
    T m22 = Det2x2(m[0], m[1], m[3], m[4]);

    T det = m[0] * m00 - m[1] * m01 + m[2] * m02;
    T detInv = 1 / det;

    //return detInv * Matrix<3, T>( m00, -m01,  m02,
    //                             -m10,  m11, -m12,
    //                              m20, -m21,  m22);
    return detInv * Matrix<3, T>(m00, -m10, m20,
                                 -m01, m11, -m21,
                                 m02, -m12, m22);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Inverse() const requires std::floating_point<T> && (N == 4)
{
    // Changing this to the PBRT-v4 version
    // https://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
    // I did not know you can expand non-row/column way.
    // Implementation of Equation 15.
    auto Det2x2 = [](T m00, T m01, T m10, T m11)
    {
        return m00 * m11 - m01 * m10;
    };
    const auto& m = matrix;

    T s0 = Det2x2(m[0], m[1], m[4], m[5]);
    T s1 = Det2x2(m[0], m[2], m[4], m[6]);
    T s2 = Det2x2(m[0], m[3], m[4], m[7]);
    T s3 = Det2x2(m[1], m[2], m[5], m[6]);
    T s4 = Det2x2(m[1], m[3], m[5], m[7]);
    T s5 = Det2x2(m[2], m[3], m[6], m[7]);

    T c5 = Det2x2(m[10], m[11], m[14], m[15]);
    T c4 = Det2x2(m[ 9], m[11], m[13], m[15]);
    T c3 = Det2x2(m[ 9], m[10], m[13], m[14]);
    T c2 = Det2x2(m[ 8], m[11], m[12], m[15]);
    T c1 = Det2x2(m[ 8], m[10], m[12], m[14]);
    T c0 = Det2x2(m[ 8], m[ 9], m[12], m[13]);

    T det = ((s0 * c5) - (s1 * c4) + (s2 * c3) +
             (s3 * c2) - (s4 * c1) + (s5 * c0));
    T detInv = 1 / det;

    Matrix<N, T> inv
    (
        // Row0
        (+m[ 5] * c5 - m[ 6] * c4 + m[ 7] * c3),
        (-m[ 1] * c5 + m[ 2] * c4 - m[ 3] * c3),
        (+m[13] * s5 - m[14] * s4 + m[15] * s3),
        (-m[ 9] * s5 + m[10] * s4 - m[11] * s3),
        // Row1
        (-m[ 4] * c5 + m[ 6] * c2 - m[ 7] * c1),
        (+m[ 0] * c5 - m[ 2] * c2 + m[ 3] * c1),
        (-m[12] * s5 + m[14] * s2 - m[15] * s1),
        (+m[ 8] * s5 - m[10] * s2 + m[11] * s1),
        // Row2
        (+m[ 4] * c4 - m[ 5] * c2 + m[ 7] * c0),
        (-m[ 0] * c4 + m[ 1] * c2 - m[ 3] * c0),
        (+m[12] * s4 - m[13] * s2 + m[15] * s0),
        (-m[ 8] * s4 + m[ 9] * s2 - m[11] * s0),
        // Row3
        (-m[ 4] * c3 + m[ 5] * c1 - m[ 6] * c0),
        (+m[ 0] * c3 - m[ 1] * c1 + m[ 2] * c0),
        (-m[12] * s3 + m[13] * s1 - m[14] * s0),
        (+m[ 8] * s3 - m[ 9] * s1 + m[10] * s0)
    );
    return inv * detInv;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>& Matrix<N, T>::InverseSelf() requires std::floating_point<T>
{
    Matrix<N, T> m = Inverse();
    (*this) = m;
    return (*this);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Transpose() const
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        UNROLL_LOOP
        for(unsigned int j = 0; j < N; j++)
        {
            m(j, i) = (*this)(i, j);
        }
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>& Matrix<N, T>::TransposeSelf()
{
    UNROLL_LOOP
    for(unsigned int i = 1; i < N; i++)
    {
        UNROLL_LOOP
        for(unsigned int j = 0; j < i; j++)
        {
            T a = (*this)(i, j);
            (*this)(i, j) = (*this)(j, i);
            (*this)(j, i) = a;
        }
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Clamp(const Matrix& minVal, const Matrix& maxVal) const
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m[i] = min(max(minVal[i], matrix[i]), maxVal[i]);
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Clamp(T minVal, T maxVal) const
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m[i] = min(max(minVal, matrix[i]), maxVal);
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>& Matrix<N, T>::ClampSelf(const Matrix& minVal, const Matrix& maxVal)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] = min(max(minVal[i], matrix[i]), maxVal[i]);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>& Matrix<N, T>::ClampSelf(T minVal, T maxVal)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] = min(max(minVal, matrix[i]), maxVal);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Abs() const requires SignedC<T>
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m[i] = abs(matrix[i]);
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>& Matrix<N, T>::AbsSelf() requires SignedC<T>
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] = abs(matrix[i]);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Round() const requires std::floating_point<T>
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m[i] = round(matrix[i]);
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>& Matrix<N, T>::RoundSelf() requires std::floating_point<T>
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] = round(matrix[i]);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Floor() const requires std::floating_point<T>
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m[i] = floor(matrix[i]);
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>& Matrix<N, T>::FloorSelf() requires std::floating_point<T>
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] = floor(matrix[i]);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Ceil() const requires std::floating_point<T>
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m[i] = ceil(matrix[i]);
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T>& Matrix<N, T>::CeilSelf() requires std::floating_point<T>
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] = ceil(matrix[i]);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Lerp(const Matrix& mat0,
                                          const Matrix& mat1,
                                          T t)  requires std::floating_point<T>
{
    assert(t >= 0 && t <= 1);
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m[i] = (1 - t) * mat0[i] + t * mat1[i];
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
 MRAY_HYBRID MRAY_CGPU_INLINE
     constexpr Matrix<N, T> Matrix<N, T>::Min(const Matrix& mat0, const Matrix& mat1)
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m[i] = min(mat0[i], mat1[i]);
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Min(const Matrix& mat0, T t)
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m[i] = min(mat0[i], t);
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Max(const Matrix& mat0, const Matrix& mat1)
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m[i] = max(mat0[i], mat1[i]);
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Max(const Matrix& mat0, T t)
{
    Matrix m;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m[i] = max(mat0[i], t);
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Identity()
{
    Matrix<N, T> matrix;
    UNROLL_LOOP
    for(unsigned int y = 0; y < N; y++)
    {
        UNROLL_LOOP
        for(unsigned int  x = 0; x < N; x++)
        {
            matrix[y * N + x] = (x == y) ? T{1} : T{0};
        }
    }
    return matrix;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> Matrix<N, T>::Zero()
{
    return Matrix<N, T>(T{0});
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr RayT<T> Matrix<N, T>::TransformRay(const RayT<T>& r) const requires (N == 3)
{
    return RayT<T>((*this) * r.Dir(), (*this) * r.Pos());
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr RayT<T> Matrix<N, T>::TransformRay(const RayT<T>& r) const requires (N == 4)
{
    auto tDir = Vector<N - 1, T>((*this) * Vector<N, T>(r.Dir(), T{0}));
    auto tPos = Vector<N - 1, T>((*this) * Vector<N, T>(r.Pos(), T{1}));

    return RayT<T>(tDir, tPos);
}

template <unsigned int N, ArithmeticC T>
template <unsigned int M>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr AABB<M, T> Matrix<N, T>::TransformAABB(const AABB<M, T>& aabb) const requires((M + 1) == N)
{
    AABB<M, T> result = AABB<M, T>::Negative();
    for(unsigned int i = 0; i < AABB<M, T>::AABBVertexCount; i++)
    {
        Vector<N, T> vertex;
        UNROLL_LOOP
        for(unsigned int j = 0; j < M; j ++)
        {
            vertex[j] = ((i >> j) & 0b1) ? aabb.Max()[j] : aabb.Min()[j];
        }
        vertex[M] = T{1};
        vertex = (*this) * vertex;

        result.SetMax(Vector<M, T>::Max(result.Max(), Vector<M, T>(vertex)));
        result.SetMin(Vector<M, T>::Min(result.Min(), Vector<M, T>(vertex)));
    }
    return result;
}

template <unsigned int N, ArithmeticC T>
template <unsigned int M>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<M, T> Matrix<N, T>::LeftMultiply(const Vector<M, T>& normal)  const requires (M <= N)
{
    // Special case of left multiply
    // Instead of transposing matrix multiplying
    // the vector from "left"
    Vector<M, T> v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < M; i++)
    {
        T result = 0;
        UNROLL_LOOP
        for(unsigned int k = 0; k < M; k++)
        {
            result += matrix[i + N * k] * normal[k];
        }
        // Dot Product
        v[i] = result;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<N, T> operator*(T t, const Matrix<N, T>& mat)
{
    return mat * t;
}

// Spacial Matrix4x4 -> Matrix3x3
template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<4, T> ToMatrix4x4(const Matrix<3, T>& m)
{
    return Matrix<4, T>(m[0], m[1], m[2], 0,
                        m[3], m[4], m[5], 0,
                        m[6], m[7], m[8], 0,
                        0,    0,    0,    1);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<3, T> TransformGen::ExtractScale(const Matrix<4, T>& m)
{
    // This is not proper!
    // This should fail if transform matrix has shear
    // (didn't tested tho)
    //
    // Proper version, still assumes scale did not have translate component
    // https://theswissbay.ch/pdf/Gentoomen%20Library/Game%20Development/Programming/Graphics%20Gems%202.pdf
    // Chapter VII

    T sX = Vector<3, T>(m[0], m[1], m[2]).Length();
    T sY = Vector<3, T>(m[4], m[5], m[6]).Length();
    T sZ = Vector<3, T>(m[8], m[9], m[10]).Length();
    return Vector<3, T>(sX, sY, sZ);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<3, T> TransformGen::ExtractTranslation(const Matrix<4, T>& m)
{
    return Vector<3, T>(m[12], m[13], m[14]);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<4, T> TransformGen::Translate(const Vector<3, T>& v)
{
    //  1       0       0       tx
    //  0       1       0       ty
    //  0       0       1       tz
    //  0       0       0       1
    return Matrix<4, T>(1, 0, 0, v[0],
                        0, 1, 0, v[1],
                        0, 0, 1, v[2],
                        0, 0, 0,   1);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<4, T> TransformGen::Scale(T s)
{
    //  s       0       0       0
    //  0       s       0       0
    //  0       0       s       0
    //  0       0       0       1
    return Matrix<4, T>(s, 0, 0, 0,
                        0, s, 0, 0,
                        0, 0, s, 0,
                        0, 0, 0, 1);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<4, T> TransformGen::Scale(T x, T y, T z)
{
    //  sx      0       0       0
    //  0       sy      0       0
    //  0       0       sz      0
    //  0       0       0       1
    return Matrix<4, T>(x, 0, 0, 0,
                        0, y, 0, 0,
                        0, 0, z, 0,
                        0, 0, 0, 1);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<4, T> TransformGen::Rotate(T angle, const Vector<3, T>& axis)
{
    using namespace std;
    //  r       r       r       0
    //  r       r       r       0
    //  r       r       r       0
    //  0       0       0       1
    T tmp1, tmp2;

    T cosAngle = cos(angle);
    T sinAngle = sin(angle);
    T t = 1 - cosAngle;

    tmp1 = axis[0] * axis[1] * t;
    tmp2 = axis[2] * sinAngle;
    T m21 = tmp1 + tmp2;
    T m12 = tmp1 - tmp2;

    tmp1 = axis[0] * axis[2] * t;
    tmp2 = axis[1] * sinAngle;
    T m31 = tmp1 - tmp2;
    T m13 = tmp1 + tmp2;

    tmp1 = axis[1] * axis[2] * t;
    tmp2 = axis[0] * sinAngle;
    T m32 = tmp1 + tmp2;
    T m23 = tmp1 - tmp2;

    T m11 = cosAngle + axis[0] * axis[0] * t;
    T m22 = cosAngle + axis[1] * axis[1] * t;
    T m33 = cosAngle + axis[2] * axis[2] * t;

    return Matrix<4, T>(m11, m12, m13, 0,
                        m21, m22, m23, 0,
                        m31, m32, m33, 0,
                        0,   0,   0,   1);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<4, T> TransformGen::Rotate(const Quat<T>& q)
{
    Matrix<4, T> result;
    T xx = q[1] * q[1];
    T xy = q[1] * q[2];
    T xz = q[1] * q[3];
    T xw = q[1] * q[0];
    T yy = q[2] * q[2];
    T yz = q[2] * q[3];
    T yw = q[2] * q[0];
    T zz = q[3] * q[3];
    T zw = q[3] * q[0];
    result[0] = 1 - (2 * (yy + zz));
    result[1] = (2 * (xy - zw));
    result[2] = (2 * (xz + yw));
    result[3] = 0;

    result[4] = (2 * (xy + zw));
    result[5] = 1 - (2 * (xx + zz));
    result[6] = (2 * (yz - xw));
    result[7] = 0;

    result[8] = (2 * (xz - yw));
    result[9] = (2 * (yz + xw));
    result[10] = 1 - (2 * (xx + yy));
    result[11] = 0;

    result[12] = 0;
    result[13] = 0;
    result[14] = 0;
    result[15] = 1;
    return result;
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<4, T> TransformGen::Perspective(T fovXRadians, T aspectRatio,
                                                 T nearPlane, T farPlane)
{
    //  p       0       0       0
    //  0       p       0       0
    //  0       0       p       -1
    //  0       0       p       0
    T f = 1 / tan(fovXRadians * static_cast<T>(0.5));
    T m33 = (farPlane + nearPlane) / (nearPlane - farPlane);
    T m34 = (2 * farPlane * nearPlane) / (nearPlane - farPlane);

    return Matrix<4, T>(f, 0, 0, 0,
                        0, f * aspectRatio, 0, 0,
                        0, 0, m33, m34,
                        0, 0, -1, 0);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<4, T> TransformGen::Ortogonal(T left, T right,
                                               T top, T bottom,
                                               T nearPlane, T farPlane)
{
    //  orto    0       0       0
    //  0       orto    0       0
    //  0       0       orto    0
    //  orto    orto    orto    1
    T xt = -((right + left) / (right - left));
    T yt = -((top + bottom) / (top - bottom));
    T zt = -((farPlane + nearPlane) / (farPlane - nearPlane));
    T xs = 2 / (right - left);
    T ys = 2 / (top - bottom);
    T zs = 2 / (farPlane - nearPlane);
    return  Matrix<4, T>(xs,  0,  0, xt,
                          0, ys,  0, yt,
                          0,  0, zs, zt,
                          0,  0,  0, 1);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<4, T> TransformGen::Ortogonal(T width, T height,
                                               T nearPlane, T farPlane)
{
    //  orto    0       0       0
    //  0       orto    0       0
    //  0       0       orto    0
    //  0       0       orto    1
    T zt = nearPlane / (nearPlane - farPlane);
    return Matrix<4, T>(2 / width, 0, 0, 0,
                        0, 2 / height, 0, 0,
                        0, 0, 1 / (nearPlane - farPlane), 0,
                        0, 0, zt, 1);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<4, T> TransformGen::LookAt(const Vector<3, T>& eyePos,
                                            const Vector<3, T>& at,
                                            const Vector<3, T>& up)
{
    // Calculate Orthogonal Vectors for this rotation
    Vector<3, T> zAxis = (eyePos - at).NormalizeSelf();
    Vector<3, T> xAxis = up.CrossProduct(zAxis).NormalizeSelf();
    Vector<3, T> yAxis = zAxis.CrossProduct(xAxis).NormalizeSelf();

    // Also Add Translation part
    return Matrix<4, T>(xAxis[0], xAxis[1], xAxis[2], -xAxis.Dot(eyePos),
                        yAxis[0], yAxis[1], yAxis[2], -yAxis.Dot(eyePos),
                        zAxis[0], zAxis[1], zAxis[2], -zAxis.Dot(eyePos),
                               0,        0,        0,                  1);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<3, T> TransformGen::ToSpaceMat(const Vector<3, T>& x,
                                                const Vector<3, T>& y,
                                                const Vector<3, T>& z)
{
    return Matrix<3, T>(x, y, z);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Matrix<3, T> TransformGen::ToInvSpaceMat(const Vector<3, T>& x,
                                                   const Vector<3, T>& y,
                                                   const Vector<3, T>& z)
{
    return Matrix<3, T>(x[0], y[0], z[0],
                        x[1], y[1], z[1],
                        x[2], y[2], z[2]);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<3, T> TransformGen::YUpToZUp(const Vector<3, T>& v)
{
    return Vector<3, T>(v[2], v[0], v[1]);
}

template<std::floating_point T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<3, T> TransformGen::ZUpToYUp(const Vector<3, T>& v)
{
    return Vector<3, T>(v[1], v[2], v[0]);
}