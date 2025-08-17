#pragma once

template <unsigned int N, ArithmeticC T>
template <std::convertible_to<T> C>
MR_PF_DEF_V Matrix<N, T>::Matrix(C t) noexcept
{
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] = static_cast<T>(t);
    }
}

template <unsigned int N, ArithmeticC T>
template <std::convertible_to<T> C>
MR_PF_DEF_V Matrix<N, T>::Matrix(Span<const C, N*N> data) noexcept
{
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] = static_cast<T>(data[i]);
    }
}

template <unsigned int N, ArithmeticC T>
template <class... Args>
MR_PF_DEF_V Matrix<N, T>::Matrix(const Args... dataList) noexcept
requires(std::convertible_to<Args, T> && ...) && (sizeof...(Args) == N * N)
    : matrix{static_cast<T>(dataList) ...}
{}

template <unsigned int N, ArithmeticC T>
template <class... Rows>
MR_PF_DEF_V Matrix<N, T>::Matrix(const Rows&... rows) noexcept
requires(std::is_same_v<Rows, Vector<N, T>> && ...) && (sizeof...(Rows) == N)
{
    auto Write = [this](const Vector3& v, unsigned int row) -> void
    {
        MRAY_UNROLL_LOOP
        for(unsigned int j = 0; j < N; j++)
        {
            matrix[row * N + j] = v[j];
        }
    };
    auto GenSequence = [Write]<unsigned int... I>
    (
        std::integer_sequence<unsigned int, I...>,
        Tuple<const Rows&...> tuple
    ) -> void
    {
        // Parameter pack expansion
        (
            (Write(get<I>(tuple), I)),
            ...
        );
    };
    // Utilize "std::get" via tuple
    GenSequence(std::make_integer_sequence<unsigned int, sizeof...(Rows)>{},
                Tuple<const Rows&...>(rows...));
}

template <unsigned int N, ArithmeticC T>
template <unsigned int M>
MR_PF_DEF_V Matrix<N, T>::Matrix(const Matrix<M, T>& other) noexcept requires (M > N)
{
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        Vector<N, T> v = Vector<N, T>(other.matrix + i * M);
        MRAY_UNROLL_LOOP
        for(int j = 0; j < N; j++)
        {
            matrix[i * N + j] = v[j];
        }
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF T& Matrix<N, T>::operator[](unsigned int i) noexcept
{
    return matrix[i];
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF const T& Matrix<N, T>::operator[](unsigned int i) const noexcept
{
    return matrix[i];
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF T& Matrix<N, T>::operator()(unsigned int row, unsigned int column) noexcept
{
    return matrix[row * N + column];
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF const T& Matrix<N, T>::operator()(unsigned int row, unsigned int column) const noexcept
{
    return matrix[row * N + column];
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF const std::array<T, N*N>& Matrix<N, T>::AsArray() const noexcept
{
    return matrix;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF std::array<T, N*N>& Matrix<N, T>::AsArray() noexcept
{
    return matrix;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Matrix<N, T>::operator+=(const Matrix& right) noexcept
{
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] += right.matrix[i];
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Matrix<N, T>::operator-=(const Matrix& right) noexcept
{
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] -= right.matrix[i];
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Matrix<N, T>::operator*=(const Matrix& right) noexcept
{
    Matrix m = (*this) * right;
    *this = m;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Matrix<N, T>::operator*=(T right) noexcept
{
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] *= right;
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Matrix<N, T>::operator/=(const Matrix& right) noexcept
{
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] /= right.matrix[i];
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Matrix<N, T>::operator/=(T right) noexcept
{
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        matrix[i] /= right;
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V Matrix<N, T> Matrix<N, T>::operator+(const Matrix& right) const noexcept
{
    Matrix m;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] + right.matrix[i];
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Matrix<N, T> Matrix<N, T>::operator-(const Matrix& right) const noexcept
{
    Matrix m;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] - right.matrix[i];
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Matrix<N, T> Matrix<N, T>::operator-() const noexcept requires SignedC<T>
{
    Matrix m;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = -matrix[i];
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Matrix<N, T> Matrix<N, T>::operator/(const Matrix& right) const noexcept
{
    Matrix m;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] / right.matrix[i];
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Matrix<N, T> Matrix<N, T>::operator/(T right) const noexcept
{
    Matrix m;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] / right;
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Matrix<N, T> Matrix<N, T>::operator*(const Matrix& right) const noexcept
{
    Matrix m;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        // Load the right column vector for this iteration
        // This is strided access unfortunately
        Vector<N, T> col;
        MRAY_UNROLL_LOOP
        for(unsigned int j = 0; j < N; j++)
        {
            col[j] = right.matrix[i + j * N];
        }
        // Dot product with each row, write is strided again
        MRAY_UNROLL_LOOP
        for(unsigned int j = 0; j < N; j++)
        {
            using Math::Dot;
            auto leftRow = Vector<N, T>(Span<const T, N>(matrix.data() + j * N, N));
            m(j, i) = Dot(leftRow, col);
        }
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
template <unsigned int M>
MR_PF_DEF Vector<M, T> Matrix<N, T>::operator*(const Vector<M, T>& right) const noexcept requires (M == N) || ((M + 1) == N)
{
    Vector<M, T> v;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < M; i++)
    {
        using Math::Dot;
        auto leftRow = Vector<M, T>(Span<const T, M>(matrix.data() + i * N, M));
        v[i] = Dot(leftRow, right);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Matrix<N, T> Matrix<N, T>::operator*(T right) const noexcept
{
    Matrix m;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] * right;
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF bool Matrix<N, T>::operator==(const Matrix& right) const noexcept
{
    bool eq = true;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N * N; i++)
    {
        eq &= matrix[i] == right.matrix[i];
    }
    return eq;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF bool Matrix<N, T>::operator!=(const Matrix& right) const noexcept
{
    return !(*this == right);
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF T Matrix<N, T>::Determinant() const noexcept requires (N == 2)
{
    const T* m = matrix;
    return m[0] * m[3] - m[1] * m[2];
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF T Matrix<N, T>::Determinant() const noexcept requires (N == 3)
{
    const T* m = matrix;
    T det1 = m[0] * (m[4] * m[8] - m[7] * m[5]);
    T det2 = m[1] * (m[3] * m[8] - m[5] * m[6]);
    T det3 = m[2] * (m[3] * m[7] - m[4] * m[6]);
    return det1 - det2 + det3;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF T Matrix<N, T>::Determinant() const noexcept requires (N == 4)
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
MR_PF_DEF Matrix<N, T> Matrix<N, T>::Inverse() const noexcept requires FloatC<T> && (N == 2)
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
MR_PF_DEF Matrix<N, T> Matrix<N, T>::Inverse() const noexcept requires FloatC<T> && (N == 3)
{
    // Do not use determinant function here hand craft it
    // Some data is used on the matrix itself
    constexpr auto Det2x2 = [](T m00, T m01, T m10, T m11)
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
    T detInv = T(1) / det;

    return detInv * Matrix<3, T>(m00, -m10, m20,
                                 -m01, m11, -m21,
                                 m02, -m12, m22);
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Matrix<N, T> Matrix<N, T>::Inverse() const noexcept requires FloatC<T> && (N == 4)
{
    // Changing this to the PBRT-v4 version
    // https://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
    // I did not know you can expand non-row/column way.
    // Implementation of Equation 15.
    constexpr auto Det2x2 = [](T m00, T m01, T m10, T m11)
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
MR_PF_DEF Matrix<N, T> Matrix<N, T>::Transpose() const noexcept
{
    Matrix m;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        MRAY_UNROLL_LOOP
        for(unsigned int j = 0; j < N; j++)
        {
            m(j, i) = (*this)(i, j);
        }
    }
    return m;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Matrix<N, T> Matrix<N, T>::Identity() noexcept
{
    Matrix<N, T> matrix;
    MRAY_UNROLL_LOOP
    for(unsigned int y = 0; y < N; y++)
    {
        MRAY_UNROLL_LOOP
        for(unsigned int  x = 0; x < N; x++)
        {
            matrix[y * N + x] = (x == y) ? T{1} : T{0};
        }
    }
    return matrix;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Matrix<N, T> Matrix<N, T>::Zero() noexcept
{
    return Matrix<N, T>(T{0});
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF RayT<T> Matrix<N, T>::TransformRay(const RayT<T>& r) const noexcept requires (N == 3)
{
    return RayT<T>((*this) * r.Dir(), (*this) * r.Pos());
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF RayT<T> Matrix<N, T>::TransformRay(const RayT<T>& r) const noexcept requires (N == 4)
{
    auto tDir = Vector<N - 1, T>((*this) * Vector<N, T>(r.Dir(), T{0}));
    auto tPos = Vector<N - 1, T>((*this) * Vector<N, T>(r.Pos(), T{1}));

    return RayT<T>(tDir, tPos);
}

template <unsigned int N, ArithmeticC T>
template <unsigned int M>
MR_PF_DEF AABB<M, T> Matrix<N, T>::TransformAABB(const AABB<M, T>& aabb) const noexcept requires((M + 1) == N)
{
    AABB<M, T> result = AABB<M, T>::Negative();
    for(unsigned int i = 0; i < AABB<M, T>::AABBVertexCount; i++)
    {
        Vector<N, T> vertex;
        MRAY_UNROLL_LOOP
        for(unsigned int j = 0; j < M; j ++)
        {
            vertex[j] = ((i >> j) & 0b1) ? aabb.Max()[j] : aabb.Min()[j];
        }
        vertex[M] = T{1};
        vertex = (*this) * vertex;

        result.SetMax(Math::Max(result.Max(), Vector<M, T>(vertex)));
        result.SetMin(Math::Min(result.Min(), Vector<M, T>(vertex)));
    }
    return result;
}

template <unsigned int N, ArithmeticC T>
template <unsigned int M>
MR_PF_DEF Vector<M, T> Matrix<N, T>::LeftMultiply(const Vector<M, T>& normal) const noexcept requires (M <= N)
{
    // Special case of left multiply
    // Instead of transposing matrix multiplying
    // the vector from "left"
    Vector<M, T> v;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < M; i++)
    {
        T result = 0;
        MRAY_UNROLL_LOOP
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
MR_PF_DEF Matrix<N, T> operator*(T t, const Matrix<N, T>& mat) noexcept
{
    return mat * t;
}

// Spacial Matrix4x4 -> Matrix3x3
template<FloatC T>
MR_PF_DEF Matrix<4, T> ToMatrix4x4(const Matrix<3, T>& m) noexcept
{
    return Matrix<4, T>(m[0], m[1], m[2], 0,
                        m[3], m[4], m[5], 0,
                        m[6], m[7], m[8], 0,
                        0,    0,    0,    1);
}

template<FloatC T>
MR_PF_DEF Vector<3, T> TransformGen::ExtractScale(const Matrix<4, T>& m) noexcept
{
    // This is not proper!
    // This should fail if transform matrix has shear
    // (didn't tested tho)
    //
    // Proper version, still assumes scale did not have translate component
    // https://theswissbay.ch/pdf/Gentoomen%20Library/Game%20Development/Programming/Graphics%20Gems%202.pdf
    // Chapter VII

    T sX = Math::Length(Vector<3, T>(m[0], m[1], m[2]));
    T sY = Math::Length(Vector<3, T>(m[4], m[5], m[6]));
    T sZ = Math::Length(Vector<3, T>(m[8], m[9], m[10]));
    return Vector<3, T>(sX, sY, sZ);
}

template<FloatC T>
MR_PF_DEF Vector<3, T> TransformGen::ExtractTranslation(const Matrix<4, T>& m) noexcept
{
    return Vector<3, T>(m[12], m[13], m[14]);
}

template<FloatC T>
MR_PF_DEF Matrix<4, T> TransformGen::Translate(const Vector<3, T>& v) noexcept
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

template<FloatC T>
MR_PF_DEF Matrix<4, T> TransformGen::Scale(T s) noexcept
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

template<FloatC T>
MR_PF_DEF Matrix<4, T> TransformGen::Scale(T x, T y, T z) noexcept
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

template<FloatC T>
MR_PF_DEF Matrix<4, T> TransformGen::Rotate(T angle, const Vector<3, T>& axis) noexcept
{
    //  r       r       r       0
    //  r       r       r       0
    //  r       r       r       0
    //  0       0       0       1
    T tmp1, tmp2;

    const auto& [sinTheta, cosTheta] = Math::SinCos(angle);
    T t = 1 - cosTheta;

    tmp1 = axis[0] * axis[1] * t;
    tmp2 = axis[2] * sinTheta;
    T m21 = tmp1 + tmp2;
    T m12 = tmp1 - tmp2;

    tmp1 = axis[0] * axis[2] * t;
    tmp2 = axis[1] * sinTheta;
    T m31 = tmp1 - tmp2;
    T m13 = tmp1 + tmp2;

    tmp1 = axis[1] * axis[2] * t;
    tmp2 = axis[0] * sinTheta;
    T m32 = tmp1 + tmp2;
    T m23 = tmp1 - tmp2;

    T m11 = cosTheta + axis[0] * axis[0] * t;
    T m22 = cosTheta + axis[1] * axis[1] * t;
    T m33 = cosTheta + axis[2] * axis[2] * t;

    return Matrix<4, T>(m11, m12, m13, 0,
                        m21, m22, m23, 0,
                        m31, m32, m33, 0,
                        0,   0,   0,   1);
}

template<FloatC T>
MR_PF_DEF Matrix<4, T> TransformGen::Rotate(const Quat<T>& q) noexcept
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

template<FloatC T>
MR_PF_DEF Matrix<4, T> TransformGen::Perspective(T fovXRadians, T aspectRatio,
                                                 T nearPlane, T farPlane) noexcept
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

template<FloatC T>
MR_PF_DEF Matrix<4, T> TransformGen::Orthogonal(T left, T right,
                                                T top, T bottom,
                                                T nearPlane, T farPlane) noexcept
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

template<FloatC T>
MR_PF_DEF Matrix<4, T> TransformGen::Orthogonal(T width, T height,
                                                T nearPlane, T farPlane) noexcept
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

template<FloatC T>
MR_PF_DEF Matrix<4, T> TransformGen::LookAt(const Vector<3, T>& eyePos,
                                            const Vector<3, T>& at,
                                            const Vector<3, T>& up) noexcept
{
    // Calculate Orthogonal Vectors for this rotation
    Vector<3, T> zAxis = Math::Normalize(eyePos - at);
    Vector<3, T> xAxis = Math::Normalize(Math::Cross(up, zAxis));
    Vector<3, T> yAxis = Math::Normalize(Math::Cross(zAxis, xAxis));

    // Also Add Translation part
    return Matrix<4, T>(xAxis[0], xAxis[1], xAxis[2], Math::Dot(xAxis, eyePos),
                        yAxis[0], yAxis[1], yAxis[2], Math::Dot(yAxis, eyePos),
                        zAxis[0], zAxis[1], zAxis[2], Math::Dot(zAxis, eyePos),
                        0,        0,        0,        1);
}

template<FloatC T>
MR_PF_DEF Matrix<3, T> TransformGen::ToSpaceMat(const Vector<3, T>& x,
                                                const Vector<3, T>& y,
                                                const Vector<3, T>& z) noexcept
{
    return Matrix<3, T>(x, y, z);
}

template<FloatC T>
MR_PF_DEF Matrix<3, T> TransformGen::ToInvSpaceMat(const Vector<3, T>& x,
                                                   const Vector<3, T>& y,
                                                   const Vector<3, T>& z) noexcept
{
    return Matrix<3, T>(x[0], y[0], z[0],
                        x[1], y[1], z[1],
                        x[2], y[2], z[2]);
}

template<FloatC T>
MR_PF_DEF Vector<3, T> TransformGen::YUpToZUp(const Vector<3, T>& v) noexcept
{
    return Vector<3, T>(v[2], v[0], v[1]);
}

template<FloatC T>
MR_PF_DEF Vector<3, T> TransformGen::ZUpToYUp(const Vector<3, T>& v) noexcept
{
    return Vector<3, T>(v[1], v[2], v[0]);
}

template<FloatC T>
MR_PF_DEF Matrix<4, T> TransformGen::YUpToZUpMat() noexcept
{
    return Matrix<4, T>(0, 0, 1, 0,
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 0, 1);
}

template<FloatC T>
MR_PF_DEF Matrix<4, T> TransformGen::ZUpToYUpMat() noexcept
{
    return Matrix<4, T>(0, 1, 0, 0,
                        0, 0, 1, 0,
                        1, 0, 0, 0,
                        0, 0, 0, 1);
}