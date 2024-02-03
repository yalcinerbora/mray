#pragma once

template <unsigned int N, ArithmeticC T>
template<std::convertible_to<T> C>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>::Vector(C data)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = static_cast<T>(data);
    }
}

template <unsigned int N, ArithmeticC T>
template<std::convertible_to<T> C>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>::Vector(Span<const C, N> data)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = static_cast<T>(data[i]);
    }
}

template <unsigned int N, ArithmeticC T>
template <std::convertible_to<T>... Args>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>::Vector(const Args... dataList)// requires (std::convertible_to<Args, T> && ...) && (sizeof...(Args) == N)
    : vector{static_cast<T>(dataList) ...}
{}

template <unsigned int N, ArithmeticC T>
template <class... Args>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>::Vector(const Vector<N - sizeof...(Args), T>& v,
                               const Args... dataList)  requires (N - sizeof...(Args) > 1)
{
    constexpr int vectorSize = N - sizeof...(dataList);
    UNROLL_LOOP
    for(unsigned int i = 0; i < vectorSize; i++)
    {
        vector[i] = v[i];
    }
    const T arr[] = {static_cast<T>(dataList)...};
    UNROLL_LOOP
    for(unsigned int i = vectorSize; i < N; i++)
    {
        vector[i] = arr[i - vectorSize];
    }
}

template <unsigned int N, ArithmeticC T>
template<std::convertible_to<T> C>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>::Vector(const Vector<N, C>& other)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = static_cast<T>(other[i]);
    }
}

template <unsigned int N, ArithmeticC T>
template <unsigned int M>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>::Vector(const Vector<M, T>& other) requires (M > N)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = other[i];
    }
}

template <unsigned int N, ArithmeticC T>
template <std::unsigned_integral IT>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>::Vector(const UNorm<N, IT>& unorm)  requires (std::floating_point<T>)
{
    constexpr float MAX = static_cast<float>(UNorm<N, IT>::Max());
    constexpr float MIN = static_cast<float>(UNorm<N, IT>::Min());
    constexpr float DELTA = 1 / (MAX - MIN);
    // TODO: Specialize using intrinsics maybe?
    // Also check more precise way to do this (if available?)
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        IT it = unorm[i];
        T result = MIN + static_cast<float>(it) * DELTA;
        vector[i] = result;
    }
}

template <unsigned int N, ArithmeticC T>
template <std::signed_integral IT>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>::Vector(const SNorm<N, IT>& snorm)  requires (std::floating_point<T>)
{
    // Math representation (assuming T is char)
    // [-128, ..., 0, ..., 127]
    // However due to 2's complement, bit to data layout is
    // [0, ..., 127, -128, ..., -1]
    // DirectX representation is
    // [0, ..., 127, -127, -127, ..., -1]
    //               ----^-----
    //               notice the two -127's
    // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-data-conversion
    // we will use classic 2's complement representation for this conversion
    // so negative numbers will have **higher** precision
    //
    // TODO: check if this is not OK
    constexpr T MIN = static_cast<T>(UNorm<N, IT>::Min());
    constexpr T MAX = static_cast<T>(UNorm<N, IT>::Max());
    constexpr T NEG_DELTA = 1 / (0 - MIN);
    constexpr T POS_DELTA = 1 / (MAX - 0);
    // Sanity check
    static_assert(MIN < 0 && MAX > 0, "For snorm types; zero should be between \"min-max\"");

    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        IT it = snorm[i];
        T delta = (it < 0) ? NEG_DELTA : POS_DELTA;
        T result = static_cast<float>(it) * delta;
        vector[i] = result;
    }
}

template <unsigned int N, ArithmeticC T>
template<unsigned int M, class C>
MRAY_HYBRID MRAY_CGPU_INLINE
Vector<N, T>::operator Vector<M, C>() const requires (M <= N) && std::convertible_to<C, T>
{
    Vector<M, C> result;
    UNROLL_LOOP
    for(unsigned int i = 0; i < M; i++)
    {
        result[i] = static_cast<C>(vector[i]);
    }
    return result;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T& Vector<N, T>::operator[](unsigned int i)
{
    return vector[i];
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const T& Vector<N, T>::operator[](unsigned int i) const
{
    return vector[i];
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const std::array<T, N>& Vector<N, T>::AsArray() const
{
    return vector;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr std::array<T, N>& Vector<N, T>::AsArray()
{
    return vector;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Vector<N, T>::operator+=(const Vector& right)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] += right[i];
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Vector<N, T>::operator-=(const Vector& right)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] -= right[i];
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Vector<N, T>::operator*=(const Vector& right)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] *= right[i];
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Vector<N, T>::operator*=(T right)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] *= right;
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Vector<N, T>::operator/=(const Vector& right)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] /= right[i];
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Vector<N, T>::operator/=(T right)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] /= right;
    }
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator+(const Vector& right) const
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] + right[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator+(T r) const
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] + r;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator-(const Vector& right) const
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] - right[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator-(T r) const
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] - r;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator-() const requires SignedC<T>
{
    Vector<N, T> v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = -vector[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator*(const Vector& right) const
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] * right[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator*(T right) const
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] * right;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator/(const Vector& right) const
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] / right[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator/(T right) const
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] / right;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator%(const Vector& right) const requires std::integral<T>
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] % right[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator%(T right) const requires std::integral<T>
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] % right;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator%(const Vector& right) const requires std::floating_point<T>
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = fmod(vector[i], right[i]);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::operator%(T right) const requires std::floating_point<T>
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = fmod(vector[i], right);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Vector<N, T>::operator==(const Vector& right) const
{
    bool b = true;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        b &= vector[i] == right[i];
    }
    return b;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Vector<N, T>::operator!=(const Vector& right) const
{
    return !(*this == right);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Vector<N, T>::operator<(const Vector& right) const
{
    bool b = true;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        b &= vector[i] < right[i];
    }
    return b;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Vector<N, T>::operator<=(const Vector& right) const
{
    bool b = true;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        b &= vector[i] <= right[i];
    }
    return b;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Vector<N, T>::operator>(const Vector& right) const
{
    bool b = true;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        b &= vector[i] > right[i];
    }
    return b;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Vector<N, T>::operator>=(const Vector& right) const
{
    bool b = true;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        b &= vector[i] >= right[i];
    }
    return b;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Vector<N, T>::Dot(const Vector& right) const
{
    T data = 0;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        data += (vector[i] * right[i]);
    }
    return data;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Vector<N, T>::Sum() const
{
    T result = 0;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        result += vector[i];
    }
    return result;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Vector<N, T>::Multiply() const
{
    T result = 1;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        result *= vector[i];
    }
    return result;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr unsigned int Vector<N, T>::Max() const
{
    unsigned int result = 0;
    T max = vector[0];
    UNROLL_LOOP
    for(unsigned int i = 1; i < N; i++)
    {
        if(vector[i] > max)
        {
            max = vector[i];
            result = i;
        }
    }
    return result;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr unsigned int Vector<N, T>::Min() const
{
    unsigned int result = 0;
    T min = vector[0];
    UNROLL_LOOP
    for(unsigned int i = 1; i < N; i++)
    {
        if(vector[i] < min)
        {
            min = vector[i];
            result = i;
        }
    }
    return result;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Vector<N, T>::Length() const requires std::floating_point<T>
{
    return std::sqrt(LengthSqr());
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Vector<N, T>::LengthSqr() const
{
    return Dot(*this);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Normalize() const requires std::floating_point<T>
{
    T lengthInv = static_cast<T>(1) / Length();

    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] * lengthInv;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>& Vector<N, T>::NormalizeSelf() requires std::floating_point<T>
{
    T lengthInv = static_cast<T>(1) / Length();
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] *= lengthInv;
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Clamp(const Vector& minVal, const Vector& maxVal) const
{
    Vector v;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        v[i] = min(max(minVal[i], vector[i]), maxVal[i]);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Clamp(T minVal, T maxVal) const
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = std::min(std::max(minVal, vector[i]), maxVal);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>& Vector<N, T>::ClampSelf(const Vector& minVal, const Vector& maxVal)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = std::min(std::max(minVal[i], vector[i]), maxVal[i]);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>& Vector<N, T>::ClampSelf(T minVal, T maxVal)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = std::min(std::max(minVal, vector[i]), maxVal);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Vector<N, T>::HasNaN() const requires std::floating_point<T>
{
    bool hasNan = false;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        hasNan |= (std::isnan(vector[i]) ||
                   std::isinf(vector[i]) ||
                   (vector[i] != vector[i]) ||
                   (vector[i] == INFINITY) ||
                   (vector[i] == -INFINITY));
    }
    return hasNan;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Abs() const requires SignedC<T>
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = std::abs(vector[i]);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>& Vector<N, T>::AbsSelf() requires SignedC<T>
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = std::abs(vector[i]);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Round() const requires std::floating_point<T>
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = std::round(vector[i]);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>& Vector<N, T>::RoundSelf() requires std::floating_point<T>
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = std::round(vector[i]);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Floor() const requires std::floating_point<T>
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = std::floor(vector[i]);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>& Vector<N, T>::FloorSelf() requires std::floating_point<T>
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = std::floor(vector[i]);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Ceil() const requires std::floating_point<T>
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = std::ceil(vector[i]);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>& Vector<N, T>::CeilSelf() requires std::floating_point<T>
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = std::ceil(vector[i]);
    }
    return *this;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Sqrt(const Vector<N, T>& v0) requires std::floating_point<T>
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = std::sqrt(v0[i]);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Min(const Vector& v0, const Vector& v1)
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = std::min(v0[i], v1[i]);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Min(const Vector& v0, T v1)
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = std::min(v0[i], v1);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Max(const Vector& v0, const Vector& v1)
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = std::max(v0[i], v1[i]);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Max(const Vector& v0, T v1)
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = std::max(v0[i], v1);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Lerp(const Vector& v0, const Vector& v1, T t) requires std::floating_point<T>
{
    assert(t >= 0 && t <= 1);
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = MathFunctions::Lerp(v0[i], v1[i], t);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Smoothstep(const Vector& v0, const Vector& v1, T t) requires std::floating_point<T>
{
    assert(t >= 0 && t <= 1);
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = MathFunctions::Smoothstep(v0[i], v1[i], t);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Zero()
{
    return Vector<N, T>(0);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::XAxis() requires (N == 3)
{
    return Vector3(1, 0, 0);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N,T> Vector<N, T>::YAxis() requires (N == 3)
{
    return Vector3(0, 1, 0);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::ZAxis() requires (N == 3)
{
    return Vector3(0, 0, 1);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::Cross(const Vector<N, T>& v0, const Vector<N, T>& v1) requires std::floating_point<T> && (N == 3)
{
    Vector<3, T> result(v0[1] * v1[2] - v0[2] * v1[1],
                        v0[2] * v1[0] - v0[0] * v1[2],
                        v0[0] * v1[1] - v0[1] * v1[0]);
    return result;
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::OrthogonalVector(const Vector<N, T>& v) requires std::floating_point<T> && (N == 3)
{
    // PBRT Book
    // https://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Vectors#CoordinateSystem
    if(abs(v[0]) > abs(v[1]))
        return Vector<3, T>(-v[2], 0, v[0]) / sqrt(v[0] * v[0] + v[2] * v[2]);
    else
        return Vector<3, T>(0, v[2], -v[1]) / sqrt(v[1] * v[1] + v[2] * v[2]);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> operator*(T left, const Vector<N, T>& vec)
{
    return vec * left;
}