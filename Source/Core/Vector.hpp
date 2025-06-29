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
constexpr Vector<N, T>::Vector(const Args... dataList)
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
constexpr Vector<N, T>::Vector(std::array<C, N>&& data)
    : vector(data)
{}

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
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        using Bit::NormConversion::FromUNorm;
        vector[i] = FromUNorm<T>(unorm[i]);
    }
}

template <unsigned int N, ArithmeticC T>
template <std::signed_integral IT>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T>::Vector(const SNorm<N, IT>& snorm)  requires (std::floating_point<T>)
{
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        using Bit::NormConversion::FromSNorm;
        vector[i] = FromSNorm<T>(snorm[i]);
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
constexpr unsigned int Vector<N, T>::Maximum() const
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
constexpr unsigned int Vector<N, T>::Minimum() const
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
        v[i] = std::min(std::max(minVal[i], vector[i]), maxVal[i]);
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
        hasNan |= (Math::IsNan(vector[i]) ||
                   Math::IsInf(vector[i]) ||
                   (vector[i] != vector[i]) ||
                   (vector[i] == std::numeric_limits<T>::infinity()) ||
                   (vector[i] == -std::numeric_limits<T>::infinity()));
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
constexpr Vector<N, T> Vector<N, T>::SqrtMax(const Vector&) requires std::floating_point<T>
{
    Vector v;
    UNROLL_LOOP
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = std::max(T(0), std::sqrt(v[i]));
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
        v[i] = Math::Lerp(v0[i], v1[i], t);
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
        v[i] = Math::Smoothstep(v0[i], v1[i], t);
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
    return Vector<3, T>(1, 0, 0);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N,T> Vector<N, T>::YAxis() requires (N == 3)
{
    return Vector<3, T>(0, 1, 0);
}

template <unsigned int N, ArithmeticC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> Vector<N, T>::ZAxis() requires (N == 3)
{
    return Vector<3, T>(0, 0, 1);
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
    #ifndef MRAY_DEVICE_CODE_PATH
    using namespace std;
    #endif
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