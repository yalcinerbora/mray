#pragma once

#include "Vector.h"

template <unsigned int N, ArithmeticC T>
template<std::convertible_to<T> C>
MR_PF_DEF_V Vector<N, T>::Vector(C data)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = static_cast<T>(data);
    }
}

template <unsigned int N, ArithmeticC T>
template<std::convertible_to<T> C>
MR_PF_DEF_V Vector<N, T>::Vector(Span<const C, N> data)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = static_cast<T>(data[i]);
    }
}

template <unsigned int N, ArithmeticC T>
template <std::convertible_to<T>... Args>
MR_PF_DEF_V Vector<N, T>::Vector(const Args... dataList)
    : vector{static_cast<T>(dataList) ...}
{}

template <unsigned int N, ArithmeticC T>
template <class... Args>
MR_PF_DEF_V
Vector<N, T>::Vector(const Vector<N - sizeof...(Args), T>& v,
                     const Args... dataList)  requires (N - sizeof...(Args) > 1)
{
    constexpr int VS = N - sizeof...(dataList);
    MRAY_UNROLL_LOOP_N(VS)
    for(unsigned int i = 0; i < VS; i++)
    {
        vector[i] = v[i];
    }
    const T arr[] = {static_cast<T>(dataList)...};
    MRAY_UNROLL_LOOP_N(N - VS)
    for(unsigned int i = VS; i < N; i++)
    {
        vector[i] = arr[i - VS];
    }
}

template <unsigned int N, ArithmeticC T>
template<std::convertible_to<T> C>
MR_PF_DEF_V
Vector<N, T>::Vector(std::array<C, N>&& data)
    : vector(data)
{}

template <unsigned int N, ArithmeticC T>
template<std::convertible_to<T> C>
MR_PF_DEF_V Vector<N, T>::Vector(const Vector<N, C>& other)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = static_cast<T>(other[i]);
    }
}

template <unsigned int N, ArithmeticC T>
template <unsigned int M>
MR_PF_DEF_V Vector<N, T>::Vector(const Vector<M, T>& other) requires (M > N)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] = other[i];
    }
}

template <unsigned int N, ArithmeticC T>
template <std::unsigned_integral IT>
MR_PF_DEF_V Vector<N, T>::Vector(const UNorm<N, IT>& unorm)  requires (std::floating_point<T>)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        using NormConversion::FromUNorm;
        vector[i] = FromUNorm<T>(unorm[i]);
    }
}

template <unsigned int N, ArithmeticC T>
template <std::signed_integral IT>
MR_PF_DEF_V Vector<N, T>::Vector(const SNorm<N, IT>& snorm)  requires (std::floating_point<T>)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        using NormConversion::FromSNorm;
        vector[i] = FromSNorm<T>(snorm[i]);
    }
}

template <unsigned int N, ArithmeticC T>
template<unsigned int M, class C>
MR_PF_DEF Vector<N, T>::operator Vector<M, C>() const requires (M <= N) && std::convertible_to<C, T>
{
    Vector<M, C> result;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < M; i++)
    {
        result[i] = static_cast<C>(vector[i]);
    }
    return result;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF T& Vector<N, T>::operator[](unsigned int i)
{
    return vector[i];
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF const T& Vector<N, T>::operator[](unsigned int i) const
{
    return vector[i];
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF const std::array<T, N>& Vector<N, T>::AsArray() const
{
    return vector;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF std::array<T, N>& Vector<N, T>::AsArray()
{
    return vector;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Vector<N, T>::operator+=(const Vector& right)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] += right[i];
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Vector<N, T>::operator-=(const Vector& right)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] -= right[i];
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Vector<N, T>::operator*=(const Vector& right)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] *= right[i];
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Vector<N, T>::operator*=(T right)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] *= right;
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Vector<N, T>::operator/=(const Vector& right)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] /= right[i];
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF_V void Vector<N, T>::operator/=(T right)
{
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        vector[i] /= right;
    }
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator+(const Vector& right) const
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] + right[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator+(T r) const
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] + r;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator-(const Vector& right) const
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] - right[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator-(T r) const
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] - r;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator-() const requires SignedC<T>
{
    Vector<N, T> v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = -vector[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator*(const Vector& right) const
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] * right[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator*(T right) const
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] * right;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator/(const Vector& right) const
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] / right[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator/(T right) const
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] / right;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator%(const Vector& right) const requires IntegralC<T>
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] % right[i];
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator%(T right) const requires IntegralC<T>
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = vector[i] % right;
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator%(const Vector& right) const requires FloatC<T>
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = fmod(vector[i], right[i]);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::operator%(T right) const requires FloatC<T>
{
    Vector v;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        v[i] = fmod(vector[i], right);
    }
    return v;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF bool Vector<N, T>::operator==(const Vector& right) const
{
    bool b = true;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        b &= vector[i] == right[i];
    }
    return b;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF bool Vector<N, T>::operator!=(const Vector& right) const
{
    return !(*this == right);
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF bool Vector<N, T>::operator<(const Vector& right) const
{
    bool b = true;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        b &= vector[i] < right[i];
    }
    return b;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF bool Vector<N, T>::operator<=(const Vector& right) const
{
    bool b = true;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        b &= vector[i] <= right[i];
    }
    return b;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF bool Vector<N, T>::operator>(const Vector& right) const
{
    bool b = true;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        b &= vector[i] > right[i];
    }
    return b;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF bool Vector<N, T>::operator>=(const Vector& right) const
{
    bool b = true;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        b &= vector[i] >= right[i];
    }
    return b;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF T Vector<N, T>::Sum() const
{
    T result = 0;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        result += vector[i];
    }
    return result;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF T Vector<N, T>::Multiply() const
{
    T result = 1;
    MRAY_UNROLL_LOOP_N(N)
    for(unsigned int i = 0; i < N; i++)
    {
        result *= vector[i];
    }
    return result;
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF unsigned int Vector<N, T>::Maximum() const
{
    unsigned int result = 0;
    T max = vector[0];
    MRAY_UNROLL_LOOP_N(N)
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
MR_PF_DEF unsigned int Vector<N, T>::Minimum() const
{
    unsigned int result = 0;
    T min = vector[0];
    MRAY_UNROLL_LOOP_N(N)
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
MR_PF_DEF Vector<N, T> Vector<N, T>::Zero()
{
    return Vector<N, T>(0);
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::XAxis() requires (N == 3)
{
    return Vector<3, T>(1, 0, 0);
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::YAxis() requires (N == 3)
{
    return Vector<3, T>(0, 1, 0);
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> Vector<N, T>::ZAxis() requires (N == 3)
{
    return Vector<3, T>(0, 0, 1);
}

template <unsigned int N, ArithmeticC T>
MR_PF_DEF Vector<N, T> operator*(T left, const Vector<N, T>& vec)
{
    return vec * left;
}