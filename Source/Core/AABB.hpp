#pragma once

#include "AABB.h"

template<unsigned int N, FloatC T>
MR_PF_DEF AABB<N, T>::AABB(const Vector<N, T>& min,
                           const Vector<N, T>& max) noexcept
    : min(min)
    , max(max)
{}

template<unsigned int N, FloatC T>
MR_PF_DEF AABB<N, T>::AABB(Span<const T, N> dataMin,
                           Span<const T, N> dataMax) noexcept
    : min(dataMin)
    , max(dataMax)
{}

template<unsigned int N, FloatC T>
MR_PF_DEF const Vector<N, T>& AABB<N, T>::Min() const noexcept
{
    return min;
}

template<unsigned int N, FloatC T>
MR_PF_DEF const Vector<N, T>& AABB<N, T>::Max() const noexcept
{
    return max;
}

template<unsigned int N, FloatC T>
MR_PF_DEF Vector<N, T> AABB<N, T>::Min() noexcept
{
    return min;
}

template<unsigned int N, FloatC T>
MR_PF_DEF Vector<N, T> AABB<N, T>::Max() noexcept
{
    return max;
}

template<unsigned int N, FloatC T>
MR_PF_DEF void AABB<N, T>::SetMin(const Vector<N, T>& v) noexcept
{
    min = v;
}

template<unsigned int N, FloatC T>
MR_PF_DEF void AABB<N, T>::SetMax(const Vector<N, T>& v) noexcept
{
    max = v;
}

template<unsigned int N, FloatC T>
MR_PF_DEF Vector<N, T> AABB<N, T>::GeomSpan() const noexcept
{
    return (max - min);
}

template<unsigned int N, FloatC T>
MR_PF_DEF Vector<N, T> AABB<N, T>::Centroid() const noexcept
{
    return min + (GeomSpan() * T{0.5});
}

template<unsigned int N, FloatC T>
MR_PF_DEF AABB<N, T> AABB<N, T>::Union(const AABB<N, T>& aabb) const noexcept
{
    return AABB<N, T>(Vector<N, T>::Min(min, aabb.min),
                      Vector<N, T>::Max(max, aabb.max));
}

template<unsigned int N, FloatC T>
MR_PF_DEF bool AABB<N, T>::IsInside(const Vector<N, T>& point) const noexcept
{
    bool result = true;
    MRAY_UNROLL_LOOP_N(N)
    for(int i = 0; i < N; i++)
    {
        result &= (point[i] >= min[i] && point[i] <= max[i]);
    }
    return result;
}

template<unsigned int N, FloatC T>
MR_PF_DEF bool AABB<N, T>::IsOutside(const Vector<N, T>& point) const noexcept
{
    return !IsInside(point);
}

template<unsigned int N, FloatC T>
MR_PF_DEF Vector<N, T> AABB<N, T>::FurthestCorner(const Vector<N, T>& point) const noexcept
{
    Vector<N, T> result;
    MRAY_UNROLL_LOOP_N(N)
    for(int i = 0; i < N; i++)
    {
        T minDist = Math::Abs(point[i] - min[i]);
        T maxDist = Math::Abs(point[i] - max[i]);
        result[i] = (minDist > maxDist) ? min[i] : max[i];
    }
    return result;
}

template<unsigned int N, FloatC T>
MR_PF_DEF bool AABB<N, T>::IntersectsSphere(const Vector<N, T>& sphrPos,
                                            float sphrRadius) const noexcept
{
    // Graphics Gems 2
    // http://www.realtimerendering.com/resources/GraphicsGems/gems/BoxSphere.c
    T dmin = 0;
    MRAY_UNROLL_LOOP_N(N)
    for(int i = 0; i < N; i++)
    {
        if(sphrPos[i] < min[i])
            dmin +=(sphrPos[i] - min[i]) * (sphrPos[i] - min[i]);
        else if(sphrPos[i] > max[i])
            dmin += (sphrPos[i] - max[i]) * (sphrPos[i] - max[i]);
    }
    if(dmin < sphrRadius * sphrRadius)
        return true;
    return false;
}

template<unsigned int N, FloatC T>
MR_PF_DEF AABB<N, T> AABB<N, T>::Zero() noexcept
{
    return AABB(Vector<N, T>(0), Vector<N, T>(0));
}

template<unsigned int N, FloatC T>
MR_PF_DEF AABB<N, T> AABB<N, T>::Covering() noexcept
{
    return AABB(Vector<N, T>(-std::numeric_limits<T>::max()),
                Vector<N, T>(std::numeric_limits<T>::max()));
}

template<unsigned int N, FloatC T>
MR_PF_DEF AABB<N, T> AABB<N, T>::Negative() noexcept
{
    return AABB(Vector<N, T>(std::numeric_limits<T>::max()),
                Vector<N, T>(-std::numeric_limits<T>::max()));
}