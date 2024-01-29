#pragma once

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr AABB<N, T>::AABB(const Vector<N, T>& min,
                           const Vector<N, T>& max)
    : min(min)
    , max(max)
{}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr AABB<N, T>::AABB(const T* dataMin,
                           const T* dataMax)
    : min(dataMin)
    , max(dataMax)
{}

template<unsigned int N, FloatingPointC T>
template <class... ArgsMin, class... ArgsMax>
requires (sizeof...(ArgsMin) == N) && (std::convertible_to<T, ArgsMin> && ...) &&
         (sizeof...(ArgsMax) == N) && (std::convertible_to<T, ArgsMax> && ...)
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr AABB<N, T>::AABB(const ArgsMin... dataListMin,
                           const ArgsMax... dataListMax)
    : min(dataListMin...)
    , max(dataListMax...)
{}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const Vector<N, T>& AABB<N, T>::Min() const
{
    return min;
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const Vector<N, T>& AABB<N, T>::Max() const
{
    return max;
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> AABB<N, T>::Min()
{
    return min;
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> AABB<N, T>::Max()
{
    return max;
}

template<unsigned int N, FloatingPointC T>
constexpr MRAY_HYBRID MRAY_CGPU_INLINE
void AABB<N, T>::SetMin(const Vector<N, T>& v)
{
    min = v;
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void AABB<N, T>::SetMax(const Vector<N, T>& v)
{
    max = v;
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> AABB<N, T>::Span() const
{
    return (max - min);
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> AABB<N, T>::Centroid() const
{
    return min + (Span() * T{0.5});
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr AABB<N, T> AABB<N, T>::Union(const AABB<N, T>& aabb) const
{
    return AABB<N, T>(Vector<N, T>::Min(min, aabb.min),
                      Vector<N, T>::Max(max, aabb.max));
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr AABB<N, T>& AABB<N, T>::UnionSelf(const AABB<N, T>& aabb)
{
    min = Vector<N, T>::Min(min, aabb.min);
    max = Vector<N, T>::Max(max, aabb.max);
    return *this;
}


template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool AABB<N, T>::IsInside(const Vector<N, T>& point) const
{
    bool result = true;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        result &= (point[i] >= min[i] && point[i] <= max[i]);
    }
    return result;
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool AABB<N, T>::IsOutside(const Vector<N, T>& point) const
{
    return !IsInside(point);
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<N, T> AABB<N, T>::FurthestCorner(const Vector<N, T>& point) const
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    Vector<N, T> result;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        T minDist = abs(point[i] - min[i]);
        T maxDist = abs(point[i] - max[i]);
        result[i] = (minDist > maxDist) ? min[i] : max[i];
    }
    return result;
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool AABB<N, T>::IntersectsSphere(const Vector<N, T>& sphrPos,
                                            float sphrRadius) const
{
    // Graphics Gems 2
    // http://www.realtimerendering.com/resources/GraphicsGems/gems/BoxSphere.c
    T dmin = 0;
    UNROLL_LOOP
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

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr AABB<N, T> AABB<N, T>::Zero()
{
    return AABB(Vector<N, T>(0), Vector<N, T>(0));
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr AABB<N, T> AABB<N, T>::Covering()
{
    return AABB(Vector<N, T>(-std::numeric_limits<T>::max()),
                Vector<N, T>(std::numeric_limits<T>::max()));
}

template<unsigned int N, FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr AABB<N, T> AABB<N, T>::Negative()
{
    return AABB(Vector<N, T>(std::numeric_limits<T>::max()),
                Vector<N, T>(-std::numeric_limits<T>::max()));
}