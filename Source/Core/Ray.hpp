#pragma once

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr RayT<T>::RayT(const Vector<3, T>& direction,
                        const Vector<3, T>& position)
    : direction(direction)
    , position(position)
{}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr RayT<T>::RayT(const Vector<3, T> vec[2])
    : direction(vec[0])
    , position(vec[1])
{}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool RayT<T>::IntersectsSphere(Vector<3, T>& intersectPos, T& t,
                                         const Vector<3, T>& sphereCenter,
                                         T sphereRadius) const
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef MRAY_DEVICE_CODE_PATH
        using namespace std;
    #endif

    // RayTracing Gems
    // Chapter 7: Precision Improvements for Ray/Sphere Intersection
    // This is similar to the geometric solution below
    // with two differences
    //
    // 1st: Beam half length square (discriminant variable)
    // is calculated using two vector differences instead of
    // Pythagorean theorem (r^2 - beamNormalLength^2).
    // METUray almost always holds the direction vector
    // normalized unless the ray is under inverse transformation
    // of an object's local transformation
    // (and that transformation has  a scale).
    // So we need to normalize the direction vector to properly
    // find out the discriminant
    //
    // 2nd: closer t value is calculated classically; however
    // further t value is calculated differently which I could
    // not explain geometrically probably a reduction of some
    // algebraic expression which improves numerical accuracy
    //
    // First improvement, has higher accuracy when
    // extremely large sphere is being intersected
    //
    // Second one is for spheres that is far away
    T dirLengthInv = 1.0f / direction.Length();
    Vector<3, T> dirNorm = direction * dirLengthInv;
    Vector<3, T> centerDir = sphereCenter - position;
    T beamCenterDist = dirNorm.Dot(centerDir);
    T cDirLengthSqr = centerDir.LengthSqr();

    // Below code is from the source
    Vector<3, T> remedyTerm = centerDir - beamCenterDist * dirNorm;
    T discriminant = sphereRadius * sphereRadius - remedyTerm.LengthSqr();
    if(discriminant >= 0)
    {
        T beamHalfLength = sqrt(discriminant);

        T t0 = (beamCenterDist >= 0)
            ? (beamCenterDist + beamHalfLength)
            : (beamCenterDist - beamHalfLength);
        T t1 = (cDirLengthSqr - sphereRadius * sphereRadius) / t0;

        // TODO: is there a better way to do this?
        // Select a T
        t = std::numeric_limits<T>::max();
        if(t0 > 0) t = min(t, t0);
        if(t1 > 0) t = min(t, t1);
        if(t != std::numeric_limits<T>::max())
        {
            t *= dirLengthInv;
            intersectPos = position + t * direction;
            return true;
        }
    }
    return false;

    // Geometric solution
    //Vector<3, T> centerDir = sphereCenter - position;
    //T beamCenterDistance = centerDir.Dot(direction);
    //T beamNormalLengthSqr = (centerDir.Dot(centerDir) -
    //                         beamCenterDistance * beamCenterDistance);
    //T beamHalfLengthSqr = sphereRadius * sphereRadius - beamNormalLengthSqr;
    //if(beamHalfLengthSqr > 0)
    //{
    //    // Inside Square
    //    T beamHalfLength = sqrt(beamHalfLengthSqr);
    //    T t0 = beamCenterDistance - beamHalfLength;
    //    T t1 = beamCenterDistance + beamHalfLength;

    //    t = (fabs(t0) <= fabs(t1)) ? t0 : t1;
    //    intersectPos = position + t * direction;
    //    return true;
    //}
    //return false;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool RayT<T>::IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                           const Vector<3, T> triCorners[3],
                                           bool cullFace) const
{
    return IntersectsTriangle(baryCoords, t,
                              triCorners[0],
                              triCorners[1],
                              triCorners[2],
                              cullFace);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool RayT<T>::IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                           const Vector<3, T>& t0,
                                           const Vector<3, T>& t1,
                                           const Vector<3, T>& t2,
                                           bool cullFace) const
{
    using namespace MathConstants;
    #ifndef MRAY_DEVICE_CODE_PATH
        using namespace std;
    #endif

    // Moller-Trumbore
    // Ray-Tri Intersection
    Vector<3, T> e0 = t1 - t0;
    Vector<3, T> e1 = t2 - t0;
    Vector<3, T> p = Vector3::Cross(direction, e1);
    T det = e0.Dot(p);

    if((cullFace && (det < SmallEpsilon<T>())) ||
       // Ray-Tri nearly parallel skip
       (abs(det) < SmallEpsilon<T>()))
        return false;

    T invDet = 1 / det;

    Vector<3, T> tVec = position - t0;
    baryCoords[0] = tVec.Dot(p) * invDet;
    // Early Skip
    if(baryCoords[0] < 0 || baryCoords[0] > 1)
        return false;

    Vector<3, T> qVec = Vector3::Cross(tVec, e0);
    baryCoords[1] = direction.Dot(qVec) * invDet;
    // Early Skip 2
    if((baryCoords[1] < 0) || (baryCoords[1] + baryCoords[0]) > 1)
        return false;

    t = e1.Dot(qVec) * invDet;
    if(t <= SmallEpsilon<T>())
        return false;

    // Calculate C
    baryCoords[2] = 1 - baryCoords[0] - baryCoords[1];
    baryCoords = Vector<3, T>(baryCoords[2],
                              baryCoords[0],
                              baryCoords[1]);
    return true;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool RayT<T>::IntersectsPlane(Vector<3, T>& intersectPos, T& t,
                                        const Vector<3, T>& planePos,
                                        const Vector<3, T>& normal)
{
    using namespace MathConstants;

    T nDotD = normal.Dot(direction);
    // Nearly parallel
    if(abs(nDotD) <= Epsilon<T>)
    {
        t = std::numeric_limits<T>::infinity();
        return false;
    }
    t = (planePos - position).Dot(normal) / nDotD;
    intersectPos = position + t * direction;
    return true;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool RayT<T>::IntersectsAABB(Vector<2, T>& tOut,
                                       const Vector<3, T>& aabbMin,
                                       const Vector<3, T>& aabbMax,
                                       const Vector<2, T>& tMinMax) const
{
    // CPU code max/min is on std namespace but CUDA has its global namespace
    #ifndef MRAY_DEVICE_CODE_PATH
    using namespace std;
    #endif

    Vector<3, T> invD = Vector<3, T>(1) / direction;
    Vector<3, T> t0 = (aabbMin - position) * invD;
    Vector<3, T> t1 = (aabbMax - position) * invD;
    tOut = tMinMax;

    UNROLL_LOOP
    for(unsigned int i = 0; i < 3; i++)
    {
        if(invD[i] < 0) std::swap(t0[i], t1[i]);

        tOut[0] = max(tOut[0], min(t0[i], t1[i]));
        tOut[1] = min(tOut[1], max(t0[i], t1[i]));
    }
    return tOut[1] >= tOut[0];
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool RayT<T>::IntersectsAABB(const Vector<3, T>& aabbMin,
                                       const Vector<3, T>& aabbMax,
                                       const Vector<2, T>& tMinMax) const
{
    Vector<2, T> tOut;
    return IntersectsAABB(tOut, aabbMin, aabbMax,
                          tMinMax);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool RayT<T>::IntersectsAABB(Vector<3, T>& pos, T& tOut,
                                       const Vector<3, T>& aabbMin,
                                       const Vector<3, T>& aabbMax,
                                       const Vector<2, T>& tMinMax) const
{
    Vector<2, T> t;
    bool intersects = IntersectsAABB(t, aabbMin, aabbMax, tMinMax);
    if(intersects)
    {
        tOut = t[0];
        pos = AdvancedPos(t[0]);
    }
    return intersects;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr RayT<T> RayT<T>::NormalizeDir() const
{
    return Ray(direction.Normalize(), position);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr RayT<T>& RayT<T>::NormalizeDirSelf()
{
    direction.NormalizeSelf();
    return *this;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr RayT<T> RayT<T>::Advance(T t) const
{
    return Ray(direction, position + t * direction);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr RayT<T> RayT<T>::Advance(T t, const Vector<3, T>& dir) const
{
    return Ray(direction, position + t * dir);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr RayT<T>& RayT<T>::AdvanceSelf(T t)
{
    position += t * direction;
    return *this;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr RayT<T>& RayT<T>::AdvanceSelf(T t, const Vector<3, T>& dir)
{
    position += t * dir;
    return *this;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<3, T> RayT<T>::AdvancedPos(T t) const
{
    return position + t * direction;
}

template<>
MRAY_HYBRID MRAY_CGPU_INLINE
RayT<float> RayT<float>::Nudge(const Vector3f& dir) const
{
    // From RayTracing Gems I
    // Chapter 6
    static constexpr float ORIGIN = 1.0f / 32.0f;
    static constexpr float FLOAT_SCALE = 1.0f / 65536.0f;
    static constexpr float INT_SCALE = 256.0f;

    const Vector3f& p = position;
    Vector3i ofi = Vector3i(INT_SCALE * dir[0],
                            INT_SCALE * dir[1],
                            INT_SCALE * dir[2]);

    Vector3f pointI;
    #ifdef MRAY_DEVICE_CODE_PATH_CUDA
        pointI[0] = __int_as_float(__float_as_int(p[0]) + ((p[0] < 0) ? -ofi[0] : ofi[0]));
        pointI[1] = __int_as_float(__float_as_int(p[1]) + ((p[1] < 0) ? -ofi[1] : ofi[1]));
        pointI[2] = __int_as_float(__float_as_int(p[2]) + ((p[2] < 0) ? -ofi[2] : ofi[2]));
    #else
        // CPU code (this will be optimized out)
        // and it is not UB
        static_assert(sizeof(int32_t) == sizeof(float));
        Vector3i pInt;
        std::memcpy(&(pInt[0]), &(p[0]), sizeof(float));
        std::memcpy(&(pInt[1]), &(p[1]), sizeof(float));
        std::memcpy(&(pInt[2]), &(p[2]), sizeof(float));

        pInt[0] += ((p[0] < 0) ? -ofi[0] : ofi[0]);
        pInt[1] += ((p[1] < 0) ? -ofi[1] : ofi[1]);
        pInt[2] += ((p[2] < 0) ? -ofi[2] : ofi[2]);

        std::memcpy(&(pointI[0]), &(pInt[0]), sizeof(float));
        std::memcpy(&(pointI[1]), &(pInt[1]), sizeof(float));
        std::memcpy(&(pointI[2]), &(pInt[2]), sizeof(float));
    #endif

    // Find the next floating point towards
    // Either use an epsilon (float_scale in this case)
    // or use the calculated offset
    Vector3f nextPos;
    nextPos[0] = (fabsf(p[0]) < ORIGIN) ? (p[0] + FLOAT_SCALE * dir[0]) : pointI[0];
    nextPos[1] = (fabsf(p[1]) < ORIGIN) ? (p[1] + FLOAT_SCALE * dir[1]) : pointI[1];
    nextPos[2] = (fabsf(p[2]) < ORIGIN) ? (p[2] + FLOAT_SCALE * dir[2]) : pointI[2];

    return RayT(direction, nextPos);
}

template<>
MRAY_HYBRID MRAY_CGPU_INLINE
RayT<double> RayT<double>::Nudge(const Vector3d& dir) const
{
    // From RayTracing Gems I
    // Chapter 6
    static constexpr double ORIGIN = 1.0 / 32.0;
    static constexpr double FLOAT_SCALE = 1.0 / 65536.0;
    static constexpr double INT_SCALE = 256.0;

    const Vector3d& p = position;

    Vector<3, int64_t> ofi(INT_SCALE * dir[0],
                           INT_SCALE * dir[1],
                           INT_SCALE * dir[2]);

    Vector3d pointI;
    // Utilize memcpy here, it is not UBO
    // and should be optimized since it is in register space
    static_assert(sizeof(int64_t) == sizeof(double));
    Vector<3, int64_t> pInt;
    std::memcpy(&(pInt[0]), &(p[0]), sizeof(double));
    std::memcpy(&(pInt[1]), &(p[1]), sizeof(double));
    std::memcpy(&(pInt[2]), &(p[2]), sizeof(double));

    pInt[0] += ((p[0] < 0) ? -ofi[0] : ofi[0]);
    pInt[1] += ((p[1] < 0) ? -ofi[1] : ofi[1]);
    pInt[2] += ((p[2] < 0) ? -ofi[2] : ofi[2]);

    std::memcpy(&(pointI[0]), &(pInt[0]), sizeof(double));
    std::memcpy(&(pointI[1]), &(pInt[1]), sizeof(double));
    std::memcpy(&(pointI[2]), &(pInt[2]), sizeof(double));

    // Find the next floating point towards
    // Either use an epsilon (float_scale in this case)
    // or use the calculated offset
    Vector3d nextPos;
    nextPos[0] = (fabs(p[0]) < ORIGIN) ? (p[0] + FLOAT_SCALE * dir[0]) : pointI[0];
    nextPos[1] = (fabs(p[1]) < ORIGIN) ? (p[1] + FLOAT_SCALE * dir[1]) : pointI[1];
    nextPos[2] = (fabs(p[2]) < ORIGIN) ? (p[2] + FLOAT_SCALE * dir[2]) : pointI[2];

    return RayT(direction, nextPos);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
RayT<T>& RayT<T>::NudgeSelf(const Vector<3, T>& dir)
{
    RayT<T> r = Nudge(dir);
    (*this) = r;
    return *this;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const Vector<3, T>& RayT<T>::Dir() const
{
    return direction;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const Vector<3, T>& RayT<T>::Pos() const
{
    return position;
}
