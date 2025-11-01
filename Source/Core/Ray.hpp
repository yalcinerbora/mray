#pragma once

#include "Ray.h"

template<FloatC T>
MR_PF_DEF_V RayT<T>::RayT(const Vector<3, T>& direction,
                        const Vector<3, T>& position) noexcept
    : dir(direction)
    , pos(position)
{}

template<FloatC T>
MR_PF_DEF_V RayT<T>::RayT(const Vector<3, T> vec[2]) noexcept
    : dir(vec[0])
    , pos(vec[1])
{}

template<FloatC T>
MR_PF_DEF bool RayT<T>::IntersectsSphere(Vector<3, T>& intersectPos, T& t,
                                         const Vector<3, T>& sphereCenter,
                                         T sphereRadius) const noexcept
{
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
    T dirLengthInv = 1.0f / Math::Length(dir);
    Vector<3, T> dirNorm = dir * dirLengthInv;
    Vector<3, T> centerDir = sphereCenter - pos;
    T beamCenterDist = Math::Dot(dirNorm, centerDir);
    T cDirLengthSqr = Math::LengthSqr(centerDir);

    // Below code is from the source
    Vector<3, T> remedyTerm = centerDir - beamCenterDist * dirNorm;
    T discriminant = sphereRadius * sphereRadius - Math::LengthSqr(remedyTerm);
    if(discriminant >= 0)
    {
        T beamHalfLength = Math::Sqrt(discriminant);

        T t0 = (beamCenterDist >= 0)
            ? (beamCenterDist + beamHalfLength)
            : (beamCenterDist - beamHalfLength);
        T t1 = (cDirLengthSqr - sphereRadius * sphereRadius) / t0;

        // TODO: is there a better way to do this?
        // Select a T
        t = std::numeric_limits<T>::max();
        if(t0 > 0) t = Math::Min(t, t0);
        if(t1 > 0) t = Math::Min(t, t1);
        if(t != std::numeric_limits<T>::max())
        {
            t *= dirLengthInv;
            intersectPos = pos + t * dir;
            return true;
        }
    }
    return false;

    // Geometric solution
    //Vector<3, T> centerDir = sphereCenter - pos;
    //T beamCenterDistance = Math::Dot(centerDir, dir);
    //T beamNormalLengthSqr = (Math::LengthSqr(centerDir) -
    //                         beamCenterDistance * beamCenterDistance);
    //T beamHalfLengthSqr = sphereRadius * sphereRadius - beamNormalLengthSqr;
    //if(beamHalfLengthSqr > 0)
    //{
    //    // Inside Square
    //    T beamHalfLength = Math::Sqrt(beamHalfLengthSqr);
    //    T t0 = beamCenterDistance - beamHalfLength;
    //    T t1 = beamCenterDistance + beamHalfLength;

    //    t = (Math::Abs(t0) <= Math::Abs(t1)) ? t0 : t1;
    //    intersectPos = pos + t * dir;
    //    return true;
    //}
    //return false;
}

template<FloatC T>
MR_PF_DEF bool RayT<T>::IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                           const Vector<3, T> triCorners[3],
                                           bool cullFace) const noexcept
{
    return IntersectsTriangle(baryCoords, t,
                              triCorners[0],
                              triCorners[1],
                              triCorners[2],
                              cullFace);
}

template<FloatC T>
MR_PF_DEF bool RayT<T>::IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                           const Vector<3, T>& t0,
                                           const Vector<3, T>& t1,
                                           const Vector<3, T>& t2,
                                           bool cullFace) const noexcept
{
    using namespace MathConstants;
    // Moller-Trumbore
    // Ray-Tri Intersection
    Vector<3, T> e0 = t1 - t0;
    Vector<3, T> e1 = t2 - t0;
    Vector<3, T> p = Math::Cross(dir, e1);
    T det = Math::Dot(e0, p);

    if((cullFace && (det < SmallEpsilon<T>())) ||
       // Ray-Tri nearly parallel skip
       (Math::Abs(det) < SmallEpsilon<T>()))
        return false;

    T invDet = 1 / det;

    Vector<3, T> tVec = pos - t0;
    baryCoords[0] = Math::Dot(tVec, p) * invDet;
    // Early Skip
    if(baryCoords[0] < 0 || baryCoords[0] > 1)
        return false;

    Vector<3, T> qVec = Math::Cross(tVec, e0);
    baryCoords[1] = Math::Dot(dir, qVec) * invDet;
    // Early Skip 2
    if((baryCoords[1] < 0) || (baryCoords[1] + baryCoords[0]) > 1)
        return false;

    t = Math::Dot(e1, qVec) * invDet;
    if(t <= SmallEpsilon<T>())
        return false;

    // Calculate C
    baryCoords[2] = 1 - baryCoords[0] - baryCoords[1];
    baryCoords = Vector<3, T>(baryCoords[2],
                              baryCoords[0],
                              baryCoords[1]);
    return true;
}

template<FloatC T>
MR_PF_DEF bool RayT<T>::IntersectsPlane(Vector<3, T>& intersectPos, T& t,
                                        const Vector<3, T>& planePos,
                                        const Vector<3, T>& normal) const noexcept
{
    using namespace MathConstants;

    T nDotD = normal.Dot(dir);
    // Nearly parallel
    if(abs(nDotD) <= Epsilon<T>)
    {
        t = std::numeric_limits<T>::infinity();
        return false;
    }
    t = (planePos - pos).Dot(normal) / nDotD;
    intersectPos = pos + t * dir;
    return true;
}

template<FloatC T>
MR_PF_DEF bool RayT<T>::IntersectsAABB(Vector<2, T>& tOut,
                                       const Vector<3, T>& aabbMin,
                                       const Vector<3, T>& aabbMax,
                                       const Vector<2, T>& tMinMax) const noexcept
{
    Vector<3, T> invD = Vector<3, T>(1) / dir;
    Vector<3, T> t0 = (aabbMin - pos) * invD;
    Vector<3, T> t1 = (aabbMax - pos) * invD;
    tOut = tMinMax;

    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < 3; i++)
    {
        if(invD[i] < 0) std::swap(t0[i], t1[i]);

        tOut[0] = Math::Max(tOut[0], Math::Min(t0[i], t1[i]));
        tOut[1] = Math::Min(tOut[1], Math::Max(t0[i], t1[i]));
    }
    return tOut[1] >= tOut[0];
}

template<FloatC T>
MR_PF_DEF bool RayT<T>::IntersectsAABB(const Vector<3, T>& aabbMin,
                                       const Vector<3, T>& aabbMax,
                                       const Vector<2, T>& tMinMax) const noexcept
{
    Vector<2, T> tOut;
    return IntersectsAABB(tOut, aabbMin, aabbMax,
                          tMinMax);
}

template<FloatC T>
MR_PF_DEF bool RayT<T>::IntersectsAABB(Vector<3, T>& pos, T& tOut,
                                       const Vector<3, T>& aabbMin,
                                       const Vector<3, T>& aabbMax,
                                       const Vector<2, T>& tMinMax) const noexcept
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

template<FloatC T>
MR_PF_DEF Vector<3, T> RayT<T>::AdvancedPos(T t) const noexcept
{
    return pos + t * dir;
}

template<FloatC T>
MR_PF_DEF RayT<T> RayT<T>::Nudge(const Vector<3, T>& nudgeDir) const noexcept
{
    using Int = IntegralSister<T>;
    using Vec3Int = Vector<3, Int>;
    using Vec3Float = Vector<3, T>;

    // From RayTracing Gems I
    // Chapter 6
    constexpr T ORIGIN = T(1) / T(32);
    constexpr T FLOAT_SCALE = T(1) / T(65536);
    constexpr T INT_SCALE = T(256);

    const Vec3Float& p = pos;
    Vec3Int ofi = Vec3Int(INT_SCALE * nudgeDir[0],
                          INT_SCALE * nudgeDir[1],
                          INT_SCALE * nudgeDir[2]);

    Vec3Float pointI;
    Vec3Int pInt;
    pInt[0] = Bit::BitCast<Int>(p[0]);
    pInt[1] = Bit::BitCast<Int>(p[1]);
    pInt[2] = Bit::BitCast<Int>(p[2]);

    pInt[0] += ((p[0] < T(0)) ? -ofi[0] : ofi[0]);
    pInt[1] += ((p[1] < T(0)) ? -ofi[1] : ofi[1]);
    pInt[2] += ((p[2] < T(0)) ? -ofi[2] : ofi[2]);

    pointI[0] = Bit::BitCast<T>(pInt[0]);
    pointI[1] = Bit::BitCast<T>(pInt[1]);
    pointI[2] = Bit::BitCast<T>(pInt[2]);

    // Find the next floating point towards
    // Either use an epsilon (float_scale in this case)
    // or use the calculated offset
    using Math::Abs;
    Vec3Float nextPos;
    nextPos[0] = (Abs(p[0]) < ORIGIN) ? (p[0] + FLOAT_SCALE * nudgeDir[0]) : pointI[0];
    nextPos[1] = (Abs(p[1]) < ORIGIN) ? (p[1] + FLOAT_SCALE * nudgeDir[1]) : pointI[1];
    nextPos[2] = (Abs(p[2]) < ORIGIN) ? (p[2] + FLOAT_SCALE * nudgeDir[2]) : pointI[2];

    return RayT(dir, nextPos);
}