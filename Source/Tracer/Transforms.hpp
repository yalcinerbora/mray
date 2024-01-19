#pragma once

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextIdentity::ApplyP(const Vector3& point) const
{
    return point;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextIdentity::ApplyV(const Vector3& vec) const
{
    return vec;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextIdentity::ApplyN(const Vector3& norm) const
{
    return norm;
}

MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 TransformContextIdentity::Apply(const AABB3& aabb) const
{
    return aabb;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Ray TransformContextIdentity::Apply(const Ray& ray) const
{
    return ray;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextIdentity::InvApplyP(const Vector3& point) const
{
    return point;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextIdentity::InvApplyV(const Vector3& vec) const
{
    return vec;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextIdentity::InvApplyN(const Vector3& norm) const
{
    return norm;
}

MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 TransformContextIdentity::InvApply(const AABB3& aabb) const
{
    return aabb;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Ray TransformContextIdentity::InvApply(const Ray& ray) const
{
    return ray;
}

MRAY_HYBRID MRAY_CGPU_INLINE
TransformContextSingle::TransformContextSingle(const typename TransformDetail::SingleTransformSoA& tData,
                                               TransformId tId)
    : transform(tData.transforms[tId])
    , invTransform(tData.invTransforms[tId])
{}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextSingle::ApplyP(const Vector3& point) const
{
    return Vector3(transform * Vector4(point, 1));
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextSingle::ApplyV(const Vector3& vec) const
{
    return transform * vec;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextSingle::ApplyN(const Vector3& norm) const
{
    return invTransform.LeftMultiply(norm);
}

MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 TransformContextSingle::Apply(const AABB3& aabb) const
{
    return transform.TransformAABB(aabb);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Ray TransformContextSingle::Apply(const Ray& ray) const
{
    return transform.TransformRay(ray);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextSingle::InvApplyP(const Vector3& point) const
{
    return Vector3(invTransform * Vector4(point, 1));
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextSingle::InvApplyV(const Vector3& vec) const
{
    return invTransform * vec;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 TransformContextSingle::InvApplyN(const Vector3& norm) const
{
    return transform.LeftMultiply(norm);
}

MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 TransformContextSingle::InvApply(const AABB3& aabb) const
{
    return invTransform.TransformAABB(aabb);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Ray TransformContextSingle::InvApply(const Ray& ray) const
{
    return invTransform.TransformRay(ray);
}