#pragma once

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 IdentityTransformContext::ApplyP(const Vector3& point) const
{
    return point;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 IdentityTransformContext::ApplyV(const Vector3& vec) const
{
    return vec;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 IdentityTransformContext::ApplyN(const Vector3& norm) const
{
    return norm;
}

MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 IdentityTransformContext::Apply(const AABB3& aabb) const
{
    return aabb;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Ray IdentityTransformContext::Apply(const Ray& ray) const
{
    return ray;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 IdentityTransformContext::InvApplyP(const Vector3& point) const
{
    return point;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 IdentityTransformContext::InvApplyV(const Vector3& vec) const
{
    return vec;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 IdentityTransformContext::InvApplyN(const Vector3& norm) const
{
    return norm;
}

MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 IdentityTransformContext::InvApply(const AABB3& aabb) const
{
    return aabb;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Ray IdentityTransformContext::InvApply(const Ray& ray) const
{
    return ray;
}

MRAY_HYBRID MRAY_CGPU_INLINE
SingleTransformContext::SingleTransformContext(const typename TransformDetail::SingleTransformSoA& tData,
                                               TransformId tId)
    : transform(tData.transforms[tId])
    , invTransform(tData.invTransforms[tId])
{}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 SingleTransformContext::ApplyP(const Vector3& point) const
{
    return Vector3(transform * Vector4(point, 1));
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 SingleTransformContext::ApplyV(const Vector3& vec) const
{
    return transform * vec;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 SingleTransformContext::ApplyN(const Vector3& norm) const
{
    return invTransform.LeftMultiply(norm);
}

MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 SingleTransformContext::Apply(const AABB3& aabb) const
{
    return transform.TransformAABB(aabb);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Ray SingleTransformContext::Apply(const Ray& ray) const
{
    return transform.TransformRay(ray);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 SingleTransformContext::InvApplyP(const Vector3& point) const
{
    return Vector3(invTransform * Vector4(point, 1));
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 SingleTransformContext::InvApplyV(const Vector3& vec) const
{
    return invTransform * vec;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 SingleTransformContext::InvApplyN(const Vector3& norm) const
{
    return transform.LeftMultiply(norm);
}

MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 SingleTransformContext::InvApply(const AABB3& aabb) const
{
    return invTransform.TransformAABB(aabb);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Ray SingleTransformContext::InvApply(const Ray& ray) const
{
    return invTransform.TransformRay(ray);
}