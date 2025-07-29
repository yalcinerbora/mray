#pragma once

#include "TransformsDefault.h"

MR_HF_DEF
TransformContextSingle::TransformContextSingle(const typename TransformDetail::SingleTransformSoA& tData,
                                               TransformKey tK)
    : transform(tData.transforms[tK.FetchIndexPortion()])
    , invTransform(tData.invTransforms[tK.FetchIndexPortion()])
{}

MR_HF_DEF
Vector3 TransformContextSingle::Scale() const
{
    return TransformGen::ExtractScale(transform.get());
}

MR_HF_DEF
Vector3 TransformContextSingle::ApplyP(const Vector3& point) const
{
    return Vector3(transform.get() * Vector4(point, 1));
}

MR_HF_DEF
Vector3 TransformContextSingle::ApplyV(const Vector3& vec) const
{
    return transform.get() * vec;
}

MR_HF_DEF
Vector3 TransformContextSingle::ApplyN(const Vector3& norm) const
{
    return invTransform.get().LeftMultiply(norm);
}

MR_HF_DEF
AABB3 TransformContextSingle::Apply(const AABB3& aabb) const
{
    return transform.get().TransformAABB(aabb);
}

MR_HF_DEF
Ray TransformContextSingle::Apply(const Ray& ray) const
{
    return transform.get().TransformRay(ray);
}

MR_HF_DEF
Vector3 TransformContextSingle::InvApplyP(const Vector3& point) const
{
    return Vector3(invTransform.get() * Vector4(point, 1));
}

MR_HF_DEF
Vector3 TransformContextSingle::InvApplyV(const Vector3& vec) const
{
    return invTransform.get() * vec;
}

MR_HF_DEF
Vector3 TransformContextSingle::InvApplyN(const Vector3& norm) const
{
    return transform.get().LeftMultiply(norm);
}

MR_HF_DEF
AABB3 TransformContextSingle::InvApply(const AABB3& aabb) const
{
    return invTransform.get().TransformAABB(aabb);
}

MR_HF_DEF
Ray TransformContextSingle::InvApply(const Ray& ray) const
{
    return invTransform.get().TransformRay(ray);
}