#pragma once

#include "Core/MathForward.h"
#include "Core/Types.h"
#include "Core/Matrix.h"

#include "TransformC.h"
#include "TracerTypes.h"

namespace TransformDetail
{
    struct SingleTransformSoA
    {
        Span<const Matrix4x4> transforms;
        Span<const Matrix4x4> invTransforms;
    };

    struct MultiTransformSoA
    {
        // TODO: Add static inner span maybe?
        Span<Span<const Matrix4x4>> transforms;
        Span<Span<const Matrix4x4>> invTransforms;
    };

}

// Most simple transform context
class IdentityTransformContext
{
    public:
    MRAY_HYBRID Vector3 ApplyP(const Vector3& point) const;
    MRAY_HYBRID Vector3 ApplyV(const Vector3& vec) const;
    MRAY_HYBRID Vector3 ApplyN(const Vector3& norm) const;
    MRAY_HYBRID AABB3   Apply(const AABB3& aabb) const;
    MRAY_HYBRID Ray     Apply(const Ray& ray) const;
    MRAY_HYBRID Vector3 InvApplyP(const Vector3& point) const;
    MRAY_HYBRID Vector3 InvApplyV(const Vector3& vec) const;
    MRAY_HYBRID Vector3 InvApplyN(const Vector3& norm) const;
    MRAY_HYBRID AABB3   InvApply(const AABB3& aabb) const;
    MRAY_HYBRID Ray     InvApply(const Ray& ray) const;

};

class SingleTransformContext
{
    private:
    const Matrix4x4& transform;
    const Matrix4x4& invTransform;

    public:
    MRAY_HYBRID         SingleTransformContext(const typename TransformDetail::SingleTransformSoA&,
                                               TransformId tId);

    MRAY_HYBRID Vector3 ApplyP(const Vector3& point) const;
    MRAY_HYBRID Vector3 ApplyV(const Vector3& vec) const;
    MRAY_HYBRID Vector3 ApplyN(const Vector3& normal) const;
    MRAY_HYBRID AABB3   Apply(const AABB3&) const;
    MRAY_HYBRID Ray     Apply(const Ray&) const;
    MRAY_HYBRID Vector3 InvApplyP(const Vector3& point) const;
    MRAY_HYBRID Vector3 InvApplyV(const Vector3& vec) const;
    MRAY_HYBRID Vector3 InvApplyN(const Vector3& normal) const;
    MRAY_HYBRID AABB3   InvApply(const AABB3&) const;
    MRAY_HYBRID Ray     InvApply(const Ray&) const;
};

class IdentityTransformGroup
{
    public:
    // Everything is implicit no need for type by concept will require it
    using DataSoA = EmptyType;
};

class SingleTransformGroup
{
    public:
    using DataSoA = typename TransformDetail::SingleTransformSoA;

};

class MultiTransformGroup
{
    public:
    using DataSoA = typename TransformDetail::MultiTransformSoA;

};

class MorphTargetGroup
{

};

// Meta Transform Generator Functions
// (Primitive invariant)
// Provided here for consistency
template <class PrimitiveGroupSoA>
MRAY_HYBRID MRAY_CGPU_INLINE
IdentityTransformContext GenTContextIdentity(const typename IdentityTransformGroup::DataSoA&,
                                             const PrimitiveGroupSoA&,
                                             TransformId,
                                             PrimitiveId)
{
    return IdentityTransformContext{};
}

template <class PrimitiveGroupSoA>
MRAY_HYBRID MRAY_CGPU_INLINE
SingleTransformContext GenTContextSingle(const typename SingleTransformGroup::DataSoA& transformData,
                                         const PrimitiveGroupSoA&,
                                         TransformId tId,
                                         PrimitiveId)
{
    return SingleTransformContext(transformData, tId);
}

static_assert(TransformContextC<IdentityTransformContext>);
static_assert(TransformContextC<SingleTransformContext>);

#include "Transforms.hpp"