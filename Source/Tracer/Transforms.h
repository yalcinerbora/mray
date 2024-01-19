#pragma once

#include "Core/MathForward.h"
#include "Core/Types.h"
#include "Core/Matrix.h"

#include "TransformC.h"
#include "TracerTypes.h"
#include "TracerInterface.h"

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

class TransformContextIdentity
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

class TransformContextSingle
{
    private:
    const Matrix4x4& transform;
    const Matrix4x4& invTransform;

    public:
    MRAY_HYBRID         TransformContextSingle(const typename TransformDetail::SingleTransformSoA&,
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

template<class Child>
using GenericGroupTransform = GenericGroup<Child, TransformId, TransAttributeInfo>;

class TransformGroupIdentity final : public GenericGroupTransform<TransformGroupIdentity>
{
    using typename GenericGroupTransform<TransformGroupIdentity>::AttribInfoList;

    public:
    using DataSoA = EmptyType;
    static std::string_view TypeName();

    public:
                    TransformGroupIdentity(uint32_t groupId,
                                           const GPUSystem& s);
    virtual void    Commit() override;
    virtual void    PushAttribute(Vector2ui idRange,
                                  uint32_t attributeIndex,
                                  std::vector<Byte> data) override;
    AttribInfoList  AttributeInfo() const override;
};

class TransformGroupSingle final : public GenericGroupTransform<TransformGroupSingle>
{
    using typename GenericGroupTransform<TransformGroupSingle>::AttribInfoList;

    public:
    using DataSoA = typename TransformDetail::SingleTransformSoA;
    static std::string_view     TypeName();

    private:
    Span<Matrix4x4> transforms;
    Span<Matrix4x4> invTransforms;
    DataSoA         soa;

    public:
                    TransformGroupSingle(uint32_t groupId, const GPUSystem& s);
    void            Commit() override;
    void            PushAttribute(Vector2ui idRange,
                                  uint32_t attributeIndex,
                                  std::vector<Byte> data) override;
    AttribInfoList  AttributeInfo() const override;
};

class TransformGroupMulti final : public GenericGroupTransform<TransformGroupMulti>
{
    using typename GenericGroupTransform<TransformGroupMulti>::AttribInfoList;

    public:
    using DataSoA = typename TransformDetail::MultiTransformSoA;
    static std::string_view TypeName();

    private:
    Span<Matrix4x4> transforms;
    Span<Matrix4x4> invTransforms;
    Span<uint32_t>  indices;
    DataSoA         soa;

    public:
                    TransformGroupMulti(uint32_t groupId, const GPUSystem& s);
    void            Commit() override;
    void            PushAttribute(Vector2ui idRange,
                                  uint32_t attributeIndex,
                                  std::vector<Byte> data) override;
    AttribInfoList  AttributeInfo() const override;
};

// Meta Transform Generator Functions
// (Primitive invariant)
// Provided here for consistency
template <class PrimitiveGroupSoA>
MRAY_HYBRID MRAY_CGPU_INLINE
TransformContextIdentity GenTContextIdentity(const typename TransformGroupIdentity::DataSoA&,
                                             const PrimitiveGroupSoA&,
                                             TransformId,
                                             PrimitiveId)
{
    return TransformContextIdentity{};
}

template <class PrimitiveGroupSoA>
MRAY_HYBRID MRAY_CGPU_INLINE
TransformContextSingle GenTContextSingle(const typename TransformGroupSingle::DataSoA& transformData,
                                         const PrimitiveGroupSoA&,
                                         TransformId tId,
                                         PrimitiveId)
{
    return TransformContextSingle(transformData, tId);
}

static_assert(TransformContextC<TransformContextIdentity>);
static_assert(TransformContextC<TransformContextSingle>);

#include "Transforms.hpp"