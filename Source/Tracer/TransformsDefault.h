#pragma once

#include "Core/MathForward.h"
#include "Core/Types.h"
#include "Core/Matrix.h"
#include "Core/TypeNameGenerators.h"

#include "TransformC.h"
#include "TracerTypes.h"

namespace TransformDetail
{
    struct alignas(32) SingleTransformSoA
    {
        Span<const Matrix4x4> transforms;
        Span<const Matrix4x4> invTransforms;
    };

    struct alignas(32) MultiTransformSoA
    {
        Span<const Span<const Matrix4x4>> transforms;
        Span<const Span<const Matrix4x4>> invTransforms;
    };
}

class TransformContextSingle
{
    private:
    Ref<const Matrix4x4>    transform;
    Ref<const Matrix4x4>    invTransform;

    public:
    MRAY_HYBRID             TransformContextSingle(const typename TransformDetail::SingleTransformSoA&,
                                                   TransformKey tId);

    MRAY_HYBRID Vector3     ApplyP(const Vector3& point) const;
    MRAY_HYBRID Vector3     ApplyV(const Vector3& vec) const;
    MRAY_HYBRID Vector3     ApplyN(const Vector3& normal) const;
    MRAY_HYBRID AABB3       Apply(const AABB3&) const;
    MRAY_HYBRID Ray         Apply(const Ray&) const;
    MRAY_HYBRID Vector3     InvApplyP(const Vector3& point) const;
    MRAY_HYBRID Vector3     InvApplyV(const Vector3& vec) const;
    MRAY_HYBRID Vector3     InvApplyN(const Vector3& normal) const;
    MRAY_HYBRID AABB3       InvApply(const AABB3&) const;
    MRAY_HYBRID Ray         InvApply(const Ray&) const;
};

class TransformGroupSingle final : public GenericGroupTransform<TransformGroupSingle>
{
    public:
    using DataSoA = typename TransformDetail::SingleTransformSoA;
    static std::string_view     TypeName();

    private:
    Span<Matrix4x4> transforms;
    Span<Matrix4x4> invTransforms;
    DataSoA         soa;

    public:
                    TransformGroupSingle(uint32_t groupId, const GPUSystem&);
    void            CommitReservations() override;
    AttribInfoList  AttributeInfo() const override;
    void            PushAttribute(TransformKey batchId,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(TransformKey batchId,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(TransformKey idStart, TransformKey  idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;

    DataSoA         SoA() const;
};

class TransformGroupMulti final : public GenericGroupTransform<TransformGroupMulti>
{
    public:
    using DataSoA = typename TransformDetail::MultiTransformSoA;
    static std::string_view TypeName();

    private:
    Span<Matrix4x4> transforms;
    Span<Matrix4x4> invTransforms;
    DataSoA         soa;

    public:
                    TransformGroupMulti(uint32_t groupId, const GPUSystem& s);
    void            CommitReservations() override;
    AttribInfoList  AttributeInfo() const override;
    void            PushAttribute(TransformKey batchId,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(TransformKey batchId,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(TransformKey idStart, TransformKey  idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;

    DataSoA         SoA() const;
};

// Meta Transform Generator Functions
// (Primitive invariant)
template <class PrimitiveGroupSoA>
MRAY_HYBRID MRAY_CGPU_INLINE
TransformContextSingle GenTContextSingle(const typename TransformGroupSingle::DataSoA& transformData,
                                         const PrimitiveGroupSoA&,
                                         TransformKey tId,
                                         PrimitiveKey)
{
    return TransformContextSingle(transformData, tId);
}

inline std::string_view TransformGroupSingle::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Single"sv;
    return TransformTypeName<Name>;
}

inline std::string_view TransformGroupMulti::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Multi"sv;
    return TransformTypeName<Name>;
}

#include "TransformsDefault.hpp"

static_assert(TransformContextC<TransformContextSingle>);
static_assert(TransformGroupC<TransformGroupSingle>);
static_assert(TransformGroupC<TransformGroupMulti>);