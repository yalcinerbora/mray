#pragma once

#include <concepts>
#include <vector>

#include "Core/MathForward.h"
#include "Core/TracerI.h"
#include "Core/TypeNameGenerators.h"

#include "GenericGroup.h"

template <class TContext>
concept TransformContextC = requires(const TContext& t,
                                     const Vector3& v,
                                     const AABB3& aabb,
                                     const Ray& r)
{
    {t.ApplyP(v)} -> std::same_as<Vector3>;
    {t.ApplyV(v)} -> std::same_as<Vector3>;
    {t.ApplyN(v)} -> std::same_as<Vector3>;
    {t.Apply(r)} -> std::same_as<Ray>;
    {t.Apply(aabb)} -> std::same_as<AABB3>;

    {t.InvApplyP(v)} -> std::same_as<Vector3>;
    {t.InvApplyV(v)} -> std::same_as<Vector3>;
    {t.InvApplyN(v)} -> std::same_as<Vector3>;
    {t.InvApply(r)} -> std::same_as<Ray>;
    {t.InvApply(aabb)} -> std::same_as<AABB3>;

    // Type traits
    requires std::is_trivially_copyable_v<TContext>;
    requires std::is_trivially_destructible_v<TContext>;
};

template <class TGType>
concept TransformGroupC = requires(TGType tg)
{
    typename TGType::DataSoA;
    // Can request DataSoA
    {tg.SoA()}-> std::same_as<typename TGType::DataSoA>;

    requires GenericGroupC<TGType>;
};

using GenericGroupTransformT    = GenericGroupT<TransformKey, TransAttributeInfo>;
using TransformGroupPtr         = std::unique_ptr<GenericGroupTransformT>;

template<class Child>
class GenericGroupTransform : public GenericGroupTransformT
{
    public:
                        GenericGroupTransform(uint32_t transGroupId,
                                              const GPUSystem& sys,
                                              size_t allocationGranularity = 2_MiB,
                                              size_t initialReservartionSize = 4_MiB);
    std::string_view    Name() const override;
};

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

class TransformGroupIdentity final : public GenericGroupTransform<TransformGroupIdentity>
{
    public:
    using DataSoA = EmptyType;
    static std::string_view TypeName();

    public:
                    TransformGroupIdentity(uint32_t groupId, const GPUSystem&);
    void            CommitReservations() override;
    AttribInfoList  AttributeInfo() const override;
    void            PushAttribute(TransformKey batchId,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(TransformKey batchId,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subBatchRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(TransformKey idStart, TransformKey idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;

    DataSoA         SoA() const;
};

template <class C>
GenericGroupTransform<C>::GenericGroupTransform(uint32_t transGroupId,
                                                const GPUSystem& sys,
                                                size_t allocationGranularity,
                                                size_t initialReservartionSize)
    : GenericGroupTransformT(transGroupId, sys,
                             allocationGranularity,
                             initialReservartionSize)
{}

template <class C>
std::string_view GenericGroupTransform<C>::Name() const
{
    return C::TypeName();
}


static_assert(TransformContextC<TransformContextIdentity>);
static_assert(TransformGroupC<TransformGroupIdentity>);

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

// Meta Transform Generator Functions
template <class PrimitiveGroupSoA>
MRAY_HYBRID MRAY_CGPU_INLINE
TransformContextIdentity GenTContextIdentity(const typename TransformGroupIdentity::DataSoA&,
                                             const PrimitiveGroupSoA&,
                                             TransformKey,
                                             PrimitiveKey)
{
    return TransformContextIdentity{};
}

inline std::string_view TransformGroupIdentity::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Identity"sv;
    return TransformTypeName<Name>;
}

inline TransformGroupIdentity::TransformGroupIdentity(uint32_t groupId,
                                                      const GPUSystem& s)
    : GenericGroupTransform<TransformGroupIdentity>(groupId, s)
{}

inline void TransformGroupIdentity::CommitReservations()
{
    isCommitted = true;
}

inline void TransformGroupIdentity::PushAttribute(TransformKey, uint32_t,
                                                  TransientData, const GPUQueue&)
{}

inline void TransformGroupIdentity::PushAttribute(TransformKey,
                                                  uint32_t,
                                                  const Vector2ui&,
                                                  TransientData,
                                                  const GPUQueue&)
{}

inline void TransformGroupIdentity::PushAttribute(TransformKey, TransformKey,
                                                  uint32_t, TransientData,
                                                  const GPUQueue&)
{}

inline TransformGroupIdentity::AttribInfoList TransformGroupIdentity::AttributeInfo() const
{
    return AttribInfoList();
}