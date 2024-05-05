#pragma once

#include <concepts>
#include <vector>

#include "Core/MathForward.h"
#include "Core/TracerI.h"

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