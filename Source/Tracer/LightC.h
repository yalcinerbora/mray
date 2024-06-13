#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"

#include "TracerTypes.h"
#include "ParamVaryingData.h"
#include "TransformsDefault.h"
#include "GenericGroup.h"

using PrimBatchList = std::vector<PrimBatchKey>;

template<class LightType>
concept LightC = requires(LightType l,
                          typename LightType::SpectrumConverter sc,
                          typename LightType::Primitive prim,
                          typename LightType::Primitive::Hit hit,
                          typename LightType::DataSoA data,
                          RNGDispenser rng)
{
    typename LightType::SpectrumConverter;
    typename LightType::Primitive;
    typename LightType::DataSoA;

    // Constructor Type
    LightType(sc, prim, data, LightKey{});

    // API
    {l.SampleSolidAngle(rng, Vector3{})} -> std::same_as<SampleT<Vector3>>;
    {l.PdfSolidAngle(hit, Vector3{}, Vector3{})} -> std::same_as<Float>;
    {l.SampleSolidAngleRNCount()} -> std::same_as<uint32_t>;
    {l.SampleRay(rng)} -> std::same_as<SampleT<Ray>>;
    {l.PdfRay(Ray{})} -> std::same_as<Float>;
    {l.SampleRayRNCount()} -> std::same_as<uint32_t>;
    {l.EmitViaHit(Vector3{}, hit)} -> std::same_as<Spectrum>;
    {l.EmitViaSurfacePoint(Vector3{}, Vector3{})} -> std::same_as<Spectrum>;
    {l.IsPrimitiveBackedLight()} -> std::same_as<bool>;

    // Type traits
    requires std::is_trivially_copyable_v<LightType>;
    requires std::is_trivially_destructible_v<LightType>;
    requires std::is_move_assignable_v<LightType>;
    requires std::is_move_constructible_v<LightType>;
};

template<class LGType>
concept LightGroupC = requires(LGType lg)
{
    // Every light mandatorily backed by a primitive
    // Light group holds the type of the backed primitive group
    typename LGType::PrimGroup;
    typename LGType:: template Primitive<>;
    // Internal Light type that satisfies its concept
    requires LightC<typename LGType::template Light<>>;

    // SoA fashion light data. This will be used to access internal
    // of the light with a given an index
    typename LGType::DataSoA;
    requires std::is_same_v<typename LGType::DataSoA,
                            typename LGType::template Light<>::DataSoA>;

    // Acquire SoA struct of this primitive group
    {lg.SoA()} -> std::same_as<typename LGType::DataSoA>;

    // Runtime Acquire the primitive group
    {lg.PrimitiveGroup()} -> std::same_as<const typename LGType::PrimGroup&>;

    // TODO: This concept requres "Reserve" function to be visible,
    // however we switched it so...
    //requires GenericGroupC<LGType>;
};

template <class Converter>
concept CoordConverterC = requires()
{
    { Converter::DirToUV(Vector3{}) } -> std::same_as<Vector2>;
    { Converter::UVToDir(Vector2{}) } -> std::same_as<Vector3>;
    { Converter::ToSolidAnglePdf(Float{}, Vector2{}) } -> std::same_as<Float>;
    { Converter::ToSolidAnglePdf(Float{}, Vector3{}) } -> std::same_as<Float>;
};

class GenericGroupLightT : public GenericTexturedGroupT<LightKey, LightAttributeInfo>
{
    using Parent = GenericTexturedGroupT<LightKey, LightAttributeInfo>;
    using typename Parent::IdList;

    protected:
    Map<LightKey, PrimBatchKey>  primMappings;

    public:
    // Constructors & Destructor
                    GenericGroupLightT(uint32_t groupId, const GPUSystem&,
                                       const TextureViewMap&,
                                       size_t allocationGranularity = 2_MiB,
                                       size_t initialReservartionSize = 4_MiB);
    // Swap the interfaces (old switcharoo)
    private:
    IdList          Reserve(const std::vector<AttributeCountList>&) override;

    public:
    virtual IdList  Reserve(const std::vector<AttributeCountList>&,
                            const PrimBatchList&);
    //
    virtual bool                            IsPrimitiveBacked() const = 0;
    virtual const GenericGroupPrimitiveT&   GenericPrimGroup() const = 0;
    PrimBatchKey                            LightPrimBatch(LightKey) const;
};

using LightGroupPtr = std::unique_ptr<GenericGroupLightT>;

template <class Child>
class GenericGroupLight : public GenericGroupLightT
{
    public:
                        GenericGroupLight(uint32_t groupId, const GPUSystem&,
                                          const TextureViewMap&,
                                          size_t allocationGranularity = 2_MiB,
                                          size_t initialReservartionSize = 4_MiB);
    std::string_view    Name() const override;

};

inline
GenericGroupLightT::GenericGroupLightT(uint32_t groupId, const GPUSystem& s,
                                       const TextureViewMap& map,
                                       size_t allocationGranularity,
                                       size_t initialReservartionSize)
    : Parent(groupId, s, map,
             allocationGranularity,
             initialReservartionSize)
{}

inline typename GenericGroupLightT::IdList
GenericGroupLightT::Reserve(const std::vector<AttributeCountList>&)
{
    throw MRayError("{}: Lights cannot be reserved via this function!",
                    Name());
}

inline typename GenericGroupLightT::IdList
GenericGroupLightT::Reserve(const std::vector<AttributeCountList>& countArrayList,
                            const PrimBatchList& primBatches)
{
    // We blocked the virutal chain, but we should be able to use it here
    // We will do the same here anyways migh as well use it.
    auto result = Parent::Reserve(countArrayList);
    if(!IsPrimitiveBacked()) return result;

    // Relock the mutex (protect the "primMapping" DS)
    std::lock_guard lock{mutex};
    assert(result.size() == primBatches.size());
    for(size_t i = 0; i < primBatches.size(); i++)
        primMappings.try_emplace(result[i], primBatches[i]);
    return result;
}

inline PrimBatchKey GenericGroupLightT::LightPrimBatch(LightKey lKey) const
{
    auto pBatchId = primMappings.at(lKey);
    if(!pBatchId)
    {
        throw MRayError("{:s}:{:d}: Unkown light key {}",
                        this->Name(), this->groupId,
                        lKey.FetchIndexPortion());
    }
    return pBatchId.value();
}

template <class C>
GenericGroupLight<C>::GenericGroupLight(uint32_t groupId, const GPUSystem& sys,
                                        const TextureViewMap& map,
                                        size_t allocationGranularity,
                                        size_t initialReservartionSize)
    : GenericGroupLightT(groupId, sys, map,
                         allocationGranularity,
                         initialReservartionSize)
{}

template <class C>
std::string_view GenericGroupLight<C>::Name() const
{
    return C::TypeName();
}