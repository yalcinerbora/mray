#pragma once

#include "Core/Definitions.h"
#include <Core/Types.h>

#include "TracerTypes.h"
#include "ParamVaryingData.h"
#include "Transforms.h"
#include "GenericGroup.h"

#include "Device/GPUSystem.hpp"

using PrimBatchList = const std::vector<PrimBatchKey>;

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
    std::is_same_v<typename LGType::DataSoA,
                   typename LGType::template Light<>::DataSoA>;

    // Acquire SoA struct of this primitive group
    {lg.SoA()} -> std::same_as<typename LGType::DataSoA>;

    // Runtime Acquire the primitive group
    {lg.PrimitiveGroup()} -> std::same_as<const typename LGType::PrimGroup&>;

    requires GenericGroupC<LGType>;
};

template<class Child>
class GenericLightGroup : public GenericTexturedGroupT<Child, LightKey, LightAttributeInfo>
{
    using Parent = GenericTexturedGroupT<Child, LightKey, LightAttributeInfo>;
    using typename Parent::IdList;

    protected:
    virtual void    HandlePrimBatches(const PrimBatchList&) = 0;

    public:
    // Constructors & Destructor
                    GenericLightGroup(uint32_t groupId, const GPUSystem&,
                                      const TextureView2DMap&,
                                      size_t allocationGranularity = 2_MiB,
                                      size_t initialReservartionSize = 4_MiB);
    // Swap the interfaces (old switcharoo)
    IdList          Reserve(const std::vector<AttributeCountList>&) override;
    virtual IdList  Reserve(const std::vector<AttributeCountList>&,
                            const PrimBatchList&);
};

// Meta Light Class
// This will be used for routines that requires
// every light type at hand. (The example for this is the NEE
// routines that will sample a light source from the lights all over the scene).
// For that case either we would require virtual polymorphism, or a variant
// type/visit. This class utilizes the second one.
// All these functions should decay to "switch case" / "if else"
// statements.
template<class CommonHitT, class MetaLightT,
         class SpectrumTransformer = SpectrumConverterContextIdentity>
class MetaLightViewT
{
    using SpectrumConverter = typename SpectrumTransformer::Converter;
    private:
    const MetaLightT&        light;
    const SpectrumConverter& sConverter;

    public:
    MRAY_HYBRID         MetaLightViewT(const SpectrumConverter& sTransContext,
                                       const MetaLightT&);

    MRAY_HYBRID
    SampleT<Vector3>    SampleSolidAngle(RNGDispenser& dispenser,
                                         const Vector3& distantPoint) const;
    MRAY_HYBRID
    Float               PdfSolidAngle(const CommonHitT& hit,
                                      const Vector3& distantPoint,
                                      const Vector3& dir) const;
    MRAY_HYBRID
    uint32_t            SampleSolidAngleRNCount() const;
    MRAY_HYBRID
    SampleT<Ray>        SampleRay(RNGDispenser& dispenser) const;
    MRAY_HYBRID
    Float               PdfRay(const Ray&) const;
    MRAY_HYBRID
    uint32_t            SampleRayRNCount() const;

    MRAY_HYBRID
    Spectrum            EmitViaHit(const Vector3& wO,
                                   const CommonHitT& hit) const;
    MRAY_HYBRID
    Spectrum            EmitViaSurfacePoint(const Vector3& wO,
                                            const Vector3& surfacePoint) const;

    MRAY_HYBRID bool    IsPrimitiveBackedLight() const;
};

template<class... >
class MetaLightArray;

// Specialize the array
template<TransformContextC... TContexts, LightC... Lights>
class MetaLightArray<Variant<TContexts...>, Variant<Lights...>>
{
    using LightVariant          = Variant<std::monostate, Lights...>;
    using PrimVariant           = UniqueVariant<std::monostate, typename Lights::Primitive...>;
    using TContextVariant       = Variant<std::monostate, TContexts...>;
    using IdentitySConverter    = typename SpectrumConverterContextIdentity::Converter;

    public:
    using MetaLight             = LightVariant;
    template<class CommonHitT, class SpectrumTransformer = SpectrumConverterContextIdentity>
    using MetaLightView         = MetaLightViewT<CommonHitT, MetaLight, SpectrumTransformer>;

    private:
    const GPUSystem&            system;
    Span<PrimVariant>           dLightPrimitiveList;
    Span<LightVariant>          dLightList;
    Span<TContextVariant>       dTContextList;
    Span<IdentitySConverter>    dSConverter;

    DeviceMemory                memory;

    public:
                                MetaLightArray(const GPUSystem&);

    template<LightGroupC LightGroup, TransformGroupC TransformGroup>
    void                        AddBatch(const LightGroup& lg, const TransformGroup& tg,
                                         const Span<const PrimitiveKey>& primitiveKeys,
                                         const Span<const LightKey>& lightKeys,
                                         const Span<const TransformKey>& transformKeys,
                                         const Vector2ui& batchRange);
    Span<const LightVariant>    Array() const;

};

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
MetaLightViewT<CH, ML, SC>::MetaLightViewT(const SpectrumConverter& sConverter,
                                           const ML& l)
    : light(l)
    , sConverter(sConverter)
{}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> MetaLightViewT<CH, ML, SC>::SampleSolidAngle(RNGDispenser& rng,
                                                              const Vector3& distantPoint) const
{
    return DeviceVisit(light, [&](auto&& l) -> Float
    {
        return l.SampleSolidAngle(rng, distantPoint);
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float MetaLightViewT<CH, ML, SC>::PdfSolidAngle(const CH& hit,
                                                const Vector3& distantPoint,
                                                const Vector3& dir) const
{
    return DeviceVisit(light, [=](auto&& l) -> Float
    {
        using HitType = decltype(l)::Primitive::Hit;
        return l.PdfSolidAngle(HitType(hit), distantPoint, dir);
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t MetaLightViewT<CH, ML, SC>::SampleSolidAngleRNCount() const
{
    return DeviceVisit(light, [&](auto&& l) -> Float
    {
        return l.SampleSolidAngleRNCount();
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Ray> MetaLightViewT<CH, ML, SC>::SampleRay(RNGDispenser& rng) const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return l.SampleRay(rng);
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float MetaLightViewT<CH, ML, SC>::PdfRay(const Ray& ray) const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return l.SampleRay(ray);
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t MetaLightViewT<CH, ML, SC>::SampleRayRNCount() const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return l.SampleRayRNCount();
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MetaLightViewT<CH, ML, SC>::EmitViaHit(const Vector3& wO, const CH& hit) const
{
    return DeviceVisit(light, [=](auto&& l) -> Spectrum
    {
        using HitType = decltype(l)::Primitive::Hit;
        return l.Emit(wO, HitType(hit));
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MetaLightViewT<CH, ML, SC>::EmitViaSurfacePoint(const Vector3& wO,
                                                         const Vector3& surfacePoint) const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return specConverter(l.SampleRay(wO, surfacePoint));
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
bool MetaLightViewT<CH, ML, SC>::IsPrimitiveBackedLight() const
{
    return DeviceVisit(light, [&](auto&& l) -> bool
    {
        return IsPrimitiveBackedLight();
    });
}

template<TransformContextC... TC, LightC... L>
MetaLightArray<Variant<TC...>, Variant<L...>>::MetaLightArray(const GPUSystem& s)
    : system(s)
    , memory(system.AllGPUs(), 2_MiB, 16_MiB)
{}

template<TransformContextC... TC, LightC... L>
template<LightGroupC LightGroup, TransformGroupC TransformGroup>
void MetaLightArray<Variant<TC...>, Variant<L...>>::AddBatch(const LightGroup& lg, const TransformGroup& tg,
                                                             const Span<const PrimitiveKey>& primitiveKeys,
                                                             const Span<const LightKey>& lightKeys,
                                                             const Span<const TransformKey>& transformKeys,
                                                             const Vector2ui& batchRange)
{
    const GPUQueue& queue = system.BestDevice().GetQueue(0);

    using TGSoA = typename TransformGroup::DataSoA;
    using LGSoA = typename LightGroup::DataSoA;
    using PGSoA = typename LightGroup::PrimGroup::DataSoA;

    PGSoA pgData = lg.PrimitiveGroup().SoA();
    LGSoA lgData = lg.SoA();
    TGSoA tgData = tg.SoA();

    uint32_t lightCount = (batchRange[1] - batchRange[0]);
    assert(lightKeys.size() == lightCount);

    // Given light construct the transformed light
    // This means having a primitive context
    auto ConstructKernel = [=, this] MRAY_HYBRID(KernelCallParams kp)
    {
        for(uint32_t i = kp.GlobalId(); i < lightCount; i += kp.TotalSize())
        {
            // Determine types, transform context primitive etc.
            // Compile-time find the transform generator function and return type
            using PrimGroup = typename LightGroup::PrimGroup;
            constexpr auto TContextGen = AcquireTransformContextGenerator<PrimGroup, TransformGroup>();
            constexpr auto TGenFunc = decltype(TContextGen)::Function;
            // Define the types
            // First, this kernel uses a transform context
            // that this primitive group provides to generate a surface
            using TContextType = typename decltype(TContextGen)::ReturnType;
            // Assert that this context is either single-transform or identity-transform
            // Currently, each light can only be transformed via
            // single or or identity transform (no skinned meshes :/)
            static_assert(std::is_same_v<TContextType, TransformContextIdentity> ||
                          std::is_same_v<TContextType, TransformContextSingle>);

            // Second, we are using this primitive
            using Primitive = typename PrimGroup:: template Primitive<TContextType>;

            // Light type has to be with identity spectrum conversion
            // meta light will handle the spectrum conversion instead of the light
            using Light = typename LightGroup:: template Light<TContextType>;
            // Check if the light type is in variant list
            static_assert((std::is_same_v<Light, L> || ...),
                          "This light type is not in variant list!");

            // Find the lights starting location
            uint32_t index = batchRange[0] + i;

            // Primitives do not own the transform contexts,
            // save it to global memory.
            dTContextList[index] = TGenFunc(tgData, pgData,
                                            transformKeys[i],
                                            primitiveKeys[i]);
            // Now construct the primitive, it refers to the tc on global memory
            auto& p = dLightPrimitiveList[index];
            p.template emplace<Primitive>(std::get<TContextType>(dTContextList[index]),
                                          pgData, primitiveKeys[i]);

            // And finally construct the light, and this also refers to primitive
            // on the global memory.
            // Construct the lights with identity spectrum transform
            // context since it depends on per-ray data.
            auto& l = dLightList[index];
            l.template emplace<Light>(dSConverter[0],
                                      std::get<Primitive>(dLightPrimitiveList[index]),
                                      lgData, lightKeys[i]);
        }
    };

    using namespace std::literals;
    queue.IssueSaturatingLambda
    (
        "KCConstructMetaLight"sv,
        KernelIssueParams{.workCount = lightCount},
        //
        std::move(ConstructKernel)
    );
}

template<TransformContextC... TC, LightC... L>
Span<const Variant<std::monostate, L...>> MetaLightArray<Variant<TC...>, Variant<L...>>::Array() const
{
    return ToConstSpan(dLightList);
}

template <class C>
GenericLightGroup<C>::GenericLightGroup(uint32_t groupId, const GPUSystem& gpuSystem,
                                        const TextureView2DMap& map,
                                        size_t allocationGranularity,
                                        size_t initialReservartionSize)
    : Parent(groupId, gpuSystem, map,
             allocationGranularity,
             initialReservartionSize)
{}

template <class C>
typename GenericLightGroup<C>::IdList
GenericLightGroup<C>::Reserve(const std::vector<AttributeCountList>&)
{
    throw MRayError("{}: Lights cannot be reserved via this function!",
                    C::TypeName());
}

template <class C>
typename GenericLightGroup<C>::IdList
GenericLightGroup<C>::Reserve(const std::vector<AttributeCountList>& countArrayList,
                              const PrimBatchList& primBatches)
{
    // We blocked the virutal chain, but we should be able to use it here
    // We will do the same here anyways migh as well use it.
    auto result = Parent::Reserve(countArrayList);
    HandlePrimBatches(primBatches);
    return result;
}