#pragma once

#include "Core/Types.h"
#include "ParamVaryingData.h"
#include "TransformC.h"
#include "LightC.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

// GPU link errors
#define MRAY_LIGHT_TGEN_FUNCTION(LG, TG) \
    AcquireTransformContextGenerator<typename LG::PrimGroup, TG>()

// Meta Light Class
// This will be used for routines that requires
// every light type at hand. (The example for this is the NEE
// routines that will sample a light source from the lights all over the scene).
// For that case either we would require virtual polymorphism, or a variant
// type/visit. This class utilizes the second one.
// All these functions should decay to "switch case" / "if else"
// statements.
namespace MetaLightDetail
{
    template<LightGroupC LG, TransformGroupC TG>
    using TContextType = typename PrimTransformContextType<typename LG::PrimGroup, TG>::Result;

    template<LightGroupC LG, TransformGroupC TG>
    using LightType = typename LG::template Light<TContextType<LG, TG>>;

    template<LightGroupC LG, TransformGroupC TG>
    using PrimType = typename LightType<LG, TG>::Primitive;

    template<class... Types>
    constexpr Pair<size_t, size_t> MaxSizeAlign()
    {
        constexpr std::array Sizes = {sizeof(Types)...};
        constexpr std::array Alignments = {sizeof(Types)...};
        size_t maxSize = *std::max_element(Sizes.cbegin(), Sizes.cend());
        size_t maxAlign = *std::max_element(Alignments.cbegin(), Alignments.cend());
        return Pair(maxSize, maxAlign);
    }

    template<class... Types>
    constexpr bool AllImplicitLifetime() { return false; }

    template<ImplicitLifetimeC... Types>
    constexpr bool AllImplicitLifetime() { return true; }
}

template<class Variant,
         class SpectrumTransformer = SpectrumConverterContextIdentity>
class MetaLightViewT
{
    using SpectrumConverter = typename SpectrumTransformer::Converter;

    private:
    const SpectrumConverter&    sConverter;
    const Variant&              light;

    public:
    MRAY_HYBRID         MetaLightViewT(const Variant&,
                                       const SpectrumConverter& sTransContext);

    MRAY_HYBRID
    SampleT<Vector3>    SampleSolidAngle(RNGDispenser& dispenser,
                                         const Vector3& distantPoint) const;
    MRAY_HYBRID
    Float               PdfSolidAngle(const MetaHit& hit,
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
                                   const MetaHit& hit) const;
    MRAY_HYBRID
    Spectrum            EmitViaSurfacePoint(const Vector3& wO,
                                            const Vector3& surfacePoint) const;

    MRAY_HYBRID bool    IsPrimitiveBackedLight() const;
};

template<class... TLTuples>
concept LightTransPairC = ((LightGroupC<std::tuple_element_t<0, TLTuples>> && ...) &&
                           (TransformGroupC<std::tuple_element_t<1, TLTuples>> && ...));

template<LightTransPairC... TransformLightTuple>
class MetaLightArrayT
{
    static constexpr size_t GroupCount = sizeof...(TransformLightTuple);

    // Some sanity checks
    static_assert(MetaLightDetail::AllImplicitLifetime
                  <
                      MetaLightDetail::PrimType<std::tuple_element_t<0, TransformLightTuple>,
                                                std::tuple_element_t<1, TransformLightTuple>>
                      ...
                  >(), "Primitive types are not implicit lifetime!");
    static_assert(MetaLightDetail::AllImplicitLifetime
                  <
                      MetaLightDetail::TContextType<std::tuple_element_t<0, TransformLightTuple>,
                                                    std::tuple_element_t<1, TransformLightTuple>>
                      ...
                  >(), "Transform context types are not implicit lifetime!");
    static_assert(MetaLightDetail::AllImplicitLifetime
                  <typename std::tuple_element_t<0, TransformLightTuple>::DataSoA...>(),
                  "Light SoA types are not implicit lifetime!");
    static_assert(MetaLightDetail::AllImplicitLifetime
                  <typename std::tuple_element_t<1, TransformLightTuple>::DataSoA...>(),
                  "Transform SoA types are not implicit lifetime!");
    static_assert(MetaLightDetail::AllImplicitLifetime
                  <typename std::tuple_element_t<0, TransformLightTuple>::PrimGroup::DataSoA...>(),
                  "Prim SoA types are not implicit lifetime!");
    //
    static constexpr Pair PrimVariantSize = MetaLightDetail::MaxSizeAlign
    <
        MetaLightDetail::PrimType<std::tuple_element_t<0, TransformLightTuple>,
                                  std::tuple_element_t<1, TransformLightTuple>>
        ...
    >();
    static constexpr Pair TContextVariantSize = MetaLightDetail::MaxSizeAlign
    <
        MetaLightDetail::TContextType<std::tuple_element_t<0, TransformLightTuple>,
                                      std::tuple_element_t<1, TransformLightTuple>>
        ...
    >();
    static constexpr Pair LightSoAVariantSize = MetaLightDetail::MaxSizeAlign
    <
        typename std::tuple_element_t<0, TransformLightTuple>::DataSoA
        ...
    >();
    static constexpr Pair PrimSoAVariantSize = MetaLightDetail::MaxSizeAlign
    <
        typename std::tuple_element_t<0, TransformLightTuple>::PrimGroup::DataSoA
        ...
    >();
    static constexpr Pair TransSoAVariantSize = MetaLightDetail::MaxSizeAlign
    <
        typename std::tuple_element_t<1, TransformLightTuple>::DataSoA
        ...
    >();

    using PrimBytePack = std::array<Byte, PrimVariantSize.first>;
    using TContextBytePack = std::array<Byte, TContextVariantSize.first>;
    //
    using PrimSoABytePack = std::array<Byte, PrimSoAVariantSize.first>;
    using LightSoABytePack = std::array<Byte, LightSoAVariantSize.first>;
    using TransformSoABytePack = std::array<Byte, TransSoAVariantSize.first>;
    // More sanity checks
    static_assert(PrimVariantSize.second <= MemAlloc::DefaultSystemAlignment());
    static_assert(TContextVariantSize.second <= MemAlloc::DefaultSystemAlignment());
    static_assert(PrimSoAVariantSize.second <= MemAlloc::DefaultSystemAlignment());
    static_assert(LightSoAVariantSize.second <= MemAlloc::DefaultSystemAlignment());
    static_assert(TransSoAVariantSize.second <= MemAlloc::DefaultSystemAlignment());

    // Actual light variant
    using LightVariant = Variant
    <
        std::monostate,
        MetaLightDetail::LightType<std::tuple_element_t<0, TransformLightTuple>,
                                   std::tuple_element_t<1, TransformLightTuple>>
        ...
    >;

    using VariantLightSoA = Variant<typename std::tuple_element_t<0, TransformLightTuple>::DataSoA...>;
    using VariantPrimSoA = UniqueVariant<typename std::tuple_element_t<0, TransformLightTuple>::PrimGroup::DataSoA...>;
    using VariantTransformSoA = UniqueVariant<typename std::tuple_element_t<1, TransformLightTuple>::DataSoA...>;
    //
    using IdentitySConverter = typename SpectrumConverterContextIdentity::Converter;
    //
    using TLGroupPtrTuple = Tuple<TransformLightTuple*...>;
    // We will memcpy the SoA's these must be implicit lifetime types.
    // And we pray that std::variant implementation does not break between CPU/GPU.
    static_assert(ImplicitLifetimeC<VariantLightSoA>);
    static_assert(ImplicitLifetimeC<VariantPrimSoA>);
    static_assert(ImplicitLifetimeC<VariantTransformSoA>);
    // Other variants will be constructed on GPU so inter CGPU will not be an issue.

    public:
    using MetaLight = LightVariant;

    class View
    {
        public:
        template <class SpectrumTransformer>
        using MetaLightView = MetaLightViewT<MetaLight, SpectrumTransformer>;

        private:
        Span<const MetaLight> dMetaLights;

        public:
        View(Span<const MetaLight> d);

        template<class SpectrumTransformer>
        MRAY_HYBRID
        MetaLightView<SpectrumTransformer>
        operator()(const typename SpectrumTransformer::Converter&, uint32_t index) const;

        MRAY_HYBRID
        uint32_t Size() const;
    };

    template <class SpectrumTransformer>
    using MetaLightView = View::MetaLightView<SpectrumTransformer>;

    private:
    const GPUSystem&    system;
    // All the stuff is in variants
    // These are per group
    Span<IdentitySConverter> dSpectrumConverter;
    Span<PrimSoABytePack> dPrimSoA;
    Span<LightSoABytePack> dLightSoA;
    Span<TransformSoABytePack> dTransSoA;
    // These are per-prim
    Span<PrimBytePack> dMetaPrims;
    Span<TContextBytePack> dMetaTContexts;
    // This is the actual variant that refers to all other things
    Span<LightVariant> dMetaLights;

    DeviceMemory        memory;

    uint32_t soaCounter = 0;
    uint32_t lightCounter = 0;

    public:
            MetaLightArrayT(const GPUSystem&);

    template<LightGroupC LightGroup, TransformGroupC TransformGroup>
    void    AddBatch(const LightGroup& lg, const TransformGroup& tg,
                     const Span<const PrimitiveKey>& primitiveKeys,
                     const Span<const LightKey>& lightKeys,
                     const Span<const TransformKey>& transformKeys,
                     const Vector2ui& lightKeyRange,
                     const GPUQueue& queue);

    void    AddBatchGeneric(const GenericGroupLightT& lg,
                            const GenericGroupTransformT& tg,
                            const Span<const PrimitiveKey>& primitiveKeys,
                            const Span<const LightKey>& lightKeys,
                            const Span<const TransformKey>& transformKeys,
                            const Vector2ui& lightKeyRange,
                            const GPUQueue& queue);

    View    Array() const;

};

#include "MetaLight.hpp"
