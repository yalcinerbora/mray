#pragma once

#include "Core/Types.h"
#include "Core/Variant.h"

#include "ParamVaryingData.h"
#include "TransformC.h"
#include "LightC.h"
#include "Random.h"

#include "Device/GPUSystem.h"

// GPU link errors
#define MRAY_LIGHT_TGEN_FUNCTION(LG, TG) \
    AcquireTransformContextGenerator<typename LG::PrimGroup, TG>()

// Similar to the accelerator params, except no surfList
struct LightPartition;

struct MetaLightListConstructionParams
{
    using LightSurfPair = Pair<LightSurfaceId, LightSurfaceParams>;

    const Map<LightGroupId, LightGroupPtr>&     lightGroups;
    const Map<TransGroupId, TransformGroupPtr>& transformGroups;
    Span<const LightSurfPair>                   lSurfList;

    std::vector<LightPartition> Partition() const;
};

struct LightSurfKeyPack
{
    CommonKey lK;
    CommonKey tK;
    CommonKey pK;

    auto operator<=>(const LightSurfKeyPack&) const = default;
};

struct LightSurfKeyHasher
{
    MR_HF_DECL
    static uint32_t Hash(const LightSurfKeyPack&);
    MR_HF_DECL
    static bool     IsSentinel(uint32_t);
    MR_HF_DECL
    static bool     IsEmpty(uint32_t);
};

using LightLookupTable = LookupTable<LightSurfKeyPack, uint32_t, uint32_t, 4,
                                     LightSurfKeyHasher>;

struct LightPartition
{
    using LSurfPair = typename MetaLightListConstructionParams::LightSurfPair;
    LightGroupId lgId;
    std::vector<Pair<TransGroupId, Span<const LSurfPair>>> ltPartitions;
};

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
        return Pair<size_t, size_t>(maxSize, maxAlign);
    }

    template<class... Types>
    constexpr bool AllImplicitLifetime() { return false; }

    template<ImplicitLifetimeC... Types>
    constexpr bool AllImplicitLifetime() { return true; }

    template<LightC... Lights>
    constexpr uint32_t SampleSolidAngleRNCountWorstCase()
    {
        uint32_t result = std::max({Lights::SampleSolidAngleRNCount...});
        return result;
    }

    template<LightC... Lights>
    constexpr uint32_t SampleRayRNCountWorstCase()
    {
        uint32_t result = std::max({Lights::SampleRayRNCount...});
        return result;
    }
}

template<class LightVariant,
         class SpectrumConverterIn = SpectrumConverterIdentity>
class MetaLightViewT
{
    using SpectrumConverter = SpectrumConverterIn;

    private:
    const SpectrumConverter&    sConv;
    const LightVariant&         light;

    public:
    MR_HF_DECL          MetaLightViewT(const LightVariant&,
                                       const SpectrumConverter& sTransContext);

    MR_HF_DECL
    SampleT<Vector3>    SampleSolidAngle(RNGDispenser& dispenser,
                                         const Vector3& distantPoint) const;
    MR_HF_DECL
    Float               PdfSolidAngle(const MetaHit& hit,
                                      const Vector3& distantPoint,
                                      const Vector3& dir) const;
    MR_HF_DECL
    uint32_t            SampleSolidAngleRNCount() const;
    MR_HF_DECL
    SampleT<Ray>        SampleRay(RNGDispenser& dispenser) const;
    MR_HF_DECL
    Float               PdfRay(const Ray&) const;
    MR_HF_DECL
    uint32_t            SampleRayRNCount() const;

    MR_HF_DECL
    Spectrum            EmitViaHit(const Vector3& wO,
                                   const MetaHit& hit,
                                   const RayCone& rayCone) const;
    MR_HF_DECL
    Spectrum            EmitViaSurfacePoint(const Vector3& wO,
                                            const Vector3& surfacePoint,
                                            const RayCone& rayCone) const;

    MR_HF_DECL bool     IsPrimitiveBackedLight() const;
};

template<class... TLTuples>
concept LightTransPairC = ((LightGroupC<TypePackElement<0, TLTuples>> && ...) &&
                           (TransformGroupC<TypePackElement<1, TLTuples>> && ...));

template<LightTransPairC... TransformLightTuple>
class MetaLightArrayT
{
    static constexpr size_t GroupCount = sizeof...(TransformLightTuple);

    // Some sanity checks
    static_assert(MetaLightDetail::AllImplicitLifetime
                  <
                      MetaLightDetail::PrimType<TypePackElement<0, TransformLightTuple>,
                                                TypePackElement<1, TransformLightTuple>>
                      ...
                  >(), "Primitive types are not implicit lifetime!");
    static_assert(MetaLightDetail::AllImplicitLifetime
                  <
                      MetaLightDetail::TContextType<TypePackElement<0, TransformLightTuple>,
                                                    TypePackElement<1, TransformLightTuple>>
                      ...
                  >(), "Transform context types are not implicit lifetime!");
    static_assert(MetaLightDetail::AllImplicitLifetime
                  <typename TypePackElement<0, TransformLightTuple>::DataSoA...>(),
                  "Light SoA types are not implicit lifetime!");
    static_assert(MetaLightDetail::AllImplicitLifetime
                  <typename TypePackElement<1, TransformLightTuple>::DataSoA...>(),
                  "Transform SoA types are not implicit lifetime!");
    static_assert(MetaLightDetail::AllImplicitLifetime
                  <typename TypePackElement<0, TransformLightTuple>::PrimGroup::DataSoA...>(),
                  "Prim SoA types are not implicit lifetime!");
    //
    static constexpr Pair<size_t, size_t> PrimVariantSize = MetaLightDetail::MaxSizeAlign
    <
        MetaLightDetail::PrimType<TypePackElement<0, TransformLightTuple>,
        TypePackElement<1, TransformLightTuple>>
        ...
    >();
    static constexpr Pair<size_t, size_t> TContextVariantSize = MetaLightDetail::MaxSizeAlign
    <
        MetaLightDetail::TContextType<TypePackElement<0, TransformLightTuple>,
                                      TypePackElement<1, TransformLightTuple>>
        ...
    >();
    static constexpr Pair<size_t, size_t> LightSoAVariantSize = MetaLightDetail::MaxSizeAlign
    <
        typename TypePackElement<0, TransformLightTuple>::DataSoA
        ...
    >();
    static constexpr Pair<size_t, size_t> PrimSoAVariantSize = MetaLightDetail::MaxSizeAlign
    <
        typename TypePackElement<0, TransformLightTuple>::PrimGroup::DataSoA
        ...
    >();
    static constexpr Pair<size_t, size_t> TransSoAVariantSize = MetaLightDetail::MaxSizeAlign
    <
        typename TypePackElement<1, TransformLightTuple>::DataSoA
        ...
    >();

    public:
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

    private:
    // Actual light variant
    using LightVariant = Variant
    <
        std::monostate,
        MetaLightDetail::LightType<TypePackElement<0, TransformLightTuple>,
                                   TypePackElement<1, TransformLightTuple>>
        ...
    >;

    using VariantLightSoA = Variant<typename TypePackElement<0, TransformLightTuple>::DataSoA...>;
    using VariantPrimSoA = UniqueVariant<typename TypePackElement<0, TransformLightTuple>::PrimGroup::DataSoA...>;
    using VariantTransformSoA = UniqueVariant<typename TypePackElement<1, TransformLightTuple>::DataSoA...>;
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
    static constexpr uint32_t SampleSolidAngleRNCountWorst = MetaLightDetail::SampleSolidAngleRNCountWorstCase
    <
        MetaLightDetail::LightType<TypePackElement<0, TransformLightTuple>,
                                   TypePackElement<1, TransformLightTuple>>
        ...
    >();

    static constexpr uint32_t SampleRayRNCountWorst = MetaLightDetail::SampleRayRNCountWorstCase
    <
        MetaLightDetail::LightType<TypePackElement<0, TransformLightTuple>,
                                   TypePackElement<1, TransformLightTuple>>
        ...
    >();

    class View
    {
        public:
        template <class SpectrumConverter>
        using MetaLightView = MetaLightViewT<MetaLight, SpectrumConverter>;

        private:
        Span<const MetaLight> dMetaLights;

        public:
        View(Span<const MetaLight> d);

        template<class SpectrumConverter>
        MR_HF_DECL
        MetaLightView<SpectrumConverter>
        operator()(const SpectrumConverter&, uint32_t index) const;

        MR_HF_DECL
        uint32_t Size() const;
    };

    private:
    const GPUSystem&    system;
    // All the stuff is in variants
    // These are per group
    Span<IdentitySConverter>    dSpectrumConverter;
    Span<PrimSoABytePack>       dPrimSoA;
    Span<LightSoABytePack>      dLightSoA;
    Span<TransformSoABytePack>  dTransSoA;
    // These are per-prim
    Span<PrimBytePack>      dMetaPrims;
    Span<TContextBytePack>  dMetaTContexts;
    Span<LightVariant>      dMetaLights;
    // These are related to the lookup table
    Span<Vector4ui>         dTableHashes;
    Span<LightSurfKeyPack>  dTableKeys;
    Span<uint32_t>          dTableValues;

    DeviceMemory    memory;

    uint32_t soaCounter = 0;
    uint32_t lightCounter = 0;

    public:
    // Constructors & Destructor
            MetaLightArrayT(const GPUSystem&);


    // We can't make these private/protected due to GPU Lambdas.
    template<LightGroupC LightGroup, TransformGroupC TransformGroup>
    void    AddBatch(const LightGroup& lg, const TransformGroup& tg,
                     const Span<const PrimitiveKey>& primitiveKeys,
                     const Span<const LightKey>& lightKeys,
                     const Span<const TransformKey>& transformKeys,
                     const GPUQueue& queue);

    void    AddBatchGeneric(const GenericGroupLightT& lg,
                            const GenericGroupTransformT& tg,
                            const Span<const PrimitiveKey>& primitiveKeys,
                            const Span<const LightKey>& lightKeys,
                            const Span<const TransformKey>& transformKeys,
                            const GPUQueue& queue);

    void    Construct(MetaLightListConstructionParams,
                      const LightSurfaceParams& boundarySurface,
                      const GPUQueue& queue);

    void                Clear();
    View                Array() const;
    LightLookupTable    IndexHashTable() const;
};

#include "MetaLight.hpp"
