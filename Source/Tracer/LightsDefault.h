#pragma once

#include "PrimitiveC.h"
#include "LightC.h"
#include "SpectrumC.h"
#include "TracerTypes.h"
#include "ParamVaryingData.h"
#include "Random.h"
#include "Distributions.h"
#include "Bitspan.h"
#include "ColorConverter.h"
#include "Texture.h"

namespace LightDetail
{
    using DistributionPwC2D = typename DistributionGroupPwC2D::Distribution2D;

    struct alignas(32) LightData
    {
        Span<const ParamVaryingData<2, Vector3>>    dRadiances;
        Bitspan<const uint32_t>                     dIsTwoSidedFlags;
    };

    struct alignas(32) LightSkysphereData
    {
        Span<const ParamVaryingData<2, Vector3>>    dRadiances;
        Span<const DistributionPwC2D>               dDistributions;
        Float                                       sceneDiameter;
    };

    struct SphericalCoordConverter
    {
        MR_HF_DECL
        static Vector2  DirToUV(const Vector3& dirYUp);
        MR_HF_DECL
        static Vector3  UVToDir(const Vector2& uv);
        MR_HF_DECL
        static Float    ToSolidAnglePdf(Float pdf, const Vector3& dirYUp);
        MR_HF_DECL
        static Float    ToSolidAnglePdf(Float pdf, const Vector2& uv);

        MR_HF_DECL
        static std::array<Vector2, 2> Gradient(Float coneAperture,
                                               const Vector3& dirYUp);
    };

    struct CoOctaCoordConverter
    {
        MR_HF_DECL
        static Vector2  DirToUV(const Vector3& dirYUp);
        MR_HF_DECL
        static Vector3  UVToDir(const Vector2& uv);
        MR_HF_DECL
        static Float    ToSolidAnglePdf(Float pdf, const Vector3& dirYUp);
        MR_HF_DECL
        static Float    ToSolidAnglePdf(Float pdf, const Vector2& uv);

        MR_HF_DECL
        static std::array<Vector2, 2> Gradient(Float coneAperture,
                                               const Vector3& dirYUp);
    };

    // Meta Primitive Related Light
    template<PrimitiveC PrimitiveT, class SpectrumContext = SpectrumContextIdentity>
    struct LightPrim
    {
        using RadianceMap = typename SpectrumContext:: template ParamVaryingRadiance<2>;

        public:
        using DataSoA           = LightData;
        using SpectrumConverter = typename SpectrumContext::Converter;
        using Primitive         = PrimitiveT;
        //
        static constexpr bool   IsPrimitiveBackedLight    = true;

        static constexpr RNRequestList SampleSolidAngleRNList = Primitive::SampleRNList;
        static constexpr RNRequestList SampleRayRNList        = SampleSolidAngleRNList.Append(2);

        private:
        Ref<const Primitive>    prim;
        RadianceMap             radiance;
        bool                    isTwoSided;

        public:
        MR_HF_DECL          LightPrim(const SpectrumConverter& sTransContext,
                                      const Primitive& p,
                                      const LightData& soa, LightKey);

        MR_HF_DECL
        SampleT<Vector3>    SampleSolidAngle(RNGDispenser& rng,
                                             const Vector3& distantPoint) const;
        MR_HF_DECL
        Float               PdfSolidAngle(const typename Primitive::Hit& hit,
                                          const Vector3& distantPoint,
                                          const Vector3& dir) const;
        MR_HF_DECL
        SampleT<Ray>        SampleRay(RNGDispenser& rng) const;
        MR_HF_DECL
        Float               PdfRay(const Ray&) const;
        MR_HF_DECL
        Spectrum            EmitViaHit(const Vector3& wO,
                                       const typename Primitive::Hit& hit,
                                       const RayCone& rayCone) const;
        MR_HF_DECL
        Spectrum            EmitViaSurfacePoint(const Vector3& wO,
                                                const Vector3& surfacePoint,
                                                const RayCone& rayCone) const;
    };

     // Meta Primitive Related Light
    template<CoordConverterC CoordConverter,
             TransformContextC TContext = TransformContextIdentity,
             class SpectrumContext = SpectrumContextIdentity>
    struct LightSkysphere
    {
        using RadianceMap   = typename SpectrumContext:: template ParamVaryingRadiance<2>;

        public:
        using DataSoA           = LightSkysphereData;
        using SpectrumConverter = typename SpectrumContext::Converter;
        using Primitive         = EmptyPrimitive<TContext>;
        //
        static constexpr bool           IsPrimitiveBackedLight = false;
        static constexpr RNRequestList  SampleSolidAngleRNList = GenRNRequestList<2>();
        static constexpr RNRequestList  SampleRayRNList        = GenRNRequestList<2, 2>();

        private:
        Ref<const Primitive>    prim;
        DistributionPwC2D       dist2D;
        RadianceMap             radiance;
        Float                   sceneDiameter;

        MR_HF_DECL
        SampleT<Vector2>    SampleUV(Vector2 xi) const;
        MR_HF_DECL
        Float               PdfUV(Vector2 uv) const;

        public:
        MR_HF_DECL          LightSkysphere(const SpectrumConverter& sTransContext,
                                           const Primitive& p,
                                           const DataSoA& soa, LightKey);

        MR_HF_DECL
        SampleT<Vector3>    SampleSolidAngle(RNGDispenser& dispenser,
                                             const Vector3& distantPoint) const;
        MR_HF_DECL
        Float               PdfSolidAngle(const typename Primitive::Hit& hit,
                                          const Vector3& distantPoint,
                                          const Vector3& dir) const;
        MR_HF_DECL
        SampleT<Ray>        SampleRay(RNGDispenser& dispenser) const;
        MR_HF_DECL
        Float               PdfRay(const Ray&) const;
        MR_HF_DECL
        Spectrum            EmitViaHit(const Vector3& wO,
                                       const typename Primitive::Hit& hit,
                                       const RayCone& rayCone) const;
        MR_HF_DECL
        Spectrum            EmitViaSurfacePoint(const Vector3& wO,
                                                const Vector3& surfacePoint,
                                                const RayCone& rayCone) const;
    };

    template <TransformContextC TContext = TransformContextIdentity,
              class SpectrumContext = SpectrumContextIdentity>
    class LightNull
    {
        public:
        using DataSoA           = EmptyType;
        using SpectrumConverter = typename SpectrumContext::Converter;
        using Primitive         = EmptyPrimitive<TContext>;
        //
        static constexpr bool     IsPrimitiveBackedLight      = false;
        static constexpr RNRequestList SampleSolidAngleRNList = RNRequestList();
        static constexpr RNRequestList SampleRayRNList        = RNRequestList();

        MR_HF_DECL          LightNull(const SpectrumConverter& sTransContext,
                                      const Primitive& p,
                                      const DataSoA& soa, LightKey);

        MR_HF_DECL
        SampleT<Vector3>    SampleSolidAngle(RNGDispenser&,
                                             const Vector3&) const;
        MR_HF_DECL
        Float               PdfSolidAngle(const typename Primitive::Hit&,
                                          const Vector3&,
                                          const Vector3&) const;
        MR_HF_DECL
        SampleT<Ray>        SampleRay(RNGDispenser&) const;
        MR_HF_DECL
        Float               PdfRay(const Ray&) const;
        MR_HF_DECL
        Spectrum            EmitViaHit(const Vector3&,
                                       const typename Primitive::Hit&,
                                       const RayCone&) const;
        MR_HF_DECL
        Spectrum            EmitViaSurfacePoint(const Vector3&, const Vector3&,
                                                const RayCone&) const;
    };
}

// TODO: Making this class final breaks linking weirdly
// (clang-19).
//
// Variadic arg of MetaLightArrayT is created in T_DefaultLights.cu:
// "
//  PackedTypes<LightGroupPrim<PrimGroupTriangle>, TransformGroupSingle>,
//  PackedTypes<LightGroupPrim<PrimGroupSkinnedTriangle>, TransformGroupMulti>
// "
// Somehow PackedTypes<LightGroupPrim<PrimGroupTriangle>, TransformGroupSingle>,
// linked perfectly fine (PackedTypes is empty variadic pack class btw)
// but skinned triangle references could not be found
//
// This smells like name mangling issue but dunno
// Report to llvm bu dont know how (minimal reproducing example will be hassle)
template <PrimitiveGroupC PrimGroupT>
class LightGroupPrim : public GenericGroupLight<LightGroupPrim<PrimGroupT>>
{
    using Parent        = GenericGroupLight<LightGroupPrim<PrimGroupT>>;

    public:
    using PrimGroup     = PrimGroupT;
    using DataSoA       = typename LightDetail::LightData;

    // Prim Type
    template<class TContext = TransformContextIdentity>
    using Primitive = typename PrimGroup:: template Primitive<TContext>;
    // Light Type
    template <class TransformContext = TransformContextIdentity,
              class SpectrumConverterContext = SpectrumContextIdentity>
    using Light = typename LightDetail::LightPrim<Primitive<TransformContext>, SpectrumConverterContext>;

    private:
    const PrimGroup&                    primGroup;
    Span<ParamVaryingData<2, Vector3>>  dRadiances;
    Span<Vector2ui>                     dPrimRanges;
    Span<uint32_t>                      dIsTwoSidedFlags;
    DataSoA                             soa;

    public:
    static std::string_view TypeName();

    public:
    // Constructors & Destructor
                    LightGroupPrim(uint32_t groupId,
                                   const GPUSystem& system,
                                   const TextureViewMap&,
                                   const TextureMap&,
                                   const GenericGroupPrimitiveT&);

    void                    CommitReservations() override;
    LightAttributeInfoList  AttributeInfo() const override;

    void            PushAttribute(LightKey id,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(LightKey id,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(LightKey idStart, LightKey idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushTexAttribute(LightKey idStart, LightKey idEnd,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(LightKey idStart, LightKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(LightKey idStart, LightKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>,
                                     const GPUQueue& queue) override;

    DataSoA                         SoA() const;
    const PrimGroup&                PrimitiveGroup() const;
    const GenericGroupPrimitiveT&   GenericPrimGroup() const override;
    bool                            IsPrimitiveBacked() const override;
    void                            Finalize(const GPUQueue& q) override;
};

template <CoordConverterC CoordConverter>
class LightGroupSkysphere final : public GenericGroupLight<LightGroupSkysphere<CoordConverter>>
{
    using Parent            = GenericGroupLight<LightGroupSkysphere<CoordConverter>>;
    using DistributionPwC2D = typename DistributionGroupPwC2D::Distribution2D;
    public:
    using PrimGroup     = PrimGroupEmpty;
    using DataSoA       = typename LightDetail::LightSkysphereData;

    // Prim Type
    template<class TContext = TransformContextIdentity>
    using Primitive         = EmptyPrimitive<TContext>;
    // Light Type
    template <class TransformContext = TransformContextIdentity,
              class SpectrumConverterContext = SpectrumContextIdentity>
    using Light = typename LightDetail::LightSkysphere<CoordConverter, TransformContext, SpectrumConverterContext>;

    private:
    const PrimGroup&                    primGroup;
    Span<ParamVaryingData<2, Vector3>>  dRadiances;
    Float                               sceneDiameter;
    DataSoA                             soa;
    DistributionGroupPwC2D              pwcDistributions;
    std::vector<const GenericTexture*>  radianceFieldTextures;

    public:
    static std::string_view TypeName();
    //
                LightGroupSkysphere(uint32_t groupId,
                                    const GPUSystem& system,
                                    const TextureViewMap&,
                                    const TextureMap&,
                                    const GenericGroupPrimitiveT&);


    void                    CommitReservations() override;
    LightAttributeInfoList  AttributeInfo() const override;

    void            PushAttribute(LightKey id,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(LightKey id,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(LightKey idStart, LightKey idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushTexAttribute(LightKey idStart, LightKey idEnd,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(LightKey idStart, LightKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(LightKey idStart, LightKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>,
                                     const GPUQueue& queue) override;

    DataSoA                         SoA() const;
    const PrimGroup&                PrimitiveGroup() const;
    const GenericGroupPrimitiveT&   GenericPrimGroup() const override;
    bool                            IsPrimitiveBacked() const override;
    void                            SetSceneDiameter(Float) override;
    void                            Finalize(const GPUQueue& q) override;
    size_t                          GPUMemoryUsage() const override;
};

class LightGroupNull : public GenericGroupLight<LightGroupNull>
{
    public:
    using PrimGroup = PrimGroupEmpty;
    using DataSoA   = EmptyType;

    template<class TContext = TransformContextIdentity>
    using Primitive = EmptyPrimitive<TContext>;

    template <class TContext = TransformContextIdentity,
              class SpectrumContext = SpectrumContextIdentity>
    using Light = LightDetail::LightNull<TContext, SpectrumContext>;

    private:
    const PrimGroup& primGroup;

    public:
    static std::string_view TypeName();

    LightGroupNull(uint32_t groupId,
                   const GPUSystem& system,
                   const TextureViewMap&,
                   const TextureMap&,
                   const GenericGroupPrimitiveT&);

    void                    CommitReservations() override;
    LightAttributeInfoList  AttributeInfo() const override;

    void            PushAttribute(LightKey,
                                  uint32_t,
                                  TransientData,
                                  const GPUQueue&) override;
    void            PushAttribute(LightKey,
                                  uint32_t,
                                  const Vector2ui&,
                                  TransientData,
                                  const GPUQueue&) override;
    void            PushAttribute(LightKey, LightKey,
                                  uint32_t,
                                  TransientData,
                                  const GPUQueue&) override;
    void            PushTexAttribute(LightKey, LightKey,
                                     uint32_t,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue&) override;
    void            PushTexAttribute(LightKey, LightKey,
                                     uint32_t,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue&) override;
    void            PushTexAttribute(LightKey, LightKey,
                                     uint32_t,
                                     std::vector<TextureId>,
                                     const GPUQueue&) override;

    DataSoA                         SoA() const;
    const PrimGroup&                PrimitiveGroup() const;
    const GenericGroupPrimitiveT&   GenericPrimGroup() const override;
    bool                            IsPrimitiveBacked() const override;
};

#include "LightsDefault.hpp"

using CoOctaCoordConverter = LightDetail::CoOctaCoordConverter;
using SphericalCoordConverter = LightDetail::SphericalCoordConverter;

extern template class LightGroupSkysphere<CoOctaCoordConverter>;
extern template class LightGroupSkysphere<SphericalCoordConverter>;

static_assert(LightGroupC<LightGroupSkysphere<CoOctaCoordConverter>>);
static_assert(LightGroupC<LightGroupSkysphere<SphericalCoordConverter>>);
static_assert(LightGroupC<LightGroupNull>);
