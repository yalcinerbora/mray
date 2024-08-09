#pragma once

#include "PrimitiveC.h"
#include "ParamVaryingData.h"
#include "TracerTypes.h"
#include "Random.h"
#include "Distributions.h"
#include "LightC.h"

#include "Core/GraphicsFunctions.h"
#include "Core/TypeNameGenerators.h"

namespace LightDetail
{
    using DistributionPwC2D = typename DistributionGroupPwC2D::Distribution2D;

    struct alignas(32) LightData
    {
        Span<const ParamVaryingData<2, Vector3>>    dRadiances;
        Span<const MediumId>                        dMediumIds;
        Span<const Vector2ui>                       dPrimRanges;
        Bitspan<const uint32_t>                     dIsTwoSidedFlags;
    };

    struct alignas(32) LightSkysphereData
    {
        Span<const ParamVaryingData<2, Vector3>>    dRadiances;
        Span<const MediumId>                        dMediumIds;
        Span<const DistributionPwC2D>               dDistributions;
        Float                                       sceneDiameter;
    };

    struct SphericalCoordConverter
    {
        MRAY_HYBRID
        static Vector2  DirToUV(const Vector3& dirYUp);
        MRAY_HYBRID
        static Vector3  UVToDir(const Vector2& uv);
        MRAY_HYBRID
        static Float    ToSolidAnglePdf(Float pdf, const Vector3& dirYUp);
        MRAY_HYBRID
        static Float    ToSolidAnglePdf(Float pdf, const Vector2& uv);
    };

    struct CoOctaCoordConverter
    {
        MRAY_HYBRID
        static Vector2  DirToUV(const Vector3& dirYUp);
        MRAY_HYBRID
        static Vector3  UVToDir(const Vector2& uv);
        MRAY_HYBRID
        static Float    ToSolidAnglePdf(Float pdf, const Vector3& dirYUp);
        MRAY_HYBRID
        static Float    ToSolidAnglePdf(Float pdf, const Vector2& uv);
    };

    // Meta Primitive Related Light
    template<PrimitiveC PrimitiveT, class SpectrumTransformer = SpectrumConverterContextIdentity>
    struct LightPrim
    {
        using RadianceMap = typename SpectrumTransformer:: template RendererParamVaryingData<2>;

        public:
        using DataSoA           = LightData;
        using SpectrumConverter = typename SpectrumTransformer::Converter;
        using Primitive         = PrimitiveT;
        //
        static constexpr uint32_t SampleSolidAngleRNCount   = Primitive::SampleRNCount;
        static constexpr uint32_t SampleRayRNCount          = Primitive::SampleRNCount + 2;

        private:
        Ref<const Primitive>    prim;
        RadianceMap             radiance;
        MediumId                initialMedium;
        bool                    isTwoSided;

        public:
        MRAY_HYBRID         LightPrim(const SpectrumConverter& sTransContext,
                                      const Primitive& p,
                                      const LightData& soa, LightKey);

        MRAY_HYBRID
        SampleT<Vector3>    SampleSolidAngle(RNGDispenser& rng,
                                             const Vector3& distantPoint) const;
        MRAY_HYBRID
        Float               PdfSolidAngle(const typename Primitive::Hit& hit,
                                          const Vector3& distantPoint,
                                          const Vector3& dir) const;
        MRAY_HYBRID
        SampleT<Ray>        SampleRay(RNGDispenser& rng) const;
        MRAY_HYBRID
        Float               PdfRay(const Ray&) const;
        MRAY_HYBRID
        Spectrum            EmitViaHit(const Vector3& wO,
                                       const typename Primitive::Hit& hit) const;
        MRAY_HYBRID
        Spectrum            EmitViaSurfacePoint(const Vector3& wO,
                                                const Vector3& surfacePoint) const;

        MRAY_HYBRID bool    IsPrimitiveBackedLight() const;
    };

     // Meta Primitive Related Light
    template<CoordConverterC CoordConverter,
             TransformContextC TContext = TransformContextIdentity,
             class SpectrumTransformer = SpectrumConverterContextIdentity>
    struct LightSkysphere
    {
        using RadianceMap   = typename SpectrumTransformer:: template RendererParamVaryingData<2>;

        public:
        using DataSoA           = LightSkysphereData;
        using SpectrumConverter = typename SpectrumTransformer::Converter;
        using Primitive         = EmptyPrimitive<TContext>;
        //
        static constexpr uint32_t SampleRayRNCount = 2;
        static constexpr uint32_t SampleSolidAngleRNCount = 2;

        private:
        Ref<const Primitive>    prim;
        DistributionPwC2D       dist2D;
        RadianceMap             radiance;
        MediumId                initialMedium;
        Float                   sceneDiameter;

        public:
        MRAY_HYBRID         LightSkysphere(const SpectrumConverter& sTransContext,
                                           const Primitive& p,
                                           const DataSoA& soa, LightKey);

        MRAY_HYBRID
        SampleT<Vector3>    SampleSolidAngle(RNGDispenser& dispenser,
                                             const Vector3& distantPoint) const;
        MRAY_HYBRID
        Float               PdfSolidAngle(const typename Primitive::Hit& hit,
                                          const Vector3& distantPoint,
                                          const Vector3& dir) const;
        MRAY_HYBRID
        SampleT<Ray>        SampleRay(RNGDispenser& dispenser) const;
        MRAY_HYBRID
        Float               PdfRay(const Ray&) const;
        MRAY_HYBRID
        Spectrum            EmitViaHit(const Vector3& wO,
                                       const typename Primitive::Hit& hit) const;
        MRAY_HYBRID
        Spectrum            EmitViaSurfacePoint(const Vector3& wO,
                                                const Vector3& surfacePoint) const;

        MRAY_HYBRID bool    IsPrimitiveBackedLight() const;
    };

    template <TransformContextC TContext = TransformContextIdentity,
              class SpectrumTransformer = SpectrumConverterContextIdentity>
    class LightNull
    {
        public:
        using DataSoA           = EmptyType;
        using SpectrumConverter = typename SpectrumTransformer::Converter;
        using Primitive         = EmptyPrimitive<TContext>;
        //
        static constexpr uint32_t SampleSolidAngleRNCount   = 0;
        static constexpr uint32_t SampleRayRNCount          = 0;

        MRAY_HYBRID         LightNull(const SpectrumConverter& sTransContext,
                                      const Primitive& p,
                                      const DataSoA& soa, LightKey);

        MRAY_HYBRID
        SampleT<Vector3>    SampleSolidAngle(RNGDispenser&,
                                             const Vector3&) const;
        MRAY_HYBRID
        Float               PdfSolidAngle(const typename Primitive::Hit&,
                                          const Vector3&,
                                          const Vector3&) const;
        MRAY_HYBRID
        SampleT<Ray>        SampleRay(RNGDispenser&) const;
        MRAY_HYBRID
        Float               PdfRay(const Ray&) const;
        MRAY_HYBRID
        Spectrum            EmitViaHit(const Vector3&,
                                       const typename Primitive::Hit&) const;
        MRAY_HYBRID
        Spectrum            EmitViaSurfacePoint(const Vector3&,
                                                const Vector3&) const;
        MRAY_HYBRID bool    IsPrimitiveBackedLight() const;
    };
}

template <PrimitiveGroupC PrimGroupT>
class LightGroupPrim final : public GenericGroupLight<LightGroupPrim<PrimGroupT>>
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
              class SpectrumConverterContext = SpectrumConverterContextIdentity>
    using Light = typename LightDetail::LightPrim<Primitive<TransformContext>, SpectrumConverterContext>;

    private:
    const PrimGroup&                    primGroup;
    Span<ParamVaryingData<2, Vector3>>  dRadiances;
    Span<MediumKey>                     dMediumIds;
    Span<Vector2ui>                     dPrimRanges;
    Bitspan<uint32_t>                   dIsTwoSidedFlags;
    DataSoA                             soa;

    public:
    static std::string_view TypeName();

    public:
    // Constructors & Destructor
                    LightGroupPrim(uint32_t groupId,
                                   const GPUSystem& system,
                                   const TextureViewMap&,
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
    using Primitive     = EmptyPrimitive<TContext>;
    // Light Type
    template <class TransformContext = TransformContextIdentity,
              class SpectrumConverterContext = SpectrumConverterContextIdentity>
    using Light = typename LightDetail::LightSkysphere<CoordConverter, TransformContext, SpectrumConverterContext>;

    private:
    const PrimGroup&                    primGroup;
    Span<ParamVaryingData<2, Vector3>>  dRadiances;
    Span<MediumId>                      dMediumIds;
    Span<DistributionPwC2D>             dDistributions;
    Float                               sceneDiameter;
    DataSoA                             soa;

    public:
    static std::string_view TypeName();
    //
                LightGroupSkysphere(uint32_t groupId,
                                    const GPUSystem& system,
                                    const TextureViewMap&,
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

};

class LightGroupNull : public GenericGroupLight<LightGroupNull>
{
    public:
    using PrimGroup = PrimGroupEmpty;
    using DataSoA   = EmptyType;

    template<class TContext = TransformContextIdentity>
    using Primitive = EmptyPrimitive<TContext>;

    template <class TContext = TransformContextIdentity,
              class SpectrumTransformer = SpectrumConverterContextIdentity>
    using Light = LightDetail::LightNull<TContext, SpectrumTransformer>;

    private:
    const PrimGroup& primGroup;

    public:
    static std::string_view TypeName();

    LightGroupNull(uint32_t groupId,
                   const GPUSystem& system,
                   const TextureViewMap&,
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
