#pragma once

#include "PrimitiveC.h"
#include "ParamVaryingData.h"
#include "TracerTypes.h"
#include "Random.h"
#include "Distributions.h"
#include "LightC.h"

#include "Core/GraphicsFunctions.h"

namespace LightDetail
{
    using DistributionPwC2D = typename DistributionGroupPwC2D::Distribution;

    struct LightData
    {
        Span<const ParamVaryingData<2, Spectrum>>   dRadiances;
        Span<const MediumId>                        dMediumIds;
        Bitspan<const uint32_t>                     dIsTwoSidedFlags;
    };

    struct LightSkysphereData
    {
        Span<const ParamVaryingData<2, Spectrum>>   dRadiances;
        Span<const MediumId>                        dMediumIds;
        Span<const DistributionPwC2D>               dDistributions;
        Float                                       sceneDiameter;
    };

    struct SphericalCoordConverter
    {
        static constexpr auto B = std::integral_constant<int, 4>::value;

        MRAY_HYBRID
        static Vector2  DirToUV(const Vector3& dirYUp);
        MRAY_HYBRID
        static Vector3  UVToDir(const Vector2& uv);
        MRAY_HYBRID
        static Float    ToSolidAnglePdf(Float pdf, const Vector3& dirYUp);
        MRAY_HYBRID
        static Float    ToSolidAnglePdf(Float pdf, const Vector2& uv);
    };

    struct CoOctoCoordConverter
    {
        static constexpr auto A = std::integral_constant<double, 3.0>::value;

        MRAY_HYBRID
        static Vector2  DirToUV(const Vector3& dirYUp);
        MRAY_HYBRID
        static Vector3  UVToDir(const Vector2& uv);
        MRAY_HYBRID
        static Float    ToSolidAnglePdf(Float pdf, const Vector3& dirYUp);
        MRAY_HYBRID
        static Float    ToSolidAnglePdf(Float pdf, const Vector2& uv);
    };

    template <class Converter>
    concept CoordConverterC = requires()
    {
        {Converter::DirToUV(Vector3{})} -> std::same_as<Vector2>;
        {Converter::UVToDir(Vector2{})} -> std::same_as<Vector3>;
        {Converter::ToSolidAnglePdf(Float{}, Vector2{})} -> std::same_as<Float>;
        {Converter::ToSolidAnglePdf(Float{}, Vector3{})} -> std::same_as<Float>;
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

        private:
        Ref<const Primitive>    prim;
        RadianceMap             radiance;
        MediumId                initialMedium;
        bool                    isTwoSided;

        public:
        MRAY_HYBRID         LightPrim(const SpectrumConverter& sTransContext,
                                      const Primitive& p,
                                      const LightData& soa, LightId);

        MRAY_HYBRID
        SampleT<Vector3>    SampleSolidAngle(RNGDispenser& rng,
                                             const Vector3& distantPoint) const;
        MRAY_HYBRID
        Float               PdfSolidAngle(const typename Primitive::Hit& hit,
                                          const Vector3& distantPoint,
                                          const Vector3& dir) const;
        MRAY_HYBRID
        uint32_t            SampleSolidAngleRNCount() const;
        MRAY_HYBRID
        SampleT<Ray>        SampleRay(RNGDispenser& rng) const;
        MRAY_HYBRID
        Float               PdfRay(const Ray&) const;
        MRAY_HYBRID
        uint32_t            SampleRayRNCount() const;

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

        private:
        Ref<const Primitive>    prim;
        DistributionPwC2D       dist2D;
        RadianceMap             radiance;
        MediumId                initialMedium;
        Float                   sceneDiameter;

        public:
        MRAY_HYBRID         LightSkysphere(const typename SpectrumTransformer::Converter& sTransContext,
                                           const Primitive& p,
                                           const DataSoA& soa, LightId);

        MRAY_HYBRID
        SampleT<Vector3>    SampleSolidAngle(RNGDispenser& dispenser,
                                             const Vector3& distantPoint) const;
        MRAY_HYBRID
        Float               PdfSolidAngle(const typename Primitive::Hit& hit,
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
                                       const typename Primitive::Hit& hit) const;
        MRAY_HYBRID
        Spectrum            EmitViaSurfacePoint(const Vector3& wO,
                                                const Vector3& surfacePoint) const;

        MRAY_HYBRID bool    IsPrimitiveBackedLight() const;
    };

}

template <PrimitiveGroupC PrimGroupT>
class LightGroupPrim
{
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
    uint32_t            groupId;
    const GPUSystem&    system;
    const PrimGroup&    primGroup;
    DataSoA             soa;

    public:
    static std::string_view TypeName();
                            LightGroupPrim(uint32_t groupId,
                                           const GPUSystem& system,
                                           const PrimGroup& primGroup);

    DataSoA                 SoA() const;
    const PrimGroup&        PrimitiveGroup() const;

};

template <LightDetail::CoordConverterC CoordConverter>
class LightGroupSkysphere
{
    public:
    using PrimGroup     = EmptyPrimGroup;
    using DataSoA       = typename LightDetail::LightSkysphereData;

    // Prim Type
    template<class TContext = TransformContextIdentity>
    using Primitive     = EmptyPrimitive<TContext>;
    // Light Type
    template <class TransformContext = TransformContextIdentity,
              class SpectrumConverterContext = SpectrumConverterContextIdentity>
    using Light = typename LightDetail::LightSkysphere<CoordConverter, TransformContext, SpectrumConverterContext>;

    private:
    uint32_t            groupId;
    const GPUSystem&    system;
    const PrimGroup&    primGroup;
    DataSoA             soa;

    public:
    static std::string_view TypeName();

                            LightGroupSkysphere(uint32_t groupId,
                                                const GPUSystem& system,
                                                const PrimGroup& primGroup);

    DataSoA                 SoA() const;
    const PrimGroup&        PrimitiveGroup() const;

};

#include "Lights.hpp"

using CoOctoCoordConverter = LightDetail::CoOctoCoordConverter;
using SphericalCoordConverter = LightDetail::SphericalCoordConverter;

