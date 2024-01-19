#pragma once

#include "PrimitiveC.h"
#include "ParamVaryingData.h"
#include "TracerTypes.h"
#include "Random.h"

namespace LightDetail
{
    struct LightData
    {
        Span<const ParamVaryingData<2, Spectrum>>   dRadiances;
        Span<const MediumId>                        dMediumIds;
        Bitspan<const uint32_t>                     dIsTwoSidedFlags;
    };

    // Meta Primitive Related Light
    template<PrimitiveC Primitive, class SpectrumTransformer = SpectrumConverterContextIdentity>
    struct PrimLight
    {
        using RadianceMap = typename SpectrumTransformer:: template RendererParamVaryingData<2>;

        public:
        using DataSoA       = LightData;
        private:
        Primitive           prim;
        RadianceMap         radiance;
        MediumId            initialMedium;
        bool                isTwoSided;

        public:
        MRAY_HYBRID         PrimLight(const typename SpectrumTransformer::Converter& sTransContext,
                                      const Primitive& p,
                                      const LightData& soa, LightId);

        MRAY_HYBRID
        SampleT<Vector3>    SampleSolidAngle(RNGDispenser& dispenser,
                                             const Vector3& distantPoint,
                                             const Vector3& dir) const;
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
        Spectrum            Emit(const Vector3& wO,
                                 const typename Primitive::Hit& hit) const;
        MRAY_HYBRID
        Spectrum            Emit(const Vector3& wO,
                                 const Vector3& surfacePoint) const;

        MRAY_HYBRID bool    IsPrimitiveBackedLight() const;
    };

}

#include "Lights.hpp"