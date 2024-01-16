#pragma once

#include "MaterialC.h"
#include "ParamVaryingData.h"

namespace LambertMatDetail
{
    struct LambertMatData
    {
        Span<ParamVaryingData<2, Spectrum>>             dAlbedo;
        Span<Optional<ParamVaryingData<2, Vector3>>>    dNormalMaps;
        Span<MediumId>                                  dMediumIds;
    };

    template <class SpectrumTransformer = SpectrumConverterContextIdentity>
    struct LambertMaterial
    {
        using Surface = DefaultSurface;
        using OptionalNormalMap = Optional<ParamVaryingData<2, Vector3>>;
        using AlbedoMap = typename SpectrumTransformer:: template RendererParamVaryingData<2>;

        private:
        const AlbedoMap                 albedoTex;
        const OptionalNormalMap&        normalMapTex;
        MediumId                        mediumId;

        public:
        MRAY_HYBRID
        LambertMaterial(const typename SpectrumTransformer::Converter& sTransContext,
                        const LambertMatData& soa, MaterialId id);
        MRAY_HYBRID
        Sample<BxDFResult> SampleBxDF(const Vector3& wI,
                                      const Surface& surface,
                                      RNGDispenser& dispenser) const;
        MRAY_HYBRID
        Float Pdf(const Ray& wI,
                  const Ray& wO,
                  const Surface& surface) const;
        MRAY_HYBRID
        uint32_t SampleRNCount() const;
        MRAY_HYBRID
        Spectrum Evaluate(const Ray& wO,
                          const Vector3& wI,
                          const Surface& surface) const;
        MRAY_HYBRID
        bool IsEmissive() const;
        MRAY_HYBRID
        Spectrum Emit(const Vector3& wO,
                      const Surface& surf) const;

        MRAY_HYBRID
        bool IsAllTexturesAreResident(const Surface& surface) const;
    };

}

struct MatGroupLambert
{
    using DataSoA = LambertMatDetail::LambertMatData;
    template<class STContext = SpectrumConverterContextIdentity>
    using Material = LambertMatDetail::LambertMaterial<STContext>;

    using Surface = typename Material<>::Surface;
};

#include "MaterialsDefault.hpp"

static_assert(MaterialGroupC<MatGroupLambert>);