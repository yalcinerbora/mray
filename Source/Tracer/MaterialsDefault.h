#pragma once

#include "Core/GraphicsFunctions.h"

#include "MaterialC.h"
#include "ParamVaryingData.h"
#include "Random.h"

namespace LambertMatDetail
{
    struct alignas(32) LambertMatData
    {
        Span<const ParamVaryingData<2, Spectrum>>       dAlbedo;
        Span<const Optional<TextureView<2, Vector3>>>   dNormalMaps;
        Span<const MediumKey>                           dMediumIds;
    };

    template <class SpectrumTransformer = SpectrumConverterContextIdentity>
    struct LambertMaterial
    {
        using Surface           = DefaultSurface;
        using OptionalNormalMap = Optional<TextureView<2, Vector3>>;
        using AlbedoMap         = typename SpectrumTransformer:: template RendererParamVaryingData<2>;
        using SpectrumConverter = typename SpectrumTransformer::Converter;
        using DataSoA           = LambertMatData;

        private:
        const AlbedoMap             albedoTex;
        const OptionalNormalMap&    normalMapTex;
        MediumKey                   mediumId;

        public:
        MRAY_HYBRID
        LambertMaterial(const SpectrumConverter& sTransContext,
                        const DataSoA& soa, MaterialKey mk);

        MRAY_HYBRID
        SampleT<BxDFResult>     SampleBxDF(const Vector3& wI,
                                           const Surface& surface,
                                           RNGDispenser& dispenser) const;
        MRAY_HYBRID Float       Pdf(const Ray& wI,
                                const Ray& wO,
                                const Surface& surface) const;
        MRAY_HYBRID uint32_t    SampleRNCount() const;
        MRAY_HYBRID Spectrum    Evaluate(const Ray& wO,
                                         const Vector3& wI,
                                         const Surface& surface) const;
        MRAY_HYBRID bool        IsEmissive() const;
        MRAY_HYBRID Spectrum    Emit(const Vector3& wO,
                                     const Surface& surf) const;
        MRAY_HYBRID bool        IsAllTexturesAreResident(const Surface& surface) const;
    };

}

class MatGroupLambert : public GenericGroupMaterial<MatGroupLambert>
{
    public:
    using DataSoA   = LambertMatDetail::LambertMatData;
    template<class STContext = SpectrumConverterContextIdentity>
    using Material  = LambertMatDetail::LambertMaterial<STContext>;
    using Surface   = typename Material<>::Surface;

    private:
    Span<ParamVaryingData<2, Spectrum>>     dAlbedo;
    Span<Optional<TextureView<2, Vector3>>> dNormalMaps;
    Span<MediumKey>                         dMediumIds;
    DataSoA                                 soa;

    public:
    static std::string_view TypeName();

                        MatGroupLambert(uint32_t groupId, const GPUSystem&);
    void                CommitReservations() override;
    AttribInfoList      AttributeInfo() const override;
    void                PushAttribute(MaterialKey batchId,
                                      uint32_t attributeIndex,
                                      MRayInput data) override;
    void                PushAttribute(MaterialKey batchId,
                                      const Vector2ui& subRange,
                                      uint32_t attributeIndex,
                                      MRayInput data) override;
    void                PushAttribute(const Vector<2, MaterialKey::Type>& idRange,
                                      uint32_t attributeIndex,
                                      MRayInput data) override;

    DataSoA             SoA() const;
};

#include "MaterialsDefault.hpp"

static_assert(MaterialGroupC<MatGroupLambert>);