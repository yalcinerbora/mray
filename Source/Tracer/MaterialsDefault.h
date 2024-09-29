#pragma once

#include "DistributionFunctions.h"

#include "MaterialC.h"
#include "ParamVaryingData.h"
#include "Random.h"

namespace LambertMatDetail
{
    struct alignas(32) LambertMatData
    {
        Span<const ParamVaryingData<2, Vector3>>        dAlbedo;
        Span<const Optional<TracerTexView<2, Vector3>>> dNormalMaps;
        Span<const MediumKey>                           dMediumIds;
    };

    template <class SpectrumTransformer = SpectrumConverterContextIdentity>
    struct LambertMaterial
    {
        using Surface           = DefaultSurface;
        using OptionalNormalMap = Optional<TracerTexView<2, Vector3>>;
        using AlbedoMap         = typename SpectrumTransformer:: template RendererParamVaryingData<2>;
        using SpectrumConverter = typename SpectrumTransformer::Converter;
        using DataSoA           = LambertMatData;
        //
        static constexpr uint32_t SampleRNCount = 2;

        private:
        const AlbedoMap             albedoTex;
        const OptionalNormalMap&    normalMapTex;
        MediumKey                   mediumId;

        public:
        MRAY_HYBRID
        LambertMaterial(const SpectrumConverter& sTransContext,
                        const DataSoA& soa, MaterialKey mk);

        MRAY_HYBRID
        SampleT<BxDFResult>     SampleBxDF(const Vector3& wO,
                                           const Surface& surface,
                                           RNGDispenser& dispenser) const;
        MRAY_HYBRID Float       Pdf(const Ray& wI,
                                    const Vector3& wO,
                                    const Surface& surface) const;
        MRAY_HYBRID Spectrum    Evaluate(const Ray& wI,
                                         const Vector3& wO,
                                         const Surface& surface) const;
        MRAY_HYBRID bool        IsEmissive() const;
        MRAY_HYBRID Spectrum    Emit(const Vector3& wO,
                                     const Surface& surf) const;
        MRAY_HYBRID bool        IsAllTexturesAreResident(const Surface& surface) const;
    };

}

namespace ReflectMatDetail
{
    template <class SpectrumTransformer = SpectrumConverterContextIdentity>
    struct ReflectMaterial
    {
        using Surface           = DefaultSurface;
        using SpectrumConverter = typename SpectrumTransformer::Converter;
        using DataSoA           = EmptyType;
        //
        static constexpr uint32_t SampleRNCount = 0;

        public:
        MRAY_HYBRID
        ReflectMaterial(const SpectrumConverter& sTransContext,
                        const DataSoA& soa, MaterialKey mk);

        MRAY_HYBRID
        SampleT<BxDFResult>     SampleBxDF(const Vector3& wO,
                                           const Surface& surface,
                                           RNGDispenser& dispenser) const;
        MRAY_HYBRID Float       Pdf(const Ray& wI,
                                    const Vector3& wO,
                                    const Surface& surface) const;
        MRAY_HYBRID Spectrum    Evaluate(const Ray& wI,
                                         const Vector3& wO,
                                         const Surface& surface) const;
        MRAY_HYBRID bool        IsEmissive() const;
        MRAY_HYBRID Spectrum    Emit(const Vector3& wO,
                                     const Surface& surf) const;
        MRAY_HYBRID bool        IsAllTexturesAreResident(const Surface& surface) const;
    };
}

namespace RefractMatDetail
{
    struct alignas(32) RefractMatData
    {
        Span<const Pair<MediumKey, MediumKey>> dMediumIds;
        Span<const Vector3>                    dFrontCauchyCoeffs;
        Span<const Vector3>                    dBackCauchyCoeffs;
    };

    template <class SpectrumTransformer = SpectrumConverterContextIdentity>
    struct RefractMaterial
    {
        using Surface           = DefaultSurface;
        using SpectrumConverter = typename SpectrumTransformer::Converter;
        using DataSoA           = RefractMatData;
        //
        static constexpr uint32_t SampleRNCount = 0;

        private:
        MediumKey   mKeyFront;
        Spectrum    frontIoR;
        MediumKey   mKeyBack;
        Spectrum    backIoR;

        public:
        MRAY_HYBRID
        RefractMaterial(const SpectrumConverter& sTransContext,
                        const DataSoA& soa, MaterialKey mk);

        MRAY_HYBRID
        SampleT<BxDFResult>     SampleBxDF(const Vector3& wO,
                                           const Surface& surface,
                                           RNGDispenser& dispenser) const;
        MRAY_HYBRID Float       Pdf(const Ray& wI,
                                    const Vector3& wO,
                                const Surface& surface) const;
        MRAY_HYBRID Spectrum    Evaluate(const Ray& wI,
                                         const Vector3& wO,
                                         const Surface& surface) const;
        MRAY_HYBRID bool        IsEmissive() const;
        MRAY_HYBRID Spectrum    Emit(const Vector3& wO,
                                     const Surface& surf) const;
        MRAY_HYBRID bool        IsAllTexturesAreResident(const Surface& surface) const;
    };
}

namespace UnrealMatDetail
{
    struct alignas(32) UnrealMatData
    {
        Span<const ParamVaryingData<2, Vector3>>        dAlbedo;
        Span<const Optional<TracerTexView<2, Vector3>>> dNormalMaps;
        //
        Span<const ParamVaryingData<2, Float>>          dRoughness;
        Span<const ParamVaryingData<2, Float>>          dSpecular;
        Span<const ParamVaryingData<2, Float>>          dMetallic;
        //
        Span<const MediumKey>                           dMediumIds;
    };

    template <class SpectrumTransformer = SpectrumConverterContextIdentity>
    struct UnrealMaterial
    {
        using Surface           = DefaultSurface;
        using AlbedoMap         = typename SpectrumTransformer:: template RendererParamVaryingData<2>;
        using SpectrumConverter = typename SpectrumTransformer::Converter;
        using FloatMap          = ParamVaryingData<2, Float>;
        using OptionalNormalMap = Optional<TracerTexView<2, Vector3>>;
        using DataSoA           = UnrealMatData;
        //
        static constexpr uint32_t SampleRNCount = 3;

        private:
        const AlbedoMap             albedoTex;
        const OptionalNormalMap&    normalMapTex;
        const FloatMap              roughnessTex;
        const FloatMap              specularTex;
        const FloatMap              metallicTex;

        MediumKey                   mediumId;

        MRAY_HYBRID
        Float MISRatio(Float metallic, Float specular, Float avgAlbedo) const;
        MRAY_HYBRID
        Spectrum CalculateF0(Spectrum albedo, Float metallic, Float specular) const;
        MRAY_HYBRID
        Float ConvertProbHToL(Float VdH, Float pdfH) const;
        MRAY_HYBRID
        Tuple<Float, Float, Float, Spectrum>
        FetchData(const Surface&) const;

        public:
        MRAY_HYBRID
        UnrealMaterial(const SpectrumConverter& sTransContext,
                       const DataSoA& soa, MaterialKey mk);

        MRAY_HYBRID
        SampleT<BxDFResult>     SampleBxDF(const Vector3& wO,
                                           const Surface& surface,
                                           RNGDispenser& dispenser) const;
        MRAY_HYBRID Float       Pdf(const Ray& wI,
                                    const Vector3& wO,
                                    const Surface& surface) const;
        MRAY_HYBRID Spectrum    Evaluate(const Ray& wI,
                                         const Vector3& wO,
                                         const Surface& surface) const;
        MRAY_HYBRID bool        IsEmissive() const;
        MRAY_HYBRID Spectrum    Emit(const Vector3& wO,
                                     const Surface& surf) const;
        MRAY_HYBRID bool        IsAllTexturesAreResident(const Surface& surface) const;
    };

}

class MatGroupLambert final : public GenericGroupMaterial<MatGroupLambert>
{
    public:
    using DataSoA   = LambertMatDetail::LambertMatData;
    template<class STContext = SpectrumConverterContextIdentity>
    using Material  = LambertMatDetail::LambertMaterial<STContext>;
    using Surface   = typename Material<>::Surface;

    private:
    Span<ParamVaryingData<2, Vector3>>          dAlbedo;
    Span<Optional<TracerTexView<2, Vector3>>>   dNormalMaps;
    Span<MediumKey>                             dMediumIds;
    DataSoA                                     soa;

    protected:
    void            HandleMediums(const MediumKeyPairList&) override;

    public:
    static std::string_view TypeName();

                    MatGroupLambert(uint32_t groupId,
                                    const GPUSystem&,
                                    const TextureViewMap&);
    void            CommitReservations() override;
    AttribInfoList  AttributeInfo() const override;
    void            PushAttribute(MaterialKey id,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MaterialKey id,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MaterialKey idStart, MaterialKey idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;

    // Extra
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>,
                                     const GPUQueue& queue) override;

    DataSoA         SoA() const;
};

class MatGroupReflect final : public GenericGroupMaterial<MatGroupReflect>
{
    public:
    using DataSoA   = EmptyType;
    template<class STContext = SpectrumConverterContextIdentity>
    using Material  = ReflectMatDetail::ReflectMaterial<STContext>;
    using Surface   = typename Material<>::Surface;

    private:
    Span<MediumKey>     dMediumIds;
    DataSoA             soa;

    protected:
    void            HandleMediums(const MediumKeyPairList&) override;

    public:
    static std::string_view TypeName();

                    MatGroupReflect(uint32_t groupId,
                                    const GPUSystem&,
                                    const TextureViewMap&);
    void            CommitReservations() override;
    AttribInfoList  AttributeInfo() const override;
    void            PushAttribute(MaterialKey id,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MaterialKey id,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MaterialKey idStart, MaterialKey idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;

    // Extra
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>,
                                     const GPUQueue& queue) override;

    DataSoA         SoA() const;
};

class MatGroupRefract final : public GenericGroupMaterial<MatGroupRefract>
{
    public:
    using DataSoA   = RefractMatDetail::RefractMatData;
    template<class STContext = SpectrumConverterContextIdentity>
    using Material  = RefractMatDetail::RefractMaterial<STContext>;
    using Surface   = typename Material<>::Surface;

    private:
    Span<Pair<MediumKey, MediumKey>>    dMediumIds;
    Span<Vector3>                       dFrontCauchyCoeffs;
    Span<Vector3>                       dBackCauchyCoeffs;
    DataSoA                             soa;

    protected:
    void            HandleMediums(const MediumKeyPairList&) override;

    public:
    static std::string_view TypeName();

                    MatGroupRefract(uint32_t groupId,
                                    const GPUSystem&,
                                    const TextureViewMap&);
    void            CommitReservations() override;
    AttribInfoList  AttributeInfo() const override;
    void            PushAttribute(MaterialKey id,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MaterialKey id,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MaterialKey idStart, MaterialKey idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;

    // Extra
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>,
                                     const GPUQueue& queue) override;

    DataSoA         SoA() const;
};

class MatGroupUnreal final : public GenericGroupMaterial<MatGroupUnreal>
{
    public:
    using DataSoA   = UnrealMatDetail::UnrealMatData;
    template<class STContext = SpectrumConverterContextIdentity>
    using Material  = UnrealMatDetail::UnrealMaterial<STContext>;
    using Surface   = typename Material<>::Surface;

    private:
    static constexpr uint32_t ALBEDO_INDEX      = 0;
    static constexpr uint32_t NORMAL_MAP_INDEX  = 1;
    static constexpr uint32_t ROUGHNESS_INDEX   = 2;
    static constexpr uint32_t SPECULAR_INDEX    = 3;
    static constexpr uint32_t METALLIC_INDEX    = 4;

    private:
    Span<ParamVaryingData<2, Vector3>>          dAlbedo;
    Span<Optional<TracerTexView<2, Vector3>>>   dNormalMaps;
    Span<ParamVaryingData<2, Float>>            dRoughness;
    Span<ParamVaryingData<2, Float>>            dSpecular;
    Span<ParamVaryingData<2, Float>>            dMetallic;
    Span<MediumKey>                             dMediumIds;
    DataSoA                                     soa;
    protected:
    void            HandleMediums(const MediumKeyPairList&) override;

    public:
    static std::string_view TypeName();

                    MatGroupUnreal(uint32_t groupId,
                                    const GPUSystem&,
                                    const TextureViewMap&);
    void            CommitReservations() override;
    AttribInfoList  AttributeInfo() const override;
    void            PushAttribute(MaterialKey id,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MaterialKey id,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MaterialKey idStart, MaterialKey idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;

    // Extra
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>,
                                     const GPUQueue& queue) override;

    DataSoA         SoA() const;
};

#include "MaterialsDefault.hpp"

static_assert(MaterialGroupC<MatGroupLambert>);