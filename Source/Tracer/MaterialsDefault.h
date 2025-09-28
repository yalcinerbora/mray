#pragma once

#include "Texture.h"
#include "MaterialC.h"
#include "SpectrumC.h"
#include "ParamVaryingData.h"
#include "Random.h"

namespace LambertMatDetail
{
    struct alignas(32) LambertMatData
    {
        Span<const ParamVaryingData<2, Vector3>>        dAlbedo;
        Span<const Optional<TracerTexView<2, Vector3>>> dNormalMaps;
        Span<const MediumKeyPair>                       dMediumKeys;
    };

    template <class SpectrumContext = SpectrumContextIdentity>
    struct LambertMaterial
    {
        using Surface           = DefaultSurface;
        using OptionalNormalMap = Optional<TracerTexView<2, Vector3>>;
        using AlbedoMap         = typename SpectrumContext:: template ParamVaryingAlbedo<2>;
        using SpectrumConverter = typename SpectrumContext::Converter;
        using DataSoA           = LambertMatData;
        //
        static constexpr uint32_t SampleRNCount = 2;

        private:
        const Surface&              surface;
        Spectrum                    albedo;
        Optional<Vector3>           optNormal;
        MediumKeyPair               mediumKeys;

        public:
        MR_GF_DECL
        LambertMaterial(const SpectrumConverter& sTransContext,
                        const Surface& surface,
                        const DataSoA& soa, MaterialKey mk);

        MR_GF_DECL
        SampleT<BxDFResult> SampleBxDF(const Vector3& wO,
                                       RNGDispenser& dispenser) const;
        MR_GF_DECL Float    Pdf(const Ray& wI, const Vector3& wO) const;

        MR_GF_DECL Spectrum Evaluate(const Ray& wI, const Vector3& wO) const;
        MR_GF_DECL bool     IsEmissive() const;
        MR_GF_DECL Spectrum Emit(const Vector3& wO) const;
        MR_GF_DECL Float    Specularity() const;
        MR_GF_DECL
        RayConeSurface      RefractRayCone(const RayConeSurface&, const Vector3& wO) const;

        MR_GF_DECL
        static bool         IsAllTexturesAreResident(const Surface&, const DataSoA&,
                                                     MaterialKey);
    };
}

namespace ReflectMatDetail
{
    struct ReflectMatData
    {
        Span<const MediumKeyPair>  dMediumKeys;
    };

    template <class SpectrumContext = SpectrumContextIdentity>
    struct ReflectMaterial
    {
        using Surface           = DefaultSurface;
        using SpectrumConverter = typename SpectrumContext::Converter;
        using DataSoA           = ReflectMatData;
        //
        static constexpr uint32_t SampleRNCount = 0;

        private:
        const Surface& surface;
        MediumKeyPair  mediumKeys;

        public:
        MR_GF_DECL
        ReflectMaterial(const SpectrumConverter& sTransContext,
                        const Surface& surface,
                        const DataSoA& soa, MaterialKey mk);

        MR_GF_DECL
        SampleT<BxDFResult> SampleBxDF(const Vector3& wO,
                                       RNGDispenser& dispenser) const;
        MR_GF_DECL Float    Pdf(const Ray& wI, const Vector3& wO) const;
        MR_GF_DECL Spectrum Evaluate(const Ray& wI, const Vector3& wO) const;
        MR_GF_DECL bool     IsEmissive() const;
        MR_GF_DECL Spectrum Emit(const Vector3& wO) const;
        MR_GF_DECL Float    Specularity() const;
        MR_GF_DECL
        RayConeSurface      RefractRayCone(const RayConeSurface&, const Vector3& wO) const;
        MR_GF_DECL
        static bool         IsAllTexturesAreResident(const Surface&, const DataSoA&,
                                                     MaterialKey);
    };
}

namespace RefractMatDetail
{
    struct alignas(32) RefractMatData
    {
        Span<const MediumKeyPair>   dMediumKeys;
        Span<const Vector3>         dFrontCauchyCoeffs;
        Span<const Vector3>         dBackCauchyCoeffs;
    };

    template <class SpectrumContext = SpectrumContextIdentity>
    struct RefractMaterial
    {
        using Surface           = DefaultSurface;
        using SpectrumConverter = typename SpectrumContext::Converter;
        using DataSoA           = RefractMatData;
        //
        static constexpr uint32_t SampleRNCount = 0;

        private:
        const Surface&  surface;
        MediumKeyPair   mediumKeys;
        Float           frontIoR;
        Float           backIoR;

        public:
        MR_GF_DECL
        RefractMaterial(const SpectrumConverter& sTransContext,
                        const Surface& surface,
                        const DataSoA& soa, MaterialKey mk);

        MR_GF_DECL
        SampleT<BxDFResult> SampleBxDF(const Vector3& wO,
                                       RNGDispenser& dispenser) const;
        MR_GF_DECL Float    Pdf(const Ray& wI, const Vector3& wO) const;

        MR_GF_DECL Spectrum Evaluate(const Ray& wI, const Vector3& wO) const;
        MR_GF_DECL bool     IsEmissive() const;
        MR_GF_DECL Spectrum Emit(const Vector3& wO) const;
        MR_GF_DECL Float    Specularity() const;
        MR_GF_DECL
        RayConeSurface      RefractRayCone(const RayConeSurface&, const Vector3& wO) const;

        MR_GF_DECL
        static bool         IsAllTexturesAreResident(const Surface&, const DataSoA&,
                                                     MaterialKey);
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
        Span<const MediumKeyPair>                       dMediumKeys;
    };

    template <class SpectrumContext = SpectrumContextIdentity>
    struct UnrealMaterial
    {
        using Surface           = DefaultSurface;
        using AlbedoMap         = typename SpectrumContext:: template ParamVaryingAlbedo<2>;
        using SpectrumConverter = typename SpectrumContext::Converter;
        using FloatMap          = ParamVaryingData<2, Float>;
        using OptionalNormalMap = Optional<TracerTexView<2, Vector3>>;
        using DataSoA           = UnrealMatData;
        //
        static constexpr uint32_t SampleRNCount = 3;

        private:
        const Surface&      surface;
        Spectrum            albedo;
        Optional<Vector3>   optNormal;
        Float               roughness;
        Float               specular;
        Float               metallic;
        MediumKeyPair       mediumKeys;

        MR_GF_DECL
        Float MISRatio(Float avgAlbedo) const;
        MR_GF_DECL
        Spectrum CalculateF0() const;
        MR_GF_DECL
        Float ConvertProbHToL(Float VdH, Float pdfH) const;

        public:
        MR_GF_DECL
        UnrealMaterial(const SpectrumConverter& sTransContext,
                       const Surface& surface,
                       const DataSoA& soa, MaterialKey mk);

        MR_GF_DECL
        SampleT<BxDFResult> SampleBxDF(const Vector3& wO,
                                       RNGDispenser& dispenser) const;
        MR_GF_DECL Float    Pdf(const Ray& wI, const Vector3& wO) const;

        MR_GF_DECL Spectrum Evaluate(const Ray& wI, const Vector3& wO) const;
        MR_GF_DECL bool     IsEmissive() const;
        MR_GF_DECL Spectrum Emit(const Vector3& wO) const;
        MR_GF_DECL Float    Specularity() const;
        MR_GF_DECL
        RayConeSurface      RefractRayCone(const RayConeSurface&, const Vector3& wO) const;

        MR_GF_DECL
        static bool         IsAllTexturesAreResident(const Surface&, const DataSoA&,
                                                     MaterialKey);
    };
}

class MatGroupLambert final : public GenericGroupMaterial<MatGroupLambert>
{
    public:
    using DataSoA   = LambertMatDetail::LambertMatData;
    template<class STContext = SpectrumContextIdentity>
    using Material  = LambertMatDetail::LambertMaterial<STContext>;
    using Surface   = typename Material<>::Surface;

    private:
    Span<ParamVaryingData<2, Vector3>>          dAlbedo;
    Span<Optional<TracerTexView<2, Vector3>>>   dNormalMaps;
    Span<MediumKeyPair>                         dMediumKeys;
    DataSoA                                     soa;

    protected:

    public:
    static std::string_view TypeName();

                    MatGroupLambert(uint32_t groupId,
                                    const GPUSystem&,
                                    const TextureViewMap&,
                                    const TextureMap&);
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
    void            Finalize(const GPUQueue&) override;
};

class MatGroupReflect final : public GenericGroupMaterial<MatGroupReflect>
{
    public:
    using DataSoA   = ReflectMatDetail::ReflectMatData;
    template<class STContext = SpectrumContextIdentity>
    using Material  = ReflectMatDetail::ReflectMaterial<STContext>;
    using Surface   = typename Material<>::Surface;

    private:
    Span<MediumKeyPair> dMediumKeys;
    DataSoA             soa;

    protected:

    public:
    static std::string_view TypeName();

                    MatGroupReflect(uint32_t groupId,
                                    const GPUSystem&,
                                    const TextureViewMap&,
                                    const TextureMap&);
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
    void            Finalize(const GPUQueue&) override;
};

class MatGroupRefract final : public GenericGroupMaterial<MatGroupRefract>
{
    public:
    using DataSoA   = RefractMatDetail::RefractMatData;
    template<class STContext = SpectrumContextIdentity>
    using Material  = RefractMatDetail::RefractMaterial<STContext>;
    using Surface   = typename Material<>::Surface;

    private:
    Span<MediumKeyPair> dMediumKeys;
    Span<Vector3>       dFrontCauchyCoeffs;
    Span<Vector3>       dBackCauchyCoeffs;
    DataSoA             soa;

    protected:

    public:
    static std::string_view TypeName();

                    MatGroupRefract(uint32_t groupId,
                                    const GPUSystem&,
                                    const TextureViewMap&,
                                    const TextureMap&);
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
    void            Finalize(const GPUQueue&) override;
};

class MatGroupUnreal final : public GenericGroupMaterial<MatGroupUnreal>
{
    public:
    using DataSoA   = UnrealMatDetail::UnrealMatData;
    template<class STContext = SpectrumContextIdentity>
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
    Span<MediumKeyPair>                         dMediumKeys;
    DataSoA                                     soa;
    protected:

    public:
    static std::string_view TypeName();

                    MatGroupUnreal(uint32_t groupId,
                                   const GPUSystem&,
                                   const TextureViewMap&,
                                   const TextureMap&);
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
    void            Finalize(const GPUQueue&) override;
};

#include "MaterialsDefault.hpp"

static_assert(MaterialGroupC<MatGroupLambert>);