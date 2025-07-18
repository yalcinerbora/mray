#pragma once

#include "MediumC.h"
#include "Random.h"
#include "Texture.h"

namespace MediumDetail
{
    // Lets try the new SoA span on homogeneous medium
    // (Instead of 80 bytes, this is 48 bytes!)
    struct alignas(16) HomogeneousMediumData
        : public SoASpan <const Vector3, const Vector3,
                          const Vector3, const Float>
    {
        using Base = SoASpan<const Vector3, const Vector3,
                             const Vector3, const Float>;
        enum I
        {
            SIGMA_A,
            SIGMA_S,
            EMISSION,
            HG_PHASE
        };

        using Base::Base;
    };

    class MediumVacuum
    {
        public:
        using enum HomogeneousMediumData::I;
        using DataSoA = EmptyType;
        using SpectrumConverter = typename SpectrumConverterContextIdentity::Converter;

        static constexpr uint32_t SampleScatteringRNCount = 0;

        public:
        MRAY_HYBRID     MediumVacuum(const SpectrumConverter&,
                                     const DataSoA&, MediumKey);

        MRAY_HYBRID
        ScatterSample   SampleScattering(const Vector3& wO, RNGDispenser& rng) const;
        MRAY_HYBRID
        Float           PdfScattering(const Vector3& wI, const Vector3& wO) const;

        MRAY_HYBRID
        Spectrum        SigmaA(const Vector3& uv) const;
        MRAY_HYBRID
        Spectrum        SigmaS(const Vector3& uv) const;
        MRAY_HYBRID
        Spectrum        Emission(const Vector3& uv) const;
    };

    template <class SpectrumTransformer = SpectrumConverterContextIdentity>
    class MediumHomogeneous
    {
        public:
        using DataSoA = HomogeneousMediumData;
        using SpectrumConverter = typename SpectrumTransformer::Converter;
        using enum HomogeneousMediumData::I;

        static constexpr uint32_t SampleScatteringRNCount = 2;

        private:
        Spectrum sigmaA;
        Spectrum sigmaS;
        Spectrum emission;
        Float    g;

        public:
        MRAY_HYBRID     MediumHomogeneous(const SpectrumConverter& specConverter,
                                          const DataSoA&, MediumKey);

        MRAY_HYBRID
        ScatterSample   SampleScattering(const Vector3& wO, RNGDispenser& rng) const;
        MRAY_HYBRID
        Float           PdfScattering(const Vector3& wI, const Vector3& wO) const;

        MRAY_HYBRID
        Spectrum        SigmaA(const Vector3& uv) const;
        MRAY_HYBRID
        Spectrum        SigmaS(const Vector3& uv) const;
        MRAY_HYBRID
        Spectrum        Emission(const Vector3& uv) const;
    };
}

class MediumGroupVacuum : public GenericGroupMedium<MediumGroupVacuum>
{
    public:
    using DataSoA   = EmptyType;

    template<class STContext = SpectrumConverterContextIdentity>
    using Medium  = MediumDetail::MediumVacuum;

    public:
    static std::string_view TypeName();

                    MediumGroupVacuum(uint32_t groupId,
                                      const GPUSystem&,
                                      const TextureViewMap&,
                                      const TextureMap&);

    //
    void            CommitReservations() override;

    AttribInfoList  AttributeInfo() const override;
    void            PushAttribute(MediumKey id,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MediumKey id,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MediumKey idStart, MediumKey idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;

    // Extra
    void            PushTexAttribute(MediumKey idStart, MediumKey idEnd,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MediumKey idStart, MediumKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MediumKey idStart, MediumKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>,
                                     const GPUQueue& queue) override;

    DataSoA         SoA() const;
};

class MediumGroupHomogeneous : public GenericGroupMedium<MediumGroupHomogeneous>
{
    public:
    using DataSoA   = MediumDetail::HomogeneousMediumData;

    template<class STContext = SpectrumConverterContextIdentity>
    using Medium  = MediumDetail::MediumHomogeneous<STContext>;

    private:
    Span<Vector3>   dSigmaA;
    Span<Vector3>   dSigmaS;
    Span<Vector3>   dEmission;
    Span<Float>     dPhaseVal;
    DataSoA         soa;

    public:
    static std::string_view TypeName();

                    MediumGroupHomogeneous(uint32_t groupId,
                                           const GPUSystem&,
                                           const TextureViewMap&,
                                           const TextureMap&);

    void            CommitReservations() override;

    AttribInfoList  AttributeInfo() const override;
    void            PushAttribute(MediumKey id,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MediumKey id,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MediumKey idStart, MediumKey idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;

    // Extra
    void            PushTexAttribute(MediumKey idStart, MediumKey idEnd,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MediumKey idStart, MediumKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MediumKey idStart, MediumKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>,
                                     const GPUQueue& queue) override;

    DataSoA         SoA() const;
};

#include "MediumsDefault.hpp"

static_assert(MediumC<MediumDetail::MediumVacuum>);
static_assert(MediumGroupC<MediumGroupVacuum>);
static_assert(MediumC<MediumDetail::MediumHomogeneous<>>);
static_assert(MediumGroupC<MediumGroupHomogeneous>);