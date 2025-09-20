#pragma once

#include "MediumC.h"
#include "SpectrumC.h"
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
        using SpectrumConverter = typename SpectrumContextIdentity::Converter;

        static constexpr uint32_t SampleScatteringRNCount = 0;

        public:
        MR_PF_DECL_V    MediumVacuum(const SpectrumConverter&,
                                     const DataSoA&, MediumKey) noexcept;

        MR_PF_DECL
        ScatterSample   SampleScattering(const Vector3& wO, RNGDispenser& rng) const noexcept;
        MR_PF_DECL
        Float           PdfScattering(const Vector3& wI, const Vector3& wO) const noexcept;

        MR_PF_DECL
        Spectrum        SigmaA(const Vector3& uv) const noexcept;
        MR_PF_DECL
        Spectrum        SigmaS(const Vector3& uv) const noexcept;
        MR_PF_DECL
        Spectrum        Emission(const Vector3& uv) const noexcept;
    };

    template <class SpectrumContext = SpectrumContextIdentity>
    class MediumHomogeneous
    {
        public:
        using DataSoA = HomogeneousMediumData;
        using SpectrumConverter = typename SpectrumContext::Converter;
        using enum HomogeneousMediumData::I;

        static constexpr uint32_t SampleScatteringRNCount = 2;

        private:
        Spectrum sigmaA;
        Spectrum sigmaS;
        Spectrum emission;
        Float    g;

        public:
        MR_HF_DECL      MediumHomogeneous(const SpectrumConverter& specConverter,
                                          const DataSoA&, MediumKey);

        MR_HF_DECL
        ScatterSample   SampleScattering(const Vector3& wO, RNGDispenser& rng) const;
        MR_HF_DECL
        Float           PdfScattering(const Vector3& wI, const Vector3& wO) const;

        MR_HF_DECL
        Spectrum        SigmaA(const Vector3& uv) const;
        MR_HF_DECL
        Spectrum        SigmaS(const Vector3& uv) const;
        MR_HF_DECL
        Spectrum        Emission(const Vector3& uv) const;
    };
}

class MediumGroupVacuum : public GenericGroupMedium<MediumGroupVacuum>
{
    public:
    using DataSoA   = EmptyType;

    template<class STContext = SpectrumContextIdentity>
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

    template<class STContext = SpectrumContextIdentity>
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