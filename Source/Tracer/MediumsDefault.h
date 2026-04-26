#pragma once

#include "MediumC.h"
#include "SpectrumC.h"
#include "Random.h"
#include "Texture.h"

#include "VolumetricSVO.h"

#include "Device/GPUTexture.h"

// TODO: Transfer this to somewhere proper later.
// Currently blackbody radiation is only used in volumes when
// they have tempature
namespace BlackbodySPD
{
    namespace Detail
    {
        // Trying to create constants as precise as possible
        using LD = long double;
        inline constexpr LD PLANCK_CONST = 6.62607015e-34;
        inline constexpr LD BOLTZ_CONST = 1.380649e-23;
        inline constexpr LD C = 299'792'458;
        inline constexpr LD WIEN_CONST = 2.897771955e-3;
        // We provide nm scale wavelengths to our system
        inline constexpr LD NM_TO_M_CONST = 1e9;
        inline constexpr LD EMIT_FACTOR = (2 * C * C * PLANCK_CONST *
                                           NM_TO_M_CONST * NM_TO_M_CONST *
                                           NM_TO_M_CONST * NM_TO_M_CONST *
                                           NM_TO_M_CONST);
        inline constexpr long double EXP_FACTOR = (PLANCK_CONST * C / BOLTZ_CONST *
                                                   NM_TO_M_CONST);
        inline constexpr LD SPD_FACTOR = []()
        {
            // Result of "e^(PLANCK_CONST * C / (WIEN_CONST * BOLTZ_CONST) - 1"
            // Our constexpr Exp is single precision so I've put this in a calculator
            // and get the result.
            constexpr LD MAX_EXP_FACTOR = 142.32492163952976;
            // We are incorporating NM_TO_M factor here to ease the FP calculation
            constexpr LD x = WIEN_CONST * NM_TO_M_CONST;
            LD x2 = x * x;
            LD w5 = x2 * x2 * x;
            return MAX_EXP_FACTOR * w5;
        }();
    }
    // Then above constants are converted to "Float", renderer's FP type.
    inline constexpr Float EMIT_FACTOR = Float(Detail::EMIT_FACTOR);
    inline constexpr Float EXP_FACTOR = Float(Detail::EXP_FACTOR);
    inline constexpr Float SPD_FACTOR = Float(Detail::SPD_FACTOR);

    // Plank's Law, from here (or PBRT Book)
    // https://topex.ucsd.edu/rs/radiation.pdf
    MR_PF_DECL
    Float PlancksLaw(Float lambda, Float tempKelvin)
    {
        Float l = lambda;
        Float l2 = l * l;
        Float l5 = l2 * l2 * l;
        //
        Float expPart = Math::Exp(EXP_FACTOR / (l * tempKelvin)) - Float(1);
        Float result = EMIT_FACTOR / (l5 * expPart);
        return result;
    }

    // This is from PBRT book
    MR_PF_DECL
    Float PeakValue(Float tempKelvin)
    {
        constexpr Float FACTOR = Float(Detail::WIEN_CONST * Detail::NM_TO_M_CONST);
        Float lambda = FACTOR / tempKelvin;
        return PlancksLaw(lambda, tempKelvin);
    }

    // This function automatically divides the two functions above
    // most of the constants are factored out etc.
    MR_PF_DECL
    Float NormalizedSPD(Float lambda, Float tempKelvin)
    {
        Float lt = lambda * tempKelvin;
        Float lt2 = lt * lt;
        Float lt5 = lt2 * lt2 * lt;
        Float e = Math::Exp(EXP_FACTOR / (lambda * tempKelvin)) - Float(1);
        return SPD_FACTOR / (lt5 * e);
    }
}

namespace MediumDetail
{
    template<uint32_t XYZ_BITS>
    struct DenseDDAIterator
    {
        static constexpr Float DELTA_XYZ = Float(1) / Float(1 << XYZ_BITS);

        const TracerTexView<3, Float>& majDensityTex;
        const Ray&                     r;
        Spectrum                       sigmaT;
        // State
        Vector3                 nextAxes;
        Vector3                 deltaT;
        Float                   tMax; // Clamped to edge of grid
        //
        RaySegment curSegment;
        MR_GF_DECL DenseDDAIterator(const TracerTexView<3, Float>&,
                                    Spectrum sigmaT,
                                    const Ray& r,
                                    const Vector2& tMM);

        MR_GF_DECL bool     Advance();
        MR_PF_DECL uint32_t SelectAxis() const;
    };

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

    // TODO: Design volume pipeline
    struct alignas(16) HeterogeneousMediumData : public SoASpan
    <
        const Vector3,                              // Sigma A
        const Vector3,                              // Sigma S
        const Float,                                // Phase G Term
        const ParamVaryingData<3, Float>,           // Density
        const Optional<ParamVaryingData<3, Float>>, // Tempature
        const Vector2,                              // TempatureRange
        const TracerTexView<3, Float>,              // Majorant
        const VolumetricSVO::VolGrid6_2             // Topology
    >
    {
        using Base = SoASpan
        <
            const Vector3,                              // Sigma A
            const Vector3,                              // Sigma S
            const Float,                                // Phase G Term
            const ParamVaryingData<3, Float>,           // Density
            const Optional<ParamVaryingData<3, Float>>, // Tempature
            const Vector2,                              // TempatureRange
            const TracerTexView<3, Float>,              // Majorant
            const VolumetricSVO::VolGrid6_2             // Topology
        >;

        enum I
        {
            SIGMA_A,
            SIGMA_S,
            HG_PHASE,
            DENSITY,
            TEMPATURE,
            TEMPATURE_RANGE,
            MAJORANT,
            TOPOLOGY
        };
        using Base::Base;
    };

    template <class SpectrumContext = SpectrumContextIdentity>
    class MediumHomogeneous
    {
        public:
        using enum HomogeneousMediumData::I;
        using DataSoA           = HomogeneousMediumData;
        using SpectrumConverter = typename SpectrumContext::Converter;
        using Traverser         = MediumTraverser<SingleSegmentIterator>;

        static constexpr RNRequestList SampleScatteringRNList = GenRNRequestList<2>();

        private:
        Spectrum           sigmaA;
        Spectrum           sigmaS;
        Optional<Spectrum> emission;
        Float              g;

        public:
        MR_HF_DECL      MediumHomogeneous(const SpectrumConverter& specConverter,
                                          const DataSoA&, MediumKey);

        MR_HF_DECL
        ScatterSample   SampleScattering(const Vector3& wO,
                                         const Vector3& p,
                                         RNGDispenser& rng) const;
        MR_HF_DECL
        Float           PdfScattering(const Vector3& wI,
                                      const Vector3& wO,
                                      const Vector3& p) const;
        MR_PF_DECL
        Float           EvalScattering(const Vector3& wI,
                                       const Vector3& wO,
                                       const Vector3& p) const noexcept;

        MR_HF_DECL
        MediumQuery     Query(const Vector3& p, Float xi) const;
        MR_HF_DECL
        Traverser       GenTraverser(const Ray& ray, const Vector2& tMM) const;
    };

    template <class SpectrumContext = SpectrumContextIdentity>
    class MediumHeterogeneous
    {
        public:
        static constexpr uint32_t GRID_BITS = 6;
        //
        using enum HeterogeneousMediumData::I;
        using DataSoA           = HeterogeneousMediumData;
        using SpectrumConverter = typename SpectrumContext::Converter;
        // We use 64x64x64 data as dense tex
        using Traverser         = MediumTraverser<DenseDDAIterator<GRID_BITS>>;

        static constexpr RNRequestList SampleScatteringRNList = GenRNRequestList<2>();

        private:
        // These are constant so we pre-load these
        Spectrum sigmaA;
        Spectrum sigmaS;
        Float    phaseG;
        Vector2  tempatureRange;
        // These are constant but the data inside is not
        ParamVaryingData<3, Float>           densityMap;
        Optional<ParamVaryingData<3, Float>> tempatureMap;
        TracerTexView<3, Float>              majMap;
        // TODO: Check if this is better as reference
        VolumetricSVO::VolGrid6_2            topology;
        //
        const SpectrumConverter* sc;

        public:
        MR_HF_DECL      MediumHeterogeneous(const SpectrumConverter& specConverter,
                                            const DataSoA&, MediumKey);

        MR_HF_DECL
        ScatterSample   SampleScattering(const Vector3& wO,
                                         const Vector3& p,
                                         RNGDispenser& rng) const;
        MR_HF_DECL
        Float           PdfScattering(const Vector3& wI,
                                      const Vector3& wO,
                                      const Vector3& p) const;
        MR_PF_DECL
        Float           EvalScattering(const Vector3& wI,
                                       const Vector3& wO,
                                       const Vector3& p) const noexcept;
        MR_GF_DECL
        MediumQuery     Query(const Vector3& p, Float xi) const;
        MR_HF_DECL
        Traverser       GenTraverser(const Ray& ray, const Vector2& tMM) const;
    };
}

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

class MediumGroupHeterogeneous : public GenericGroupMedium<MediumGroupHeterogeneous>
{
    public:
    using DataSoA   = MediumDetail::HeterogeneousMediumData;

    template<class STContext = SpectrumContextIdentity>
    using Medium  = MediumDetail::MediumHeterogeneous<STContext>;

    private:
    Span<Vector3>                              dSigmaA;
    Span<Vector3>                              dSigmaS;
    Span<Float>                                dPhases;
    Span<ParamVaryingData<3, Float>>           dDensityMaps;
    //
    Span<Optional<ParamVaryingData<3, Float>>> dTempMaps;
    Span<Vector2>                              dTempRanges;
    Span<VolumetricSVO::VolGrid6_2>            dTopologies;
    //
    // Generated Data
    TextureBackingMemory                       majTexMemory;
    std::vector<Texture<3, Float>>             majTextures;
    Span<TracerTexView<3, Float>>              dMajorantMaps;
    //
    DataSoA       soa;


    public:
    static std::string_view TypeName();

                    MediumGroupHeterogeneous(uint32_t groupId,
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

static_assert(MediumC<MediumDetail::MediumVacuum<>>);
static_assert(MediumGroupC<MediumGroupVacuum>);
static_assert(MediumC<MediumDetail::MediumHomogeneous<>>);
static_assert(MediumGroupC<MediumGroupHomogeneous>);
static_assert(MediumC<MediumDetail::MediumHeterogeneous<>>);
static_assert(MediumGroupC<MediumGroupHeterogeneous>);