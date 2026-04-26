#pragma once

#include "Device/GPUSystem.h"
#include "TracerTypes.h"
#include "GenericGroup.h"
#include "SpectrumC.h"
#include "DistributionFunctions.h"

struct ScatterSampleT
{
    Vector3 wI;
    Float   phaseVal;
};

struct RaySegment
{
    Vector2 tMM;
    Spectrum sMajor;
};

enum class MediumEvent : uint8_t
{
    ABSORBED    = 0,
    SCATTERED   = 1,
    TRANSMITTED = 2
};

struct MediumQuery
{
    Spectrum           sigmaA;
    Spectrum           sigmaS;
    Optional<Spectrum> emission;
};

template<class SegmentItType>
concept SegmentIteratorC = requires(SegmentItType& t)
{
    {t.curSegment} -> std::same_as<RaySegment>;
    {t.Advance()} -> std::same_as<bool>;
};

using ScatterSample = SampleT<ScatterSampleT>;

template<class MediumType>
concept MediumC = requires(MediumType md,
                           typename MediumType::SpectrumConverter sc,
                           RNGDispenser& rng)
{
    typename MediumType::DataSoA;
    typename MediumType::SpectrumConverter;
    typename MediumType::Traverser;

    // API
    MediumType(sc, typename MediumType::DataSoA{}, MediumKey{});

    // TODO: Designing this requires paper reading
    // Currently I've no/minimal information about this topic.
    // We need to do a ray marching style approach probably.
    // Instead of doing a single thread per scatter,
    // We can do single warp (or even block?, prob too much)
    // per ray.
    //
    // I've checked the PBRT book/code, it does similar but code
    // is hard to track.
    //
    // All in all,
    // Medium generates an iterator, (for homogeneous it is the full ray)
    // for spatially varying media it is dense grid and does DDA march over it.
    //
    // Iterator calls a callback function that does the actual work,
    // It can prematurely terminate the iteration due to scattering/absorption etc.
    // March logic should not be here it will be the renderer's responsibility
    // Phase function should be here so we need a scatter function
    // that creates a ray.
    {md.SampleScattering(Vector3{}, Vector3{}, rng)} -> std::same_as<ScatterSample>;
    {md.PdfScattering(Vector3{}, Vector3{}, Vector3{})} -> std::same_as<Float>;
    {md.EvalScattering(Vector3{}, Vector3{}, Vector3{})} -> std::same_as<Float>;
    {md.Query(Vector3{}, Float{})} -> std::same_as<MediumQuery>;
    {md.GenTraverser(Ray{}, Vector2{})} -> std::same_as<typename MediumType::Traverser>;
    // Sample RN count
    MediumType::SampleScatteringRNList;
    requires std::is_same_v<decltype(MediumType::SampleScatteringRNList), const RNRequestList>;

    // TODO:
    // We need to expose the iterator in a different way here, because we may
    // dedicate a warp to handle a single ray, so code should abstract it away
};

template<class MGType>
concept MediumGroupC = requires(MGType mg)
{
    // Internal Medium type that satisfies its concept
    requires MediumC<typename MGType::template Medium<>>;

    // SoA fashion light data. This will be used to access internal
    // of the light with a given an index
    typename MGType::DataSoA;
    requires std::is_same_v<typename MGType::DataSoA,
                            typename MGType::template Medium<>::DataSoA>;

    // Acquire SoA struct of this primitive group
    {mg.SoA()} -> std::same_as<typename MGType::DataSoA>;
};

using GenericGroupMediumT   = GenericTexturedGroupT<MediumKey, MediumAttributeInfo>;
using MediumGroupPtr        = std::unique_ptr<GenericGroupMediumT>;

template<class Child>
class GenericGroupMedium : public GenericGroupMediumT
{
    public:
                        GenericGroupMedium(uint32_t groupId,
                                           const GPUSystem&,
                                           const TextureViewMap&,
                                           const TextureMap&,
                                           size_t allocationGranularity = 2_MiB,
                                           size_t initialReservationSize = 4_MiB);
    std::string_view    Name() const override;
};

namespace MediumDetail
{
    struct SingleSegmentIterator
    {
        RaySegment curSegment;

        MR_PF_DECL bool Advance();
    };

    template<class SegmentIteratorT>
    struct MediumTraverser
    {
        using SegmentIterator = SegmentIteratorT;
        SegmentIterator it;
        Float           dt;

        //
        MR_HF_DECL
        MediumTraverser(const SegmentIterator& it);

        MR_HF_DECL
        bool SampleTMajor(Spectrum& tMaj, Spectrum& sMaj,
                          Float& rayT, Float xi, uint32_t channelIndex = 0);
    };

    template <class SpectrumContext = SpectrumContextIdentity>
    class MediumVacuum
    {
        public:
        using DataSoA           = EmptyType;
        using SpectrumConverter = typename SpectrumContext::Converter;
        using Traverser         = MediumTraverser<SingleSegmentIterator>;

        static constexpr RNRequestList SampleScatteringRNList = RNRequestList();

        public:
        MR_PF_DECL_V    MediumVacuum(const SpectrumConverter&,
                                     const DataSoA&, MediumKey) noexcept;

        MR_PF_DECL
        ScatterSample   SampleScattering(const Vector3& wO,
                                         const Vector3& p,
                                         RNGDispenser& rng) const noexcept;
        MR_PF_DECL
        Float           PdfScattering(const Vector3& wI,
                                      const Vector3& wO,
                                      const Vector3& p) const noexcept;
        MR_PF_DECL
        Float           EvalScattering(const Vector3& wI,
                                       const Vector3& wO,
                                       const Vector3& p) const noexcept;

        MR_PF_DECL
        MediumQuery     Query(const Vector3& p, Float xi) const;
        MR_HF_DECL
        Traverser       GenTraverser(const Ray& ray, const Vector2& tMM) const;
    };
}

class MediumGroupVacuum : public GenericGroupMedium<MediumGroupVacuum>
{
    public:
    using DataSoA   = EmptyType;

    template<class STContext = SpectrumContextIdentity>
    using Medium  = MediumDetail::MediumVacuum<STContext>;

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

namespace MediumDetail
{

MR_PF_DEF
bool SingleSegmentIterator::Advance()
{
    return false;
}

template<class S>
MR_HF_DEF
MediumTraverser<S>::MediumTraverser(const SegmentIterator& it)
    : it(it)
    , dt(0)
{}

template<class S>
MR_HF_DEF
bool MediumTraverser<S>::SampleTMajor(Spectrum& tMaj, Spectrum& sMaj,
                                      Float& t, Float xi, uint32_t channelIndex)
{
    using Distribution::Common::SampleExp;
    const uint32_t I = channelIndex;

    tMaj = Spectrum(1);
    do
    {
        using MathConstants::VerySmallEpsilon;
        //
        const RaySegment& segment = it.curSegment;
        Float dist = segment.tMM[1] - segment.tMM[0];
        bool isNearVacuum = segment.sMajor[I] < VerySmallEpsilon<Float>();
        Float dtSample = (isNearVacuum)
                            ? std::numeric_limits<Float>::max()
                            : SampleExp(xi, segment.sMajor[I]).value;

        // Sampling
        dt = Math::Min(dtSample, dist);
        tMaj *= Math::Exp(-dt * segment.sMajor);
        assert(Math::IsFinite(tMaj));
        t += dt;
        sMaj = segment.sMajor;

        // We are inside the segment and sampled
        // signal that the sampling will continue
        if(t < dist) return true;

        // If we overshoot the current segment,
        // advance the iterator. If it fails
        // we fully passthrough the media
        if(!it.Advance()) return false;

        // Segment is changed, update the t as anchor,
        // we will add dt starting from here
        t = segment.tMM[0];
    }
    while(true);

    // This should not be reached, but terminate if it is reached
    assert(false && "Unreachable Segment Iteration");
    return false;
}

template <class SC>
MR_PF_DEF_V
MediumVacuum<SC>::MediumVacuum(const SpectrumConverter&,
                               const DataSoA&, MediumKey) noexcept
{}

template <class SC>
MR_PF_DEF
ScatterSample MediumVacuum<SC>::SampleScattering(const Vector3&, const Vector3&,
                                                 RNGDispenser&) const noexcept
{
    return ScatterSample
    {
        .value =
        {
            .wI = Vector3::Zero(),
            .phaseVal = Float(0.0)
        },
        .pdf = Float(0.0)
    };
}

template <class SC>
MR_PF_DEF
Float MediumVacuum<SC>::PdfScattering(const Vector3&, const Vector3&,
                                      const Vector3&) const noexcept
{
    return Float(0.0);
}

template <class SC>
MR_PF_DECL
Float MediumVacuum<SC>::EvalScattering(const Vector3&,
                                       const Vector3&,
                                       const Vector3&) const noexcept
{
    return Float(1.0);
}

template <class SC>
MR_PF_DECL
MediumQuery MediumVacuum<SC>::Query(const Vector3&, Float) const
{
    return MediumQuery
    {
        .sigmaA = Spectrum::Zero(),
        .sigmaS = Spectrum::Zero(),
        .emission = std::nullopt
    };
}

template <class SC>
MR_HF_DEF
typename MediumVacuum<SC>::Traverser
MediumVacuum<SC>::GenTraverser(const Ray&, const Vector2& tMM) const
{
    return MediumTraverser(SingleSegmentIterator
    {
        .curSegment =
        {
            .tMM = tMM,
            .sMajor = Spectrum::Zero()
        }
    });
}

}

template <class C>
GenericGroupMedium<C>::GenericGroupMedium(uint32_t groupId,
                                          const GPUSystem& sys,
                                          const TextureViewMap& texViewMap,
                                          const TextureMap& texMap,
                                          size_t allocationGranularity,
                                          size_t initialReservationSize)
    : GenericGroupMediumT(groupId, sys,
                          texViewMap, texMap,
                          allocationGranularity,
                          initialReservationSize)
{}

template <class C>
std::string_view GenericGroupMedium<C>::Name() const
{
    return C::TypeName();
}
