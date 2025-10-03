#pragma once

#include "Device/GPUSystem.h"
#include "TracerTypes.h"
#include "GenericGroup.h"

struct ScatterSampleT
{
    Vector3 wI;
    Float   phaseVal;
};

using ScatterSample = SampleT<ScatterSampleT>;

template<class MediumType>
concept MediumC = requires(MediumType md,
                           typename MediumType::SpectrumConverter sc,
                           RNGDispenser& rng)
{
    typename MediumType::DataSoA;
    typename MediumType::SpectrumConverter;

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
    {md.SampleScattering(Vector3{}, rng)} -> std::same_as<ScatterSample>;
    {md.PdfScattering(Vector3{}, Vector3{})} -> std::same_as<Float>;

    {md.SigmaA(Vector3{})} -> std::same_as<Spectrum>;
    {md.SigmaS(Vector3{})} -> std::same_as<Spectrum>;
    {md.Emission(Vector3{})} -> std::same_as<Spectrum>;
    // Sample RN count
    MediumType::SampleScatteringRNList;
    requires std::is_same_v<decltype(MediumType::SampleScatteringRNList), const RNRequestList>;

    // TODO:
    // We need to expose the iterator in a different way here, because we may
    // dedicate a warp to handle a single ray, so code should abstract it away

    // Type traits
    requires std::is_trivially_copyable_v<MediumType>;
    requires std::is_trivially_destructible_v<MediumType>;
    requires std::is_move_assignable_v<MediumType>;
    requires std::is_move_constructible_v<MediumType>;
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
