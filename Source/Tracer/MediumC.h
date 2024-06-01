#pragma once

#include "Device/GPUSystem.h"
#include "Core/TracerI.h"
#include "TracerTypes.h"
#include "GenericGroup.h"
#include <map>

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

    {md.SampleScattering(Vector3{}, rng)} -> std::same_as<ScatterSample>;
    {md.PdfScattering(Vector3{}, Vector3{})} -> std::same_as<Float>;
    {md.SampleScatteringRNCount()} -> std::same_as<uint32_t>;
    {md.IoR(Vector3{})} -> std::same_as<Spectrum>;
    {md.SigmaA(Vector3{})} -> std::same_as<Spectrum>;
    {md.SigmaS(Vector3{})} -> std::same_as<Spectrum>;
    {md.Emission(Vector3{})} -> std::same_as<Spectrum>;

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
                                           size_t allocationGranularity = 2_MiB,
                                           size_t initialReservartionSize = 4_MiB);
    std::string_view    Name() const override;
};

template <class C>
GenericGroupMedium<C>::GenericGroupMedium(uint32_t groupId,
                                          const GPUSystem& sys,
                                          const TextureViewMap& map,
                                          size_t allocationGranularity,
                                          size_t initialReservartionSize)
    : GenericGroupMediumT(groupId, sys, map,
                          allocationGranularity,
                          initialReservartionSize)
{}

template <class C>
std::string_view GenericGroupMedium<C>::Name() const
{
    return C::TypeName();
}