#pragma once

#include "Core/Types.h"
#include "Core/TracerI.h"

#include "TracerTypes.h"
#include "GenericGroup.h"

template <class MatType>
concept MaterialC = requires(MatType mt,
                             typename MatType::SpectrumConverter sc,
                             RNGDispenser rng)
{
    // Has a surface definition
    // Materials can only act on a single surface type
    typename MatType::SpectrumConverter;
    typename MatType::Surface;
    typename MatType::DataSoA;

    // Constructor
    MatType(sc, typename MatType::DataSoA{}, MaterialKey{});

    // Sample should support BSSRDF (it will return a "ray"
    // instead of a direction)
    // This means for other types ray.pos == surface.pos
    // At the same time we sample a reflectio
    {mt.SampleBxDF(Vector3{}, typename MatType::Surface{},
                   rng)
    } -> std::same_as<SampleT<BxDFResult>>;

    // Given wO (with outgoing position)
    // and wI (with incoming position)
    // Calculate the pdf value
    // TODO: should we provide a surface?
    // For BSSRDF how tf we get the pdf???
    {mt.Pdf(Ray{}, Ray{}, typename MatType::Surface{})
    } -> std::same_as<Float>;

    // How many random numbers the sampler of this class uses
    {mt.SampleRNCount()} -> std::same_as<uint32_t>;

    // Evaluate material given w0, wI
    {mt.Evaluate(Ray{}, Vector3{}, typename MatType::Surface{})
    }-> std::same_as<Spectrum>;

    // Emissive Query
    {mt.IsEmissive()} -> std::same_as<bool>;

    // Emission
    {mt.Emit(Vector3{}, typename MatType::Surface{})
    } -> std::same_as<Spectrum>;

    // Streaming texture query
    // Given surface, all textures of this material should be accessible
    { mt.IsAllTexturesAreResident(typename MatType::Surface{})} -> std::same_as<bool>;
};

template <class MGType>
concept MaterialGroupC = requires()
{
    // Material type satisfies its concept (at least on default form)
    requires MaterialC<typename MGType::template Material<>>;
    // SoA fashion material data. This will be used to access internal
    // of the primitive with a given an index
    typename MGType::DataSoA;
    std::is_same_v<typename MGType::DataSoA,
                   typename MGType::template Material<>::DataSoA>;
    // Surface Type. Materials can only act on single surface
    typename MGType::Surface;
    // Sanity check
    requires std::is_same_v<typename MGType::Surface,
                            typename MGType::template Material<>::Surface>;

    // TODO: Some Functions
    GenericGroupC<MGType>;
};

template<class Child>
using GenericGroupMaterial = GenericGroupT<Child, MaterialKey, MatAttributeInfo>;