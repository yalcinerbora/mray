#pragma once

#include "Core/Types.h"

#include "TracerTypes.h"
#include "GenericGroup.h"
#include "TextureView.h"

namespace MaterialCommon
{
    static constexpr Float SpecularThreshold = Float(0.95);

    MR_PF_DECL bool IsSpecular(Float specularity) noexcept;
}

using NormalMap = Optional<TracerTexView<2, Vector3>>;

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

    //
    {MatType::GetNormalMap(typename MatType::DataSoA{}, MaterialKey{})
    } -> std::same_as<NormalMap>;

    // Constructor
    MatType(sc, typename MatType::Surface{},
            typename MatType::DataSoA{}, MaterialKey{});

    // Sample should support BSSRDF (it will return a "ray"
    // instead of a direction)
    // This means for other types ray.pos == surface.pos
    // At the same time we sample a reflection
    {mt.SampleBxDF(Vector3{}, rng)} -> std::same_as<BxDFSample>;

    // Given wO (with outgoing position)
    // and wI (with incoming position)
    // Calculate the pdf value
    // TODO: should we provide a surface?
    // For BSSRDF how tf we get the pdf???
    {mt.Pdf(Ray{}, Vector3{})} -> std::same_as<Float>;

    // How many random numbers the sampler of this class uses
    MatType::SampleRNList;
    requires std::is_same_v<decltype(MatType::SampleRNList), const RNRequestList>;

    // Evaluate material given w0, wI
    {mt.Evaluate(Ray{}, Vector3{})}-> std::same_as<BxDFEval>;

    // Emissive Query
    {mt.IsEmissive()} -> std::same_as<bool>;

    // Emission
    {mt.Emit(Vector3{})} -> std::same_as<Spectrum>;

    // Specularity of the material
    // The value is between [0-1]. If one the material
    // is perfectly specular (non-physical perfect mirror, glass
    // etc)
    // This is not a bool to make it flexible
    {mt.Specularity()} -> std::same_as<Float>;
    // TODO: Add for position as well (except for BSSRDF
    // it will return zero)

    // Refract the RayCone and calculate the
    // invBetaN of the cone. If refraction
    // does not make sense with this material
    // This function should be an identity function.
    {mt.RefractRayCone(RayConeSurface{}, Vector3{})
    } -> std::same_as<RayConeSurface>;

    // Streaming texture query
    // Given surface, all textures of this material should be accessible
    {MatType::IsAllTexturesAreResident(typename MatType::Surface{},
                                       typename MatType::DataSoA{},
                                       MaterialKey{})
    } -> std::same_as<bool>;
};

template <class MGType>
concept MaterialGroupC = requires(MGType mg)
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

    // TODO: Some more functions
    // ...

    // TODO: This concept requires "Reserve" function to be visible,
    // however we switched it so...
    //requires GenericGroupC<MGType>;
};

using GenericGroupMaterialT = GenericTexturedGroupT<MaterialKey, MatAttributeInfo>;

using MaterialGroupPtr = std::unique_ptr<GenericGroupMaterialT>;

template <class Child>
class GenericGroupMaterial : public GenericGroupMaterialT
{
    public:
                        GenericGroupMaterial(uint32_t groupId, const GPUSystem&,
                                             const TextureViewMap&,
                                             const TextureMap&,
                                             size_t allocationGranularity = 2_MiB,
                                             size_t initialReservationSize = 4_MiB);
    std::string_view    Name() const override;
};

MR_PF_DEF
bool MaterialCommon::IsSpecular(Float specularity) noexcept
{
    constexpr auto Threshold = SpecularThreshold;
    return specularity >= Threshold;
}

template <class C>
GenericGroupMaterial<C>::GenericGroupMaterial(uint32_t groupId, const GPUSystem& system,
                                              const TextureViewMap& texViewMap,
                                              const TextureMap& texMap,
                                              size_t allocationGranularity,
                                              size_t initialReservationSize)
    : GenericGroupMaterialT(groupId, system,
                            texViewMap, texMap,
                            allocationGranularity,
                            initialReservationSize)
{}

template <class C>
std::string_view GenericGroupMaterial<C>::Name() const
{
    return C::TypeName();
}