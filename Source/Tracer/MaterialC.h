#pragma once

#include "Core/Types.h"
#include "Core/TracerI.h"

#include "TracerTypes.h"
#include "GenericGroup.h"

using MediumKeyPairList = std::vector<Pair<MediumKey, MediumKey>>;

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
    {mt.Pdf(Ray{}, Vector3{}, typename MatType::Surface{})
    } -> std::same_as<Float>;

    // How many random numbers the sampler of this class uses
    MatType::SampleRNCount;
    requires std::is_same_v<decltype(MatType::SampleRNCount), const uint32_t>;

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

    // TODO: This concept requres "Reserve" function to be visible,
    // however we switched it so...
    //requires GenericGroupC<MGType>;
};

class GenericGroupMaterialT : public GenericTexturedGroupT<MaterialKey, MatAttributeInfo>
{
    using Parent = GenericTexturedGroupT<MaterialKey, MatAttributeInfo>;
    using typename Parent::IdList;

    protected:
    virtual void    HandleMediums(const MediumKeyPairList&) = 0;

    public:
    // Constructors & Destructor
                    GenericGroupMaterialT(uint32_t groupId, const GPUSystem&,
                                          const TextureViewMap&,
                                          size_t allocationGranularity = 2_MiB,
                                          size_t initialReservartionSize = 4_MiB);
    // Swap the interfaces (old switcharoo)
    private:
    IdList          Reserve(const std::vector<AttributeCountList>&) override;

    public:
    virtual IdList  Reserve(const std::vector<AttributeCountList>&,
                            const MediumKeyPairList&);
};

using MaterialGroupPtr = std::unique_ptr<GenericGroupMaterialT>;

template <class Child>
class GenericGroupMaterial : public GenericGroupMaterialT
{
    public:
                        GenericGroupMaterial(uint32_t groupId, const GPUSystem&,
                                             const TextureViewMap&,
                                             size_t allocationGranularity = 2_MiB,
                                             size_t initialReservartionSize = 4_MiB);
    std::string_view    Name() const override;
};

inline
GenericGroupMaterialT::GenericGroupMaterialT(uint32_t groupId, const GPUSystem& gpuSystem,
                                             const TextureViewMap& map,
                                             size_t allocationGranularity,
                                             size_t initialReservartionSize)
    : Parent(groupId, gpuSystem, map,
             allocationGranularity,
             initialReservartionSize)
{}

inline typename GenericGroupMaterialT::IdList
GenericGroupMaterialT::Reserve(const std::vector<AttributeCountList>&)
{
    throw MRayError("{}: Materials cannot be reserved via this function!", Name());
}

inline typename GenericGroupMaterialT::IdList
GenericGroupMaterialT::Reserve(const std::vector<AttributeCountList>& countArrayList,
                               const MediumKeyPairList& mediumPairs)
{
    // We blocked the virutal chain, but we should be able to use it here
    // We will do the same here anyways migh as well use it.
    auto result = Parent::Reserve(countArrayList);

    std::lock_guard lock(mutex);
    HandleMediums(mediumPairs);
    return result;
}

template <class C>
GenericGroupMaterial<C>::GenericGroupMaterial(uint32_t groupId, const GPUSystem& system,
                                              const TextureViewMap& map,
                                              size_t allocationGranularity,
                                              size_t initialReservartionSize)
    : GenericGroupMaterialT(groupId, system, map,
                            allocationGranularity,
                            initialReservartionSize)
{}

template <class C>
std::string_view GenericGroupMaterial<C>::Name() const
{
    return C::TypeName();
}