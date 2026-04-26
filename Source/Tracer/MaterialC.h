#pragma once

#include "Core/Types.h"

#include "TracerTypes.h"
#include "GenericGroup.h"
#include "TextureView.h"
#include "SpectrumC.h"

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

namespace PassthroughMatDetail
{
    template <class SpectrumContext = SpectrumContextIdentity>
    struct PassthroughMaterial
    {
        using Surface           = DefaultSurface;
        using DataSoA           = EmptyType;
        using SpectrumConverter = typename SpectrumContext::Converter;
        //
        static constexpr RNRequestList SampleRNList = RNRequestList();

        private:
        const Surface& surface;

        public:
        MR_GF_DECL
        static NormalMap GetNormalMap(const DataSoA& soa, MaterialKey k);

        MR_PF_DECL_V
        PassthroughMaterial(const SpectrumConverter& sTransContext,
                            const Surface& surface,
                            const DataSoA& soa, MaterialKey mk);

        MR_PF_DECL
        BxDFSample          SampleBxDF(const Vector3& wO,
                                       RNGDispenser& dispenser) const;
        MR_PF_DECL Float    Pdf(const Ray& wI, const Vector3& wO) const;

        MR_PF_DECL BxDFEval Evaluate(const Ray& wI, const Vector3& wO) const;
        MR_PF_DECL bool     IsEmissive() const;
        MR_PF_DECL Spectrum Emit(const Vector3& wO) const;
        MR_PF_DECL Float    Specularity() const;
        MR_PF_DECL
        RayConeSurface      RefractRayCone(const RayConeSurface&, const Vector3& wO) const;

        MR_PF_DECL
        static bool         IsAllTexturesAreResident(const Surface&, const DataSoA&,
                                                     MaterialKey);
    };
}

class MatGroupPassthrough final : public GenericGroupMaterial<MatGroupPassthrough>
{
    public:
    using DataSoA   = EmptyType;
    template<class STContext = SpectrumContextIdentity>
    using Material  = PassthroughMatDetail::PassthroughMaterial<STContext>;
    using Surface   = typename Material<>::Surface;

    private:
    protected:

    public:
    static std::string_view TypeName();

    MatGroupPassthrough(uint32_t groupId,
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

MR_PF_DEF
bool MaterialCommon::IsSpecular(Float specularity) noexcept
{
    constexpr auto Threshold = SpecularThreshold;
    return specularity >= Threshold;
}

namespace PassthroughMatDetail
{

template <class SC>
MR_GF_DEF
NormalMap PassthroughMaterial<SC>::GetNormalMap(const DataSoA&, MaterialKey)
{
    return std::nullopt;
}

template <class SC>
MR_PF_DEF_V
PassthroughMaterial<SC>::PassthroughMaterial(const SpectrumConverter&,
                                             const Surface& surface,
                                             const DataSoA&, MaterialKey)
    : surface(surface)
{}

template <class SC>
MR_PF_DEF
BxDFSample
PassthroughMaterial<SC>::SampleBxDF(const Vector3& wO,
                                    RNGDispenser&) const
{
    return BxDFSample
    {
        .wI     = Ray(-wO, surface.position),
        .pdf    = Float(1.0),
        .eval   = BxDFEval
        {
            .reflectance     = Spectrum(1.0),
            .isPassedThrough = true,
            .isDispersed     = false
        }
    };
}

template <class SC>
MR_PF_DEF
Float
PassthroughMaterial<SC>::Pdf(const Ray&, const Vector3&) const
{
    // We can not sample this
    return Float(0);
}

template <class SC>
MR_PF_DEF
BxDFEval PassthroughMaterial<SC>::Evaluate(const Ray&, const Vector3&) const
{
    return BxDFEval
    {
        .reflectance     = Spectrum(1),
        .isPassedThrough = true,
        .isDispersed     = false
    };
}

template <class SC>
MR_PF_DEF
bool PassthroughMaterial<SC>::IsEmissive() const
{
    return false;
}

template <class SC>
MR_PF_DEF
Spectrum PassthroughMaterial<SC>::Emit(const Vector3&) const
{
    return Spectrum::Zero();
}

template <class SC>
MR_PF_DEF
Float PassthroughMaterial<SC>::Specularity() const
{
    return Float(1);
}

template <class SC>
MR_PF_DEF
RayConeSurface PassthroughMaterial<SC>::RefractRayCone(const RayConeSurface& r,
                                                       const Vector3&) const
{
    return r;
}

template <class SC>
MR_PF_DEF
bool PassthroughMaterial<SC>::IsAllTexturesAreResident(const Surface&, const DataSoA&,
                                                       MaterialKey)
{
    return true;
}

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