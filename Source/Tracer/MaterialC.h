#pragma once

#include "Core/Types.h"
#include "Core/TracerI.h"

#include "TracerTypes.h"
#include "GenericGroup.h"
#include "ParamVaryingData.h"

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
    requires GenericGroupC<MGType>;
};


using GenericTextureView2D = Variant
<
    TextureView<2, Float>,
    TextureView<2, Vector2>,
    TextureView<2, Vector3>,
    TextureView<2, Vector4>,
    TextureView<2, Spectrum>
>;

using GenericTextureView3D = Variant
<
    TextureView<3, Float>,
    TextureView<3, Vector2>,
    TextureView<3, Vector3>,
    TextureView<3, Vector4>,
    TextureView<3, Spectrum>
>;

using MaterialTexture2DMap = std::map<TextureId, GenericTextureView2D>;

template<class Child>
class GenericGroupMaterial : public GenericGroupT<Child, MaterialKey, MatAttributeInfo>
{
    using Parent = GenericGroupT<Child, MaterialKey, MatAttributeInfo>;
    using typename Parent::IdList;


    protected:
    const MaterialTextureMap&   globalTextureViews;

    virtual void    HandleMediumPairs(const std::vector<Pair<MediumKey, MediumKey>>&) = 0;

    template<class T>
    void            GenericPushTex2DAttribute(Span<ParamVaryingData<2, T>>,
                                              //
                                              MaterialKey idStart, MaterialKey idEnd,
                                              uint32_t attributeIndex,
                                              TransientData,
                                              const std::vector<Optional<TextureView<2, T>>>&,
                                              const GPUQueue& queue);
    template<class T>
    void            GenericPushTex2DAttribute(Span<Optional<TextureView<2, T>>>,
                                              //
                                              MaterialKey idStart, MaterialKey idEnd,
                                              uint32_t attributeIndex,
                                              const std::vector<Optional<TextureView<2, T>>>&,
                                              const GPUQueue& queue);

    template<class T>
    void            GenericPushTex2DAttribute(Span<TextureView<2, T>>,
                                              //
                                              MaterialKey idStart, MaterialKey idEnd,
                                              uint32_t attributeIndex,
                                              const std::vector<TextureView<2, T>>&,
                                              const GPUQueue& queue);

    public:
    // Constructors & Destructor
                    GenericGroupMaterial(uint32_t groupId, const GPUSystem&,
                                         const MaterialTextureMap&,
                                         size_t allocationGranularity = 2_MiB,
                                         size_t initialReservartionSize = 4_MiB);

    // Swap the interfaces (old switcharoo)
    IdList            Reserve(const std::vector<AttributeCountList>&) override;
    virtual IdList    Reserve(const std::vector<AttributeCountList>&,
                              const std::vector<Pair<MediumKey, MediumKey>>&) = 0;


    // Textured Push Functions
    // TODO: Add more maybe?
    // Materials probably will use 2D texture only so...
    virtual void    PushTex2DAttribute(MaterialKey idStart, MaterialKey idEnd,
                                       uint32_t attributeIndex,
                                       TransientData,
                                       const std::vector<Optional<TextureId>>&,
                                       const GPUQueue& queue) = 0;
    virtual void    PushTex2DAttribute(MaterialKey idStart, MaterialKey idEnd,
                                       uint32_t attributeIndex,
                                       const std::vector<TextureId>&,
                                       const GPUQueue& queue) = 0;
};

template<class C>
GenericGroupMaterial<C>::GenericGroupMaterial(uint32_t groupId, const GPUSystem& gpuSystem,
                                              const MaterialTextureMap& map,
                                              size_t allocationGranularity,
                                              size_t initialReservartionSize)
    : Parent(groupId, gpuSystem,
             allocationGranularity,
             initialReservartionSize)
    , globalTextureViews(map)
{}

template<class C>
template<class T>
void GenericGroupMaterial<C>::GenericPushTex2DAttribute(Span<ParamVaryingData<2, T>> attributeSpan,
                                                        //
                                                        MaterialKey idStart, MaterialKey idEnd,
                                                        uint32_t attributeIndex,
                                                        TransientData hData,
                                                        const std::vector<Optional<TextureView<2, T>>>& hOptTextures,
                                                        const GPUQueue& queue)
{

    // Now we need to be careful
    auto rangeStart = this->itemRanges.at(idStart.FetchIndexPortion())[attributeIndex];
    auto rangeEnd = this->itemRanges.at(idStart.FetchIndexPortion())[attributeIndex];
    size_t count = rangeEnd[1] - rangeStart[0];
    Span<ParamVaryingData<2, T>> dSubspan = attributeSpan.subspan(rangeStart[0],
                                                                  count);

    // TODO: Use stream ordered memory allocator
    // This is blocking
    DeviceMemory tempMem;
    Span<Optional<TextureView<2, T>>> dTempTexViews;
    Span<T> dTempAttributeData;
    MemAlloc::AllocateMultiData(std::tie(dTempTexViews, dTempAttributeData),
                                tempMem, {count, count});

    Span<Optional<TextureView<2, T>>> hTempTexViews = hOptTextures;
    Span<T> hTempAttributeData = hData.AccessAs<T>();
    queue.MemcpyAsync(dTempTexViews, ToConstSpan(hTempTexViews));
    queue.MemcpyAsync(dTempAttributeData, ToConstSpan(hTempAttributeData));


    using namespace std::string_literals;
    static const std::string KernelName = "KCGenParamVaryingData\""s + std::string(C::TypeName()) + "\"";

    auto KCGenParamVaryingData = [dTempTexViews, dTempAttributeData, dSubspan, count]
    (KernelCallParams kp)
    {
        for(uint32_t i = kp.GlobalId(); i < count;
            i += kp.TotalSize())
        {
            dSubspan[i] = (dTempTexViews[i].has_value())
                                ? dTempTexViews[i].value()
                                : dTempAttributeData[i];
        }
    };

    queue.DeviceIssueSaturatingLambda
    (
        KernelName,
        KernelIssueParams{.workCount = count},
        //
        std::move(KCGenParamVaryingData)
    );
}

template<class C>
template<class T>
void GenericGroupMaterial<C>::GenericPushTex2DAttribute(Span<Optional<TextureView<2, T>>> dOptTexSpan,
                                                        //
                                                        MaterialKey idStart, MaterialKey idEnd,
                                                        uint32_t attributeIndex,
                                                        const std::vector<Optional<TextureView<2, T>>>& hOptTexViews,
                                                        const GPUQueue& queue)
{
    // YOLO memcpy!
}

template<class C>
template<class T>
void GenericGroupMaterial<C>::GenericPushTex2DAttribute(Span<TextureView<2, T>> dTexSpan,
                                                        //
                                                        MaterialKey idStart, MaterialKey idEnd,
                                                        uint32_t attributeIndex,
                                                        const std::vector<TextureView<2, T>>& hTexViews,
                                                        const GPUQueue& queue)
{
    // YOLO memcpy!
}

template<class C>
typename GenericGroupMaterial<C>::IdList
GenericGroupMaterial<C>::Reserve(const std::vector<AttributeCountList>&)
{
    throw MRayError("{}: Materials cannot be reserved via this function!",
                    C::TypeName());
}

template<class C>
typename GenericGroupMaterial<C>::IdList
GenericGroupMaterial<C>::Reserve(const std::vector<AttributeCountList>& countArrayList,
                                 const std::vector<Pair<MediumKey, MediumKey>>& mediumPairs)
{
    // We blocked the virutal chain, but we should be able to use it here
    // We will do the same here anyways migh as well use it.
    auto result = Parent:::Reserve(countArrayList);
    HandleMediumPairs(mediumPairs);
    return result;
}