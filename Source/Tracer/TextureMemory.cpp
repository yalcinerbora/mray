#include "TextureMemory.h"
#include "ColorConverter.h"

#include "Core/Error.hpp"
#include "Core/GraphicsFunctions.h"
#include "Core/Timer.h"
#include "Core/Vector.h"

#include "Device/GPUTexture.h"
#include "Device/GPUTextureView.h"

namespace TexDetail
{

template <typename T>
class Concept : public GenericTextureI
{
    private:
    T                   tex;
    public:
    template<class... Args>
                        Concept(Args&&...);

    void                CommitMemory(const GPUQueue& queue,
                                     const TextureBackingMemory& deviceMem,
                                     size_t offset) override;
    size_t              Size() const override;
    size_t              Alignment() const override;
    uint32_t            MipCount() const override;
    uint32_t            ChannelCount() const override;
    const GPUDevice&    Device() const override;
    //
    TextureExtent<3>    Extents() const override;
    uint32_t            DimensionCount() const override ;
    void                CopyFromAsync(const GPUQueue& queue,
                                      uint32_t mipLevel,
                                      const TextureExtent<3>& offset,
                                      const TextureExtent<3>& size,
                                      TransientData regionFrom) override;
    void                CopyFromAsync(const GPUQueue& queue,
                                      uint32_t mipLevel,
                                      const TextureExtent<3>& offset,
                                      const TextureExtent<3>& size,
                                      Span<const Byte> regionFrom) override;
    void                CopyToAsync(Span<Byte> regionTo,
                                    const GPUQueue& queue,
                                    uint32_t mipLevel,
                                    const TextureExtent<3>& offset,
                                    const TextureExtent<3>& size) const override;
    GenericTextureView  View(TextureReadMode mode) const override;
    bool                HasRWView() const override;
    SurfRefVariant      RWView(uint32_t mipLevel) override;
    bool                IsBlockCompressed() const override;
};

static_assert(GenericTexture::BuffSize == std::max({sizeof(Concept<Texture<3, Vector4>>),
                                                    sizeof(Concept<Texture<2, Vector4>>),
                                                    sizeof(Concept<Texture<2, PixelBC1>>)}),
              "Size of GenericTexture does not match, please change it to the exact value");
static_assert(GenericTexture::BuffAlignment == std::max({alignof(Concept<Texture<3, Vector4>>),
                                                         alignof(Concept<Texture<2, Vector4>>),
                                                         alignof(Concept<Texture<2, PixelBC1>>)}),
              "Alignment of GenericTexture does not match, please change it to the exact value");

template<class T>
template<class... Args>
Concept<T>::Concept(Args&&... args)
    : tex(std::forward<Args>(args)...)
{}

template<class T>
void Concept<T>::CommitMemory(const GPUQueue& queue,
                              const TextureBackingMemory& deviceMem,
                              size_t offset)
{
    tex.CommitMemory(queue, deviceMem, offset);
}

template<class T>
size_t Concept<T>::Size() const
{
    return tex.Size();
}

template<class T>
size_t Concept<T>::Alignment() const
{
    return tex.Alignment();
}

template<class T>
uint32_t Concept<T>::MipCount() const
{
    return tex.MipCount();
}

template<class T>
uint32_t Concept<T>::ChannelCount() const
{
    return T::ChannelCount;
}

template<class T>
const GPUDevice& Concept<T>::Device() const
{
    return tex.Device();
}

template<class T>
TextureExtent<3> Concept<T>::Extents() const
{
    auto ext = tex.Extents();
    if constexpr(T::Dims == 1)
        return TextureExtent<3>(ext, 0, 0);
    else if constexpr(T::Dims == 2)
        return TextureExtent<3>(ext[0], ext[1], 0);
    else return ext;
}

template<class T>
uint32_t Concept<T>::DimensionCount() const
{
    return T::Dims;
}

template<class T>
void Concept<T>::CopyFromAsync(const GPUQueue& queue,
                               uint32_t mipLevel,
                               const TextureExtent<3>& offset,
                               const TextureExtent<3>& size,
                               TransientData regionFrom)
{
    using ExtType = TextureExtent<T::Dims>;
    ExtType offsetIn;
    ExtType sizeIn;

    if constexpr(T::Dims == 1)
    {
        offsetIn = ExtType(offset[0]);
        sizeIn = ExtType(size[0]);
    }
    else
    {
        offsetIn = ExtType(offset);
        sizeIn = ExtType(size);
    }

    using PaddedChannelType = typename T::PaddedChannelType;
    auto input = regionFrom.AccessAs<const PaddedChannelType>();
    tex.CopyFromAsync(queue, mipLevel, offsetIn, sizeIn, input);
    queue.IssueBufferForDestruction(std::move(regionFrom));
}

template<class T>
void Concept<T>::CopyFromAsync(const GPUQueue& queue,
                               uint32_t mipLevel,
                               const TextureExtent<3>& offset,
                               const TextureExtent<3>& size,
                               Span<const Byte> regionFrom)
{
    using ExtType = TextureExtent<T::Dims>;
    ExtType offsetIn;
    ExtType sizeIn;

    if constexpr(T::Dims == 1)
    {
        offsetIn = ExtType(offset[0]);
        sizeIn = ExtType(size[0]);
    }
    else
    {
        offsetIn = ExtType(offset);
        sizeIn = ExtType(size);
    }

    using PaddedChannelType = typename T::PaddedChannelType;
    assert(regionFrom.size_bytes() % sizeof(PaddedChannelType) == 0);
    using MemAlloc::RepurposeAlloc;
    auto input = RepurposeAlloc<const PaddedChannelType>(regionFrom);
    tex.CopyFromAsync(queue, mipLevel, offsetIn, sizeIn, input);
}

template<class T>
void Concept<T>::CopyToAsync(Span<Byte> regionTo,
                             const GPUQueue& queue,
                             uint32_t mipLevel,
                             const TextureExtent<3>& offset,
                             const TextureExtent<3>& size) const
{
    // TODO: Change this later, only BC textures have this functionality
    if constexpr(T::IsBlockCompressed)
    {
        using ExtType = TextureExtent<T::Dims>;
        ExtType offsetIn;
        ExtType sizeIn;

        if constexpr(T::Dims == 1)
        {
            offsetIn = ExtType(offset[0]);
            sizeIn = ExtType(size[0]);
        }
        else
        {
            offsetIn = ExtType(offset);
            sizeIn = ExtType(size);
        }

        using PaddedChannelType = typename T::PaddedChannelType;
        assert(regionTo.size_bytes() % sizeof(PaddedChannelType) == 0);
        using MemAlloc::RepurposeAlloc;
        Span<PaddedChannelType> output = RepurposeAlloc<PaddedChannelType>(regionTo);
        tex.CopyToAsync(output, queue, mipLevel, offsetIn, sizeIn);
    }
    else throw MRayError("Non-bc texture's data cannot be copied!");
}

template<class T>
GenericTextureView Concept<T>::View(TextureReadMode mode) const
{
    using enum TextureReadMode;
    static constexpr uint32_t C = T::ChannelCount;
    static constexpr uint32_t D = T::Dims;
    static constexpr bool IsBC = T::IsBlockCompressed;
    auto GenView = [this]<class X>(TextureReadMode mode,
                                   // For overload resolution
                                   std::in_place_type_t<X>)
    {
        if constexpr(C == 1)
            return TracerTexView<D, X>(tex.template View<Float>(), mode, IsBC);
        else if constexpr(C == 2)
            return TracerTexView<D, X>(tex.template View<Vector2>(), mode, IsBC);
        else if constexpr(C == 3)
            return TracerTexView<D, X>(tex.template View<Vector3>(), mode, IsBC);
        else
            return TracerTexView<D, X>(tex.template View<Vector4>(), mode, IsBC);
    };

    switch(mode)
    {
        case TO_3C_BASIC_NORMAL_FROM_SIGNED_2C:
        case TO_3C_OCTA_NORMAL_FROM_SIGNED_2C:
        case TO_3C_BASIC_NORMAL_FROM_UNSIGNED_2C:
        case TO_3C_OCTA_NORMAL_FROM_UNSIGNED_2C:
        case TO_3C_FROM_4C:
            return GenView(mode, std::in_place_type_t<Vector3>{});
        case TO_2C_FROM_4C:
        case TO_2C_FROM_3C:
            return GenView(mode, std::in_place_type_t<Vector2>{});
        case TO_1C_FROM_4C:
        case TO_1C_FROM_3C:
        case TO_1C_FROM_2C:
            return GenView(mode, std::in_place_type_t<Float>{});
        case DIRECT:
        default:
        {
            if constexpr(C == 1)
                return TracerTexView<D, Float>(tex.template View<Float>(), mode, IsBC);
            else if constexpr(C == 2)
                return TracerTexView<D, Vector2>(tex.template View<Vector2>(), mode, IsBC);
            else if constexpr(C == 3)
                return TracerTexView<D, Vector3>(tex.template View<Vector3>(), mode, IsBC);
            else
                return TracerTexView<D, Vector4>(tex.template View<Vector4>(), mode, IsBC);
        }
    }
}

template<class T>
bool Concept<T>::IsBlockCompressed() const
{
    // Only 2D textures that are not block
    // compressed are supported
    return T::IsBlockCompressed;
}

template<class T>
bool Concept<T>::HasRWView() const
{
    // Only 2D textures that are not block
    // compressed are supported
    return T::Dims == 2 && !T::IsBlockCompressed;
}

template<class T>
SurfRefVariant Concept<T>::RWView(uint32_t mipLevel)
{
    // If not supported return monostate
    if constexpr(T::Dims != 2 || T::IsBlockCompressed)
        return std::monostate{};
    else
        return tex.GenerateRWRef(mipLevel);
}

}

template<class T, class... Args>
inline
GenericTexture::GenericTexture(std::in_place_type_t<T>,
                               MRayColorSpaceEnum cs, Float gammaIn,
                               AttributeIsColor col, MRayPixelTypeRT pt,
                               Args&&... args)
    : impl(nullptr)
    , pixelType(pt)
    , colorSpace(cs)
    , gamma(gammaIn)
    , isColor(col)
    , isMipLoaded(false)
{
    isMipLoaded.Reset();
    using ConceptType = TexDetail::Concept<T>;
    static_assert(sizeof(ConceptType) <= BuffSize,
                  "Unable construct type over storage!");
    ConceptType* ptr = reinterpret_cast<ConceptType*>(storage.data());
    impl = std::construct_at(ptr, std::forward<Args>(args)...);
}

TextureReadMode DetermineReadMode(MRayPixelTypeRT pixelType,
                                  MRayTextureReadMode rm)
{
    size_t channelCount = pixelType.ChannelCount();
    bool isSigned = pixelType.ChannelCount();
    using enum MRayTextureReadMode;
    switch(rm)
    {
        case MR_PASSTHROUGH: return TextureReadMode::DIRECT;
        //
        case MR_AS_3C_TS_NORMAL_BASIC:
        {
            if(channelCount != 2)
            {
                throw MRayError("Non 2-channel texture cannot be used "
                                "for normal expansion");
            }
            if(isSigned)
                return TextureReadMode::TO_3C_BASIC_NORMAL_FROM_SIGNED_2C;
            else
                return TextureReadMode::TO_3C_BASIC_NORMAL_FROM_UNSIGNED_2C;
        }
        case MR_AS_3C_TS_NORMAL_COOCTA:
        {
            if(channelCount != 2)
            {
                throw MRayError("Non 2-channel texture cannot be used "
                                "for normal expansion");
            }
            if(isSigned)
                return TextureReadMode::TO_3C_OCTA_NORMAL_FROM_SIGNED_2C;
            else
                return TextureReadMode::TO_3C_OCTA_NORMAL_FROM_UNSIGNED_2C;
        }
        case MR_DROP_1:
        {
            if(channelCount <= 1)
                throw MRayError("Not enough channels to drop 1 channel");
            else if(channelCount == 2)
                return TextureReadMode::TO_1C_FROM_2C;
            else if(channelCount == 3)
                return TextureReadMode::TO_2C_FROM_3C;
            else if(channelCount == 4)
                return TextureReadMode::TO_3C_FROM_4C;
            [[fallthrough]];
        }
        case MR_DROP_2:
        {
            if(channelCount <= 2)
                throw MRayError("Not enough channels to drop 2 channels");
            else if(channelCount == 3)
                return TextureReadMode::TO_1C_FROM_3C;
            else if(channelCount == 4)
                return TextureReadMode::TO_2C_FROM_4C;
            [[fallthrough]];
        }
        case MR_DROP_3:
        {
            if(channelCount <= 3)
                throw MRayError("Not enough channels to drop 3 channels");
            else if(channelCount == 4)
                return TextureReadMode::TO_1C_FROM_4C;
            [[fallthrough]];
        }
        default: break;
    }
    return TextureReadMode::DIRECT;
}

bool FindNormIntegers(MRayPixelTypeRT pixelType)
{
    return std::visit([](auto&& v) -> bool
    {
        using VisitedType = std::remove_cvref_t<decltype(v)>;
        constexpr auto Enum = VisitedType::Name;
        using PixType = typename VisitedType::Type;
        using enum MRayPixelEnum;
        if constexpr(Enum == MRayPixelEnum::MR_BC6H_SFLOAT ||
                     Enum == MRayPixelEnum::MR_BC6H_UFLOAT)
            return false;
        else if constexpr(VisitedType::IsBCPixel)
            return true;
        else if constexpr(VectorC<PixType>)
            return std::is_integral_v<typename PixType::InnerType>;
        else
            return std::is_integral_v<PixType>;
    }, pixelType);
}

void TextureMemory::ConvertColorspaces()
{
    ColorConverter colorConv(gpuSystem);
    std::vector<MipArray<SurfRefVariant>> texSurfs;
    std::vector<ColorConvParams> colorConvParams;
    std::vector<GenericTexture*> bcTextures;

    bcTextures.reserve(textures.GetMap().size());
    texSurfs.reserve(textures.GetMap().size());
    colorConvParams.reserve(textures.GetMap().size());
    // Load to linear memory
    for(auto& [_, t] : textures.GetMap())
    {
        bool isColor = (t.IsColor() == AttributeIsColor::IS_COLOR);
        bool requiresConversion = isColor;
        bool rq = ((t.ColorSpace() != MRayColorSpaceEnum::MR_DEFAULT &&
                    t.ColorSpace() != tracerParams.globalTextureColorSpace) ||
                    t.Gamma() != Float(1));
        requiresConversion &= rq;
        // Skip if not color or writable
        if(!requiresConversion)
        {
            t.SetColorSpace(tracerParams.globalTextureColorSpace);
            continue;
        }

        if(!t.IsBlockCompressed())
        {
            uint8_t mipCount = static_cast<uint8_t>(t.MipCount());
            ColorConvParams p =
            {
                .validMips = t.ValidMips(),
                .mipCount = mipCount,
                .fromColorSpace = t.ColorSpace(),
                .gamma = t.Gamma(),
                .mipZeroRes = Vector2ui(t.Extents())
            };
            colorConvParams.push_back(p);

            MipArray<SurfRefVariant> mips;
            for(uint8_t i = 0; i < p.mipCount; i++)
                mips[i] = t.RWView(i);

            texSurfs.push_back(std::move(mips));
            t.SetColorSpace(tracerParams.globalTextureColorSpace);
        }
        else bcTextures.push_back(&t);

    }

    // Finally call the kernel
    if(!texSurfs.empty() || !bcTextures.empty())
        colorConv.ConvertColor(std::move(texSurfs), std::move(colorConvParams),
                               std::move(bcTextures),
                               tracerParams.globalTextureColorSpace);
}

void TextureMemory::GenerateMipmaps()
{
    std::vector<MipArray<SurfRefVariant>> texSurfs;
    std::vector<MipGenParams> mipGenParams;
    texSurfs.reserve(textures.GetMap().size());
    mipGenParams.reserve(textures.GetMap().size());
    // Load to linear memory
    for(auto& [_, t] : textures.GetMap())
    {
        if(!t.HasRWView()) continue;

        MipGenParams p =
        {
            .validMips = t.ValidMips(),
            .mipCount = static_cast<uint16_t>(t.MipCount()),
            .mipZeroRes = Vector2ui(t.Extents())
        };
        mipGenParams.push_back(p);

        MipArray<SurfRefVariant> mips;
        for(uint16_t i = 0; i < p.mipCount; i++)
            mips[i] = t.RWView(i);

        texSurfs.push_back(std::move(mips));

        t.SetAllMipsToLoaded();
    }
    // Finally call the kernel
    if(!texSurfs.empty())
        mipGenFilter->GenerateMips(texSurfs, mipGenParams);
}

template<uint32_t D>
TextureId TextureMemory::CreateTexture(const Vector<D, uint32_t>& size, uint32_t mipCount,
                                       const MRayTextureParameters& inputParams)
{
    // BC textures can not be written efficiently (or at all?).
    // Since it requires function minimization etc. So some functionality
    // will be skipped for BC textures. For example, clamping texture
    // resolutions will only happen if there is a mip available for that level
    // if not one level higher (untill a valid mip is reached) will be used.
    bool isBlockCompressed = inputParams.pixelType.IsBlockCompressed();

    // Round robin deploy textures
    // TODO: Only load to single GPU currently
    uint32_t gpuIndex = gpuIndexCounter.fetch_add(1);
    gpuIndex %= 1;// gpuSystem.SystemDevices().size();
    const GPUDevice& device = gpuSystem.SystemDevices()[gpuIndex];

    TexClampParameters tClampParams = {Vector2ui(size), 0, 0, false, SurfRefVariant()};
    uint32_t newMipCount = mipCount;
    Vector<D, uint32_t> newSize = size;
    if constexpr(D == 2)
    {

        // Here we need to clamp the resolution of the texture
        // if requested.
        uint32_t maxDim = size[size.Maximum()];
        uint32_t clampCheck = inputParams.ignoreResClamp ? maxDim : tracerParams.clampedTexRes;
        uint32_t clampRes = std::min(clampCheck, maxDim);

        uint32_t ratio = Math::DivideUp(maxDim, clampRes);
        int32_t mipReduceAmount = int32_t(std::ceil(std::log2(ratio)));
        // We will have atleast one mip
        int32_t mipCountI = std::max(1, int32_t(mipCount) - mipReduceAmount);

        // For BC textures, find highest available mip level
        // (We can not generate these)
        // For other textures we can generate mips,
        // so directly generate these
        mipReduceAmount = (isBlockCompressed)
            ? std::min(mipCountI - 1, mipReduceAmount)
            : mipReduceAmount;

        // Finally the new size
        newSize = Graphics::TextureMipSize(size, uint32_t(mipReduceAmount));
        //
        uint32_t ignoredMipCount = uint32_t(mipReduceAmount);
        uint32_t filteredMip = std::min(ignoredMipCount, mipCount - 1);
        bool willBeFiltered = (ignoredMipCount > (mipCount - 1));
        // Store the last ignored mip (we will use this to
        // filtering during load time).
        if(willBeFiltered)
        {
            auto filterInPixCount = Graphics::TextureMipSize(size, uint32_t(filteredMip));
            size_t total = filterInPixCount.Multiply() * inputParams.pixelType.PaddedPixelSize();
            texClampBufferSize = std::max(texClampBufferSize, total);
        }
        // Create the clamp params
        tClampParams = TexClampParameters
        {
            .inputMaxRes = size,
            .filteredMipLevel = uint16_t(filteredMip),
            .ignoredMipCount = uint16_t(ignoredMipCount),
            .willBeFiltered = willBeFiltered,
            .surface = SurfRefVariant()
        };
        newMipCount = uint32_t(mipCountI);

        // Expand mip count if generate mipmaps is requested
        if(tracerParams.genMips && !isBlockCompressed)
        {
            newMipCount = Graphics::TextureMipCount(newSize);
        }
    }

    TextureInitParams<D> p;
    p.size = newSize;
    p.mipCount = newMipCount;
    p.eResolve = inputParams.edgeResolve;
    p.interp = inputParams.interpolation;
    p.normIntegers = FindNormIntegers(inputParams.pixelType);
    TextureId texId = std::visit([&](auto&& v) -> TextureId
    {
        using enum MRayPixelEnum;
        using ArgType = std::remove_cvref_t<decltype(v)>;
        using Type = typename ArgType::Type;
        if constexpr(D != 3 || !IsBlockCompressedPixel<Type>)
        {
            TextureId id = TextureId(texCounter.fetch_add(1));
            auto loc = textures.try_emplace(id, std::in_place_type_t<Texture<D, Type>>{},
                                            inputParams.colorSpace, inputParams.gamma,
                                            inputParams.isColor,
                                            inputParams.pixelType, device, p);

            TextureReadMode rm = DetermineReadMode(inputParams.pixelType,
                                                   inputParams.readMode);
            textureViews.try_emplace(id, loc.first->second.View(rm));
            // Save the clamp parameters as well
            texClampParams.try_emplace(id, std::move(tClampParams));

            return id;
        }
        else throw MRayError("3D Block compressed textures are not supported!");
    }, inputParams.pixelType);

    return texId;
}

TextureMemory::TextureMemory(const GPUSystem& sys,
                             const TracerParameters& tParams,
                             const FilterGeneratorMap& fGenMap)
    : gpuSystem(sys)
    , tracerParams(tParams)
    , texClampBuffer(gpuSystem.BestDevice())
{
    if(gpuSystem.SystemDevices().size() != 1)
        MRAY_WARNING_LOG("Textures will be loaded on to single GPU!");

    // Create texture memory for each device (not allocated yet)
    for(const GPUDevice& device : gpuSystem.SystemDevices())
        texMemList.emplace_back(device);

    FilterType::E filterType = tracerParams.mipGenFilter.type;
    auto fGen = fGenMap.at(filterType);
    if(!fGen.has_value())
    {
        throw MRayError("Unable to find a filter for type {}",
                        FilterType::ToString(filterType));
    }
    Float radius = tracerParams.mipGenFilter.radius;
    mipGenFilter = fGen.value().get()(gpuSystem, std::move(radius));
}

TextureId TextureMemory::CreateTexture2D(const Vector2ui& size, uint32_t mipCount,
                                         const MRayTextureParameters& p)
{
    return CreateTexture(size, mipCount, p);
}

TextureId TextureMemory::CreateTexture3D(const Vector3ui& size, uint32_t mipCount,
                                         const MRayTextureParameters& p)
{
    return CreateTexture(size, mipCount, p);
}

void TextureMemory::CommitTextures()
{
    // Linearize per-gpu textures
    auto& texMap = textures.GetMap();
    size_t gpuCount = gpuSystem.AllGPUs().size();
    size_t totalTexCount = texMap.size();

    std::vector<std::vector<size_t>> texSizes(gpuCount);
    std::vector<std::vector<size_t>> texAlignments(gpuCount);
    std::vector< std::vector<GenericTexture*>> texPtrs(gpuCount);
    assert(texMemList.size() == gpuCount);
    // Reserve for worst case
    for(size_t i = 0; i < gpuCount; i++)
    {
        texSizes[i].reserve(totalTexCount);
        texAlignments.reserve(totalTexCount);
        texPtrs.reserve(totalTexCount);
    }

    // Linearize the values for allocation
    for(auto& kv : texMap)
    {
        GenericTexture& tex = kv.second;
        size_t gpuIndex = static_cast<size_t>(std::distance(gpuSystem.SystemDevices().data(),
                                                            &tex.Device()));
        texPtrs[gpuIndex].push_back(&tex);
        texSizes[gpuIndex].push_back(tex.Size());
        texAlignments[gpuIndex].push_back(tex.Alignment());
    }

    // Allocate
    std::vector<std::vector<size_t>> offsets(gpuCount);
    for(size_t i = 0; i < texMemList.size(); i++)
    {
        if(texSizes[i].size() == 0) continue;

        using namespace MemAlloc;
        std::vector<size_t> offsetList = AllocateTextureSpace(texMemList[i],
                                                              texSizes[i],
                                                              texAlignments[i]);
        offsets[i] = std::move(offsetList);
    }

    // Attach the memory
    for(size_t i = 0; i < texMemList.size(); i++)
    {
        const GPUDevice& currentDevice = gpuSystem.SystemDevices()[i];
        uint32_t queueIndex = 0;
        for(size_t j = 0; j < texSizes[i].size(); j++)
        {
            const GPUQueue& queue = currentDevice.GetComputeQueue(queueIndex);
            texPtrs[i][j]->CommitMemory(queue, texMemList[i], offsets[i][j]);
        }
        queueIndex++;
        queueIndex %= TotalQueuePerDevice();
    }

    // Allocate the filter buffer now, texture data will come
    // now
    if(texClampBufferSize)
        texClampBuffer.ResizeBuffer(texClampBufferSize);

}

void TextureMemory::PushTextureData(TextureId id, uint32_t mipLevel,
                                    TransientData data)
{
    assert(data.IsFull());
    auto texLoc = textures.at(id);
    if(!texLoc)
    {
        throw MRayError("Unable to find texture(id)",
                        static_cast<CommonKey>(id));
    }
    GenericTexture& tex = texLoc.value().get();
    auto clampLoc = texClampParams.at(id);
    TexClampParameters& clampParams = clampLoc.value().get();

    // TODO: Again multi-gpu/queue management
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    const GPUQueue& texQueue = tex.Device().GetComputeQueue(0);

    // If texture mip level
    assert((clampParams.willBeFiltered && (mipLevel <= clampParams.filteredMipLevel))
           || (!clampParams.willBeFiltered));

    if(clampParams.willBeFiltered &&
       mipLevel < clampParams.filteredMipLevel) return;

    if(clampParams.willBeFiltered &&
       mipLevel == clampParams.filteredMipLevel)
    {
        Vector2ui size = Graphics::TextureMipSize(clampParams.inputMaxRes, mipLevel);
        auto texClampBufferSpan = Span<Byte>(static_cast<Byte*>(texClampBuffer), texClampBuffer.Size());
        auto dataSpan = ToConstSpan(data.AccessAs<Byte>());

        // Utilize the TexClampParamters
        clampParams.surface = tex.RWView(0);

        tex.SetMipToLoaded(mipLevel);

        // Order is important here
        // Multiple threads can call this function
        // So we need to lock the queue issue
        std::lock_guard _{texClampMutex};

        // Fist copy to staging device buffer
        queue.MemcpyAsync(texClampBufferSpan, dataSpan);
        // Issue the filter
        mipGenFilter->ClampImageFromBuffer(clampParams.surface,
                                           ToConstSpan(texClampBufferSpan),
                                           Vector2ui(tex.Extents()), size,
                                           queue);
        // Defer delete the transient host buffer
        queue.IssueBufferForDestruction(std::move(data));
        // We dont have the wait the queue here since we always issue to the same queue
        // so device queue semantics will handle the buffer usage
    }
    else if(mipLevel < tex.MipCount())
    {
        // Nothing fancy just copy the data to buffer
        Vector3ui mipSize = Graphics::TextureMipSize(tex.Extents(), mipLevel);
        tex.CopyFromAsync(texQueue,
                          mipLevel,
                          Vector3ui::Zero(),
                          mipSize,
                          std::move(data));
    }
}

MRayPixelTypeRT TextureMemory::GetPixelType(TextureId id) const
{
    auto loc = textures.at(id);
    if(!loc)
    {
        throw MRayError("Unable to find texture(id)",
                        static_cast<CommonKey>(id));
    }
    const GenericTexture& tex = loc.value().get();
    return tex.PixelType();
}

void TextureMemory::Finalize()
{
    // Clear the clamp buffer
    if(texClampBufferSize > 0)
        texClampBuffer = DeviceLocalMemory(gpuSystem.BestDevice());
    // Destroy the surface objects due to clamping op
    for(auto&[_, cp] : texClampParams.GetMap())
        cp.surface = SurfRefVariant();

    Timer t;
    t.Start();
    MRAY_LOG("[Tracer]: Converting textures to global color space ...");
    ConvertColorspaces();
    t.Lap();
    MRAY_LOG("[Tracer]: Texture color conversion took {}ms.",
             t.Elapsed<Millisecond>());

    if(tracerParams.genMips)
    {
        MRAY_LOG("[Tracer]: Generating texture mipmaps...");
        GenerateMipmaps();
        t.Split();
        MRAY_LOG("[Tracer]: Texture mipmap generation took {}ms.",
                 t.Elapsed<Millisecond>());
    }
}

const TextureViewMap& TextureMemory::TextureViews() const
{
    return textureViews.GetMap();
}

const TextureMap& TextureMemory::Textures() const
{
    return textures.GetMap();
}

void TextureMemory::Clear()
{
    texCounter = 0;
    gpuIndexCounter = 0;
    texMemList.clear();
    textures.clear();
    textureViews.clear();

    // Create texture memory for each device (not allocated yet)
    for(const GPUDevice& device : gpuSystem.SystemDevices())
        texMemList.emplace_back(device);
}

size_t TextureMemory::GPUMemoryUsage() const
{
    size_t total = 0;
    for(const auto& mem : texMemList)
    {
        total += mem.Size();
    }
    return total;
}