#include "TextureMemory.h"
#include "CommonTexture.hpp"

#include "Core/Error.hpp"
#include "Core/GraphicsFunctions.h"
#include "Core/Timer.h"

namespace TexDetail
{

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

    auto ext = tex.Extents();
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
}

template<class T>
GenericTextureView Concept<T>::View() const
{
    static constexpr uint32_t ChannelCount = T::ChannelCount;

    if constexpr(ChannelCount == 1)
        return tex.View<Float>();
    else if constexpr(ChannelCount == 2)
        return tex.View<Vector2>();
    else if constexpr(ChannelCount == 3)
        return tex.View<Vector3>();
    else
        return tex.View<Vector4>();
}

template<class T>
bool Concept<T>::HasRWView() const
{
    // Only 2D and non block compressed formats are supported
    return T::Dims == 2 && !T::IsBlockCompressed;
}

template<class T>
SurfRefVariant Concept<T>::RWView(uint32_t mipLevel)
{
    // If non supported texture's view is requested return
    // monostate
    if constexpr(T::Dims != 2 || T::IsBlockCompressed)
        return std::monostate{};
    else
    {
        return tex.GenerateRWRef(mipLevel);
    }
}

}

void TextureMemory::ConvertColorspaces()
{

}

void TextureMemory::GenerateMipmaps()
{
    using TextureFilterPtr = std::unique_ptr<TextureFilterI>;
    FilterType::E filterType = tracerParams.mipGenFilter.type;
    auto fGen = filterGenMap.at(filterType);
    if(!fGen.has_value())
    {
        throw MRayError("Unable to find a filter for type {}",
                        FilterType::ToString(filterType));
    }
    Float radius = tracerParams.mipGenFilter.radius;
    TextureFilterPtr texFilter = fGen.value().get()(gpuSystem, std::move(radius));

    std::vector<MipArray<SurfRefVariant>> texSurfs;
    std::vector<MipGenParams> mipGenParams;
    texSurfs.reserve(textures.Map().size());
    mipGenParams.reserve(textures.Map().size());
    // Load to linear memory
    for(auto& [_, t] : textures.Map())
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
    texFilter->GenerateMips(texSurfs, mipGenParams);
}

template<uint32_t D>
TextureId TextureMemory::CreateTexture(const Vector<D, uint32_t>& size, uint32_t mipCount,
                                       const MRayTextureParameters& inputParams)
{
    // Round robin deploy textures
    // TODO: Only load to single GPU currently
    uint32_t gpuIndex = gpuIndexCounter.fetch_add(1);
    gpuIndex %= 1;// gpuSystem.SystemDevices().size();
    const GPUDevice& device = gpuSystem.SystemDevices()[gpuIndex];

    // Expand mip count if generate mipmaps is requested
    if(tracerParams.genMips)
        mipCount = Graphics::TextureMipCount(size);

    TextureInitParams<D> p;
    p.size = size;
    p.mipCount = mipCount;
    p.eResolve = inputParams.edgeResolve;
    p.interp = inputParams.interpolation;
    TextureId texId = std::visit([&](auto&& v) -> TextureId
    {
        using enum MRayPixelEnum;
        using ArgType = std::remove_cvref_t<decltype(v)>;
        using Type = typename ArgType::Type;
        if constexpr(D != 3 || !IsBlockCompressedPixel<Type>)
        {
            TextureId id = TextureId(texCounter.fetch_add(1));
            auto loc = textures.try_emplace(id, std::in_place_type_t<Texture<D, Type>>{},
                                            inputParams.colorSpace, inputParams.isColor,
                                            MRayPixelTypeRT(v),
                                            device, p);

            textureViews.try_emplace(id, loc.first->second.View());
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
    , filterGenMap(fGenMap)
{
    if(gpuSystem.SystemDevices().size() != 1)
        MRAY_WARNING_LOG("Textures will be loaded on to single GPU!");

    // Create texture memory for each device (not allocated yet)
    for(const GPUDevice& device : gpuSystem.SystemDevices())
        texMemList.emplace_back(device);
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
    auto& texMap = textures.Map();
    size_t gpuCount = gpuSystem.AllGPUs().size();
    size_t totalTexCount = texMap.size();

    std::vector<std::vector<size_t>> texSizes(gpuCount);
    std::vector<std::vector<size_t>> texAlignments(gpuCount);
    std::vector< std::vector<CommonTexture*>> texPtrs(gpuCount);
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
        CommonTexture& tex = kv.second;
        size_t gpuIndex = std::distance(gpuSystem.SystemDevices().data(),
                                        &tex.Device());
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
}
void TextureMemory::PushTextureData(TextureId id, uint32_t mipLevel,
                                    TransientData data)
{
    auto loc = textures.at(id);
    if(!loc)
    {
        throw MRayError("Unable to find texture(id)",
                        static_cast<uint32_t>(id));
    }
    CommonTexture& tex = loc.value().get();

    // TODO: Again multi-gpu/queue management
    const GPUQueue& queue = tex.Device().GetComputeQueue(0);
    Vector3ui mipSize = Graphics::TextureMipSize(tex.Extents(), mipLevel);
    tex.CopyFromAsync(queue,
                      mipLevel,
                      Vector3ui::Zero(),
                      mipSize,
                      std::move(data));
}

MRayPixelTypeRT TextureMemory::GetPixelType(TextureId id) const
{
    auto loc = textures.at(id);
    if(!loc)
    {
        throw MRayError("Unable to find texture(id)",
                        static_cast<uint32_t>(id));
    }
    const CommonTexture& tex = loc.value().get();
    return tex.PixelType();
}

void TextureMemory::Finalize()
{
    Timer t;
    t.Start();
    MRAY_LOG("[Tracer] Converting textures to global color space ...");

    t.Lap();
    MRAY_LOG("[Tracer] Texture color conversion took {}ms.",
             t.Elapsed<Millisecond>());

    if(tracerParams.genMips)
    {
        MRAY_LOG("[Tracer] Generating texture mipmaps...");
        GenerateMipmaps();
        t.Split();
        MRAY_LOG("[Tracer] Texture mipmap generation took {}ms.",
                 t.Elapsed<Millisecond>());
    }
}

const TextureViewMap& TextureMemory::TextureViews() const
{
    return textureViews.Map();
}

const TextureMap& TextureMemory::Textures() const
{
    return textures.Map();
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