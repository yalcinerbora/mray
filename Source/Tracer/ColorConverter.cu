#include "ColorConverter.h"
#include "GenericTextureRW.h"
#include "BCColorIO.h"
#include "TextureMemory.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

#include "Core/ColorFunctions.h"
#include "Core/GraphicsFunctions.h"

using MipBlockCountList = StaticVector<uint64_t, TracerConstants::MaxTextureMipCount>;

static constexpr uint32_t BC_TEX_PER_BATCH = 16;
struct BCColorConvParams
{
    Vector2ul           blockRange;
    Float               gamma;
    MRayColorSpaceEnum  fromColorSpace;
};
using BCColorConvParamList = std::array<BCColorConvParams, BC_TEX_PER_BATCH>;

// Order is important here
template <MRayColorSpaceEnum E>
using ConverterList = Tuple
<
    Color::ColorspaceTransfer<MRayColorSpaceEnum::MR_ACES2065_1,    E>,
    Color::ColorspaceTransfer<MRayColorSpaceEnum::MR_ACES_CG,       E>,
    Color::ColorspaceTransfer<MRayColorSpaceEnum::MR_REC_709,       E>,
    Color::ColorspaceTransfer<MRayColorSpaceEnum::MR_REC_2020,      E>,
    Color::ColorspaceTransfer<MRayColorSpaceEnum::MR_DCI_P3,        E>,
    Color::ColorspaceTransfer<MRayColorSpaceEnum::MR_ADOBE_RGB,     E>
>;

static constexpr Tuple ConverterLists =
{::
    ConverterList<MRayColorSpaceEnum::MR_ACES2065_1>{},
    ConverterList<MRayColorSpaceEnum::MR_ACES_CG>{},
    ConverterList<MRayColorSpaceEnum::MR_REC_709>{},
    ConverterList<MRayColorSpaceEnum::MR_REC_2020>{},
    ConverterList<MRayColorSpaceEnum::MR_DCI_P3>{},
    ConverterList<MRayColorSpaceEnum::MR_ADOBE_RGB>{}
};

using BCTypeMap = TypeFinder::E_TMapper<MRayPixelEnum>;
using BCReaderFinder = BCTypeMap::Map
<
    typename BCTypeMap::template ETPair<MRayPixelEnum::MR_BC1_UNORM,    BlockCompressedIO::BC1>,
    typename BCTypeMap::template ETPair<MRayPixelEnum::MR_BC2_UNORM,    BlockCompressedIO::BC2>,
    typename BCTypeMap::template ETPair<MRayPixelEnum::MR_BC3_UNORM,    BlockCompressedIO::BC3>,
    typename BCTypeMap::template ETPair<MRayPixelEnum::MR_BC4_UNORM,    BlockCompressedIO::BC4<false>>,
    typename BCTypeMap::template ETPair<MRayPixelEnum::MR_BC4_SNORM,    BlockCompressedIO::BC4<true>>,
    typename BCTypeMap::template ETPair<MRayPixelEnum::MR_BC5_UNORM,    BlockCompressedIO::BC5<false>>,
    typename BCTypeMap::template ETPair<MRayPixelEnum::MR_BC5_SNORM,    BlockCompressedIO::BC5<true>>,
    typename BCTypeMap::template ETPair<MRayPixelEnum::MR_BC6H_UFLOAT,  BlockCompressedIO::BC6H<false>>,
    typename BCTypeMap::template ETPair<MRayPixelEnum::MR_BC6H_SFLOAT,  BlockCompressedIO::BC6H<true>>,
    typename BCTypeMap::template ETPair<MRayPixelEnum::MR_BC7_UNORM,    BlockCompressedIO::BC7>
>;

template<uint32_t TPB, class BCReader, class ConverterTuple>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCConvertColorBC(// I-O
                      MRAY_GRID_CONSTANT const Span<typename BCReader::BlockType>,
                      // Inputs
                      MRAY_GRID_CONSTANT const BCColorConvParamList,
                      // Constants
                      MRAY_GRID_CONSTANT const uint32_t,
                      MRAY_GRID_CONSTANT const MRayColorSpaceEnum,
                      MRAY_GRID_CONSTANT const uint32_t);

MRAY_GPU MRAY_GPU_INLINE
Vector3 GenericFromNorm(const Vector3& t, const SurfViewVariant& surfView)
{
    using namespace Bit::NormConversion;
    // Utilize the surfView variant to get the type etc
    return DeviceVisit(surfView, [t](auto&& sv) -> Vector3
    {
        Vector3 result = t;
        using VariantType = std::remove_cvref_t<decltype(sv)>;
        if constexpr(!std::is_same_v<VariantType, std::monostate>)
        {
            using PixelType = typename VariantType::Type;
            constexpr uint32_t C = (VariantType::Channels == 4)
                                    ? 3
                                    : VariantType::Channels;
            if constexpr(C == 1)
            {
                if constexpr(std::is_floating_point_v<PixelType>)
                    result[0] = t[0];
                else if constexpr(std::is_unsigned_v<PixelType>)
                    result[0] = FromUNorm<Float>(PixelType(t[0]));
                else
                    result[0] = FromSNorm<Float>(PixelType(t[0]));
            }
            else
            {
                using InnerT = typename PixelType::InnerType;
                UNROLL_LOOP
                for(uint32_t c = 0; c < C; c++)
                {
                    if constexpr(std::is_floating_point_v<InnerT>)
                        result[c] = t[c];
                    else if constexpr(std::is_unsigned_v<InnerT>)
                        result[c] = FromUNorm<Float>(InnerT(t[c]));
                    else
                        result[c] = FromSNorm<Float>(InnerT(t[c]));
                }
            }
        }
        return result;
    });
}

MRAY_GPU MRAY_GPU_INLINE
Vector3 GenericToNorm(const Vector3& t, const SurfViewVariant& surfView)
{
    using namespace Bit::NormConversion;
    // Utilize the surfView variant to get the type etc
    return DeviceVisit(surfView, [t](auto&& sv) -> Vector3
    {
        using MathFunctions::Clamp;
        Vector3 result = t;
        using VariantType = std::remove_cvref_t<decltype(sv)>;
        if constexpr(!std::is_same_v<VariantType, std::monostate>)
        {
            using PixelType = typename VariantType::Type;
            constexpr uint32_t C = (VariantType::Channels == 4)
                                    ? 3
                                    : VariantType::Channels;
            if constexpr(C == 1)
            {
                if constexpr(std::is_floating_point_v<PixelType>)
                    result[0] = t[0];
                else if constexpr(std::is_unsigned_v<PixelType>)
                {
                    result[0] = Clamp<Float>(t[0], 0, 1);
                    result[0] = Float(ToUNorm<PixelType>(result[0]));
                }
                else
                {
                    result[0] = Clamp<Float>(t[0], -1, 1);
                    result[0] = Float(ToSNorm<PixelType>(t[0]));
                }
            }
            else
            {
                using InnerT = typename PixelType::InnerType;
                UNROLL_LOOP
                for(uint32_t c = 0; c < C; c++)
                {
                    if constexpr(std::is_floating_point_v<InnerT>)
                        result[c] = t[c];
                    else if constexpr(std::is_unsigned_v<InnerT>)
                    {
                        result[c] = Clamp<Float>(t[c], 0, 1);
                        result[c] = Float(ToUNorm<InnerT>(result[c]));
                    }
                    else
                    {
                        result[c] = Clamp<Float>(t[c], -1, 1);
                        result[c] = Float(ToSNorm<InnerT>(result[c]));
                    }
                }
            }
        }
        return result;
    });
}


template<uint32_t TPB, class ConverterTuple>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCConvertColor(// I-O
                    MRAY_GRID_CONSTANT const Span<MipArray<SurfViewVariant>> dSurfaces,
                    // Inputs
                    MRAY_GRID_CONSTANT const Span<const ColorConvParams> dColorConvParamsList,
                    // Constants
                    MRAY_GRID_CONSTANT const MRayColorSpaceEnum globalColorSpace,
                    MRAY_GRID_CONSTANT const uint32_t currentMipLevel,
                    MRAY_GRID_CONSTANT const uint32_t blockPerTexture)
{
    static constexpr ConverterTuple converterList = {};

    assert(dSurfaces.size() == dColorConvParamsList.size());
    // Block-stride loop
    KernelCallParams kp;
    uint32_t textureCount = static_cast<uint32_t>(dSurfaces.size());
    uint32_t blockCount = blockPerTexture * textureCount;
    for(uint32_t bI = kp.blockId; bI < blockCount; bI += kp.gridSize)
    {
        uint32_t tI = bI / blockPerTexture;
        uint32_t localBI = bI % blockPerTexture;
        // Load to local space
        ColorConvParams curParams = dColorConvParamsList[tI];
        SurfViewVariant rwSurf = dSurfaces[tI][currentMipLevel];
        //
        // Padded
        Vector2ui mipRes = Graphics::TextureMipSize(curParams.mipZeroRes,
                                                    currentMipLevel);
        // Skip this mip if it is not available.
        // Mips may be partially available so we check all mips.
        if(!curParams.validMips[currentMipLevel]) continue;
        if(std::holds_alternative<std::monostate>(rwSurf)) continue;

        bool noColorConvert = (curParams.fromColorSpace == MRayColorSpaceEnum::MR_DEFAULT ||
                               curParams.fromColorSpace == globalColorSpace);
        bool noGammaConvert = curParams.gamma == Float(1);

        if(noGammaConvert && noColorConvert) continue;

        // Loop over the blocks for this tex
        static constexpr Vector2ui TILE_SIZE = Vector2ui(32, 16);
        static_assert(TILE_SIZE.Multiply() == TPB);
        Vector2ui totalTiles = MathFunctions::DivideUp(mipRes, TILE_SIZE);
        for(uint32_t tileI = localBI; tileI < totalTiles.Multiply();
            tileI += blockPerTexture)
        {
            Vector2ui localPI = Vector2ui(kp.threadId % TILE_SIZE[0],
                                          kp.threadId / TILE_SIZE[0]);
            Vector2ui tile2D = Vector2ui(tileI % totalTiles[0],
                                         tileI / totalTiles[0]);
            Vector2ui pixCoord = tile2D * TILE_SIZE + localPI;

            if(pixCoord[0] >= mipRes[0] ||
               pixCoord[1] >= mipRes[1]) continue;

            // We assume 4 channel textures are RGB (first 3 channels)
            // and something else (A channel) so clamp to Vector3 and calculate
            // If it is two or one channel, other one/two parameter(s) will be
            // zeroed out.
            Vector4 localPix = GenericRead(pixCoord, rwSurf);
            Vector3 localPixRGB = Vector3(localPix);

            // Another problem is that RWTextureRead returns exact values (in float
            // representation, for example 8-bit will return [0, 255] or [-128, 127]
            // we need to normalize these (we could get a texture view, but writing
            // these is also a problem anyway)
            // Technically signed normalization should not be used so we can abuse that
            localPixRGB = GenericFromNorm(localPixRGB, rwSurf);

            using namespace Color;
            // First do gamma correction
            if(!noGammaConvert)
                localPixRGB = OpticalTransferGamma(curParams.gamma).ToLinear(localPixRGB);

            // Then color
            if(!noColorConvert)
            {
                uint32_t i = static_cast<uint32_t>(curParams.fromColorSpace);
                [[maybe_unused]]
                bool invoked = InvokeAt(i, converterList, [i, &localPixRGB](auto&& tupleElem)
                {
                    localPixRGB = tupleElem.Convert(localPixRGB);
                    return true;
                });
                assert(invoked);
            }

            // Convert it back to normalized state
            localPixRGB = GenericToNorm(localPixRGB, rwSurf);

            GenericWrite(rwSurf, Vector4(localPixRGB, localPix[3]), pixCoord);
        }
    }
}

template<uint32_t TPB, class BCReader, class ConverterTuple>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCConvertColorBC(// I-O
                      MRAY_GRID_CONSTANT const Span<typename BCReader::BlockType> dBCBlocks,
                      // Inputs
                      MRAY_GRID_CONSTANT const BCColorConvParamList dTexConvParams,
                      // Constants
                      MRAY_GRID_CONSTANT const uint32_t validTexCount,
                      MRAY_GRID_CONSTANT const MRayColorSpaceEnum globalColorSpace,
                      MRAY_GRID_CONSTANT const uint32_t processorPerTexture)
{
    static constexpr ConverterTuple converterList = {};

    // Block-stride loop
    KernelCallParams kp;
    uint32_t textureCount = static_cast<uint32_t>(dTexConvParams.size());
    uint32_t blockCount = processorPerTexture * textureCount;
    for(uint32_t procI = kp.blockId; procI < blockCount; procI += kp.gridSize)
    {
        uint32_t texI = procI / processorPerTexture;
        uint32_t localProcI = procI % processorPerTexture;
        // Load to local space
        const BCColorConvParams& curParams = dTexConvParams[texI];
        uint32_t tileCount = curParams.blockRange[1] - curParams.blockRange[0];
        if(texI >= validTexCount) continue;


        // Skip this mip if it is not available.
        // Mips may be partially available so we check all mips.
        bool noColorConvert = (curParams.fromColorSpace == MRayColorSpaceEnum::MR_DEFAULT ||
                               curParams.fromColorSpace == globalColorSpace);
        bool noGammaConvert = (curParams.gamma == Float(1));
        if(noGammaConvert && noColorConvert) continue;

        // Convert color routine, this wrapped to a lambda due to
        // different compress decompress systems
        auto ConvColor = [&](Vector3 color)
        {
            using namespace Color;
            // First do gamma correction
            if(!noGammaConvert)
                color = OpticalTransferGamma(curParams.gamma).ToLinear(color);

            // Then color
            if(!noColorConvert)
            {
                uint32_t i = static_cast<uint32_t>(curParams.fromColorSpace);
                [[maybe_unused]]
                bool invoked = InvokeAt(i, converterList, [&](auto&& tupleElem)
                {
                    color = tupleElem.Convert(color);
                    return true;
                });
                assert(invoked);
            }
            return color;
        };

        // Loop over the blocks for this tex
        uint32_t tileStart = localProcI * kp.blockSize + kp.threadId;
        uint32_t tileIncrement = processorPerTexture * kp.blockSize;
        for(uint32_t tileI = tileStart; tileI < tileCount; tileI += tileIncrement)
        {
            using BlockType = typename BCReader::BlockType;
            BlockType block = dBCBlocks[curParams.blockRange[0] + tileI];

            // YOLO constexpr here,
            // BC7 has too many colors, register spill galore.
            // For BC7 we do streaming architecture
            // so that we do not have to store the all colors.
            //
            // BC(1-5) has small inter color dependency
            // (i.e. color0 > color1 switches to alpha mode etc.)
            // For these we could not do streaming.
            // Thus; this constexpr if statement
            static constexpr bool IsBC7 = std::is_same_v<BCReader, BlockCompressedIO::BC7>;
            if constexpr(IsBC7)
            {
                BCReader bcIO(block);
                //for(uint32_t i = 0; i < bcIO.ColorCount(); i++)
                for(uint32_t i = 0; i < 6; i++)
                {
                    Vector3 c = bcIO.ExtractColor(i);
                    c = ConvColor(c);
                    bcIO.InjectColor(i, c);
                }
                block = bcIO.Block();
            }
            else
            {
                using ColorPack = typename BCReader::ColorPack;
                ColorPack localPixels = BCReader::ExtractColors(block);
                for(auto& localPixRGB : localPixels)
                    localPixRGB = ConvColor(localPixRGB);
                block = BCReader::InjectColors(block, localPixels);
            }

            // Finally, Write back
            dBCBlocks[curParams.blockRange[0] + tileI] = block;
        }
    }
}

class BCColorConverter
{
    private:
    std::vector<GenericTexture*>    bcTextures;
    std::vector<Vector2ui>          partitions;
    size_t                          bufferSize;

    template <std::contiguous_iterator It, class Compare>
    std::vector<Vector2ui> PartitionRange(It first, It last, Compare&& cmp);

    template<MRayPixelEnum E>
    Pair<MipBlockCountList, uint64_t>
            FindTotalTilesOf(const GenericTexture* t) const;
    //
    template<MRayPixelEnum E>
    void    CallKernelForType(Span<Byte> dScratchBuffer,
                              // Constants
                              const Vector2ui& range,
                              MRayColorSpaceEnum globalColorSpace,
                              const GPUQueue& queue);

    public:
    // Constructors & Destructor
            BCColorConverter(std::vector<GenericTexture*>&& bcTextures);
    //
    size_t  ScratchBufferSize() const;
    void    CallBCColorConvertKernels(Span<Byte> dScratchBuffer,
                                      MRayColorSpaceEnum globalColorSpace,
                                     const GPUQueue& queue);
};

template <std::contiguous_iterator It, class Compare>
std::vector<Vector2ui> BCColorConverter::PartitionRange(It first, It last, Compare&& cmp)
{
    assert(std::is_sorted(first, last, cmp));
    std::vector<Vector2ui> result;
    It start = first;
    while(start != last)
    {
        It end = std::upper_bound(start, last, *start, cmp);
        Vector2ui r(std::distance(first, start),
                    std::distance(first, end));
        result.push_back(r);
        start = end;
    }
    return result;
}

template<MRayPixelEnum E>
Pair<MipBlockCountList, uint64_t>
BCColorConverter::FindTotalTilesOf(const GenericTexture* t) const
{
    static_assert(MRayPixelType<E>::IsBCPixel,
                  "This function can only be generated for BC types");
    using PixType = MRayPixelType<E>::Type;
    static constexpr uint32_t TILE_SIZE = PixType::TileSize;

    using MathFunctions::DivideUp;
    uint64_t total = 0;
    MipBlockCountList result;
    for(uint32_t i = 0; i < t->MipCount(); i++)
    {
        Vector3ui mipSize = Graphics::TextureMipSize(t->Extents(), i);
        Vector3ui blockCount = Vector3ui(DivideUp(Vector2ui(mipSize),
                                                  Vector2ui(TILE_SIZE)),
                                         mipSize[2]);
        uint64_t mipBlockCount = blockCount.Multiply();
        result.push_back(mipBlockCount);
        total += mipBlockCount;
    }
    return {result, total};
}

template<MRayPixelEnum E>
void BCColorConverter::CallKernelForType(Span<Byte> dScratchBuffer,
                                         // Constants
                                         const Vector2ui& range,
                                         MRayColorSpaceEnum globalColorSpace,
                                         const GPUQueue& queue)
{

    using BCReaderType = typename BCReaderFinder::template Find<E>;
    using BlockT = typename BCReaderType::BlockType;
    Span<BlockT> dBlocks = MemAlloc::RepurposeAlloc<BlockT>(dScratchBuffer);

    // Process batch by batch
    auto textureRange = Span<GenericTexture*>(bcTextures.begin() + range[0],
                                              bcTextures.begin() + range[1]);

    uint32_t localTexCount = uint32_t(range[1] - range[0]);
    uint32_t batchCount = MathFunctions::DivideUp(localTexCount, BC_TEX_PER_BATCH);
    for(uint32_t batchIndex = 0; batchIndex < batchCount; batchIndex++)
    {
        uint32_t  start = batchIndex * BC_TEX_PER_BATCH;
        uint32_t  end = std::min((batchIndex + 1) * BC_TEX_PER_BATCH, localTexCount);
        auto localTextures = textureRange.subspan(start, uint32_t(end - start));
        //
        uint64_t texBlockOffset = 0;
        BCColorConvParamList paramsList;
        for(size_t tI = 0; tI < localTextures.size(); tI++)
        {
            const auto* t = localTextures[tI];
            auto [mipBlocks, totalBlocks] = FindTotalTilesOf<E>(t);
            assert(mipBlocks.size() == t->MipCount());
            paramsList[tI] = BCColorConvParams
            {
                .blockRange = Vector2ul(texBlockOffset, totalBlocks),
                .gamma = t->Gamma(),
                .fromColorSpace = t->ColorSpace()
            };

            // While calculating other stuff, issue memcpy of
            // this texture
            uint64_t mipBlockOffset = texBlockOffset;
            for(uint32_t mipLevel = 0; mipLevel < t->MipCount(); mipLevel++)
            {
                uint64_t mipBlockSize = mipBlocks[mipLevel];
                Vector3ui mipSize = Graphics::TextureMipSize(t->Extents(), mipLevel);
                Span<Byte> copyRegion = dScratchBuffer.subspan(mipBlockOffset * sizeof(BlockT),
                                                               mipBlockSize * sizeof(BlockT));
                t->CopyToAsync(copyRegion, queue, mipLevel,
                               Vector3ui::Zero(), mipSize);
                mipBlockOffset += mipBlockSize;
            }
            // Advance the to the next texture
            texBlockOffset += totalBlocks;
        }

        //================//
        //  Kernel Call!  //
        //================//
        uint32_t colorSpaceI = static_cast<uint32_t>(globalColorSpace);
        InvokeAt(colorSpaceI, ConverterLists,
        [&](auto ConvList) -> bool
        {
            static constexpr uint32_t PROCESSOR_PER_TEXTURE = 256;
            static constexpr uint32_t THREAD_PER_BLOCK = 512;
            // Get Compile Time Type
            using ConvListType = std::remove_cvref_t<decltype(ConvList)>;
            static constexpr auto Kernel = KCConvertColorBC<THREAD_PER_BLOCK, BCReaderType,
                                                            ConvListType>;
            // Find maximum block count for state allocation
            uint32_t blockCount = queue.RecommendedBlockCountDevice(Kernel,
                                                                    THREAD_PER_BLOCK, 0);

            using namespace std::string_literals;
            static const std::string KernelName = ("KCConvertColorspaceBC"s +
                                                   std::string(MRayPixelTypeStringifier::ToString(E)));
            queue.IssueExactKernel<Kernel>
            (
                KernelName,
                KernelExactIssueParams
                {
                    .gridSize = blockCount,
                    .blockSize = THREAD_PER_BLOCK
                },
                // I-O
                dBlocks,
                // Inputs
                paramsList,
                // Constants
                uint32_t(localTextures.size()),
                globalColorSpace,
                PROCESSOR_PER_TEXTURE
            );
            return true;
        });

        texBlockOffset = 0;
        for(size_t tI = 0; tI < localTextures.size(); tI++)
        {
            auto* t = localTextures[tI];
            auto [mipBlocks, totalBlocks] = FindTotalTilesOf<E>(t);
            assert(mipBlocks.size() == t->MipCount());
            size_t mipBlockOffset = texBlockOffset;
            for(uint32_t mipLevel = 0; mipLevel < t->MipCount(); mipLevel++)
            {
                uint64_t mipBlockSize = mipBlocks[mipLevel];
                Vector3ui mipSize = Graphics::TextureMipSize(t->Extents(), mipLevel);
                Span<Byte> copyRegion = dScratchBuffer.subspan(mipBlockOffset * sizeof(BlockT),
                                                               mipBlockSize * sizeof(BlockT));
                t->CopyFromAsync(queue, mipLevel, Vector3ui::Zero(),
                                 mipSize, ToConstSpan(copyRegion));
                mipBlockOffset += mipBlockSize;
            }
            // Advance the to the next texture
            texBlockOffset += totalBlocks;
            // Texture is converted (or in process of)
            // so set its color space
            t->SetColorSpace(globalColorSpace);
        }
    }
}

BCColorConverter::BCColorConverter(std::vector<GenericTexture*>&& bcTex)
    : bcTextures(std::move(bcTex))
    , bufferSize(0)
{
    if(bcTextures.empty()) return;

    // Sort
    auto PixTypeComp = [](GenericTexture* l, GenericTexture* r)
    {
        return l->PixelType().Name() < r->PixelType().Name();
    };
    std::sort(bcTextures.begin(), bcTextures.end(), PixTypeComp);
    // Find the partitions (Per BC texture type)
    partitions = PartitionRange(bcTextures.begin(), bcTextures.end(),
                                PixTypeComp);
    // Find the memory
    for(Vector2ui& range : partitions)
    {
        uint32_t localTexCount = uint32_t(range[1] - range[0]);
        uint32_t iterCount = MathFunctions::DivideUp(localTexCount, BC_TEX_PER_BATCH);
        for(uint32_t i = 0; i < iterCount; i++)
        {
            uint32_t  start = range[0] + i * BC_TEX_PER_BATCH;
            uint32_t  end = range[0] + (i + 1) * BC_TEX_PER_BATCH;
            end = std::min(end, range[0] + localTexCount);
            // "Size()" gives the aligned size (mutiple of 64k in CUDA)
            // so unnecessarily large maybe?
            // TODO: Profile and check this later
            size_t localSize = 0;
            for(uint32_t j = start; j < end; j++)
                localSize += bcTextures[j]->Size();

            bufferSize = std::max(bufferSize, localSize);
        }
    }
}

size_t BCColorConverter::ScratchBufferSize() const
{
    return bufferSize;
}

void BCColorConverter::CallBCColorConvertKernels(Span<Byte> dScratchBuffer,
                                                 // Constants
                                                 MRayColorSpaceEnum globalColorSpace,
                                                 const GPUQueue& queue)
{
    if(bcTextures.empty()) return;

    for(size_t pIndex = 0; pIndex < partitions.size(); pIndex++)
    {
        const Vector2ui& range = partitions[pIndex];
        std::visit([&, this](auto&& v)
        {
            using PT = std::_Remove_cvref_t<decltype(v)>;
            constexpr MRayPixelEnum E = PT::Name;
            if constexpr(PT::IsBCPixel)
            {
                CallKernelForType<E>(dScratchBuffer, range,
                                     globalColorSpace, queue);
            }
        }, bcTextures[range[0]]->PixelType());
    }
}

void CallColorConvertKernel(Span<MipArray<SurfViewVariant>> dSufViews,
                            Span<const ColorConvParams> dColorConvParams,
                            uint8_t maxMipCount,
                            MRayColorSpaceEnum globalColorSpace,
                            const GPUQueue& queue)
{
    using enum MRayColorSpaceEnum;
    // Start from 1, we assume miplevel zero is already available
    for(uint16_t i = 0; i < maxMipCount; i++)
    {
        // We will dedicate N blocks for each texture.
        static constexpr uint32_t THREAD_PER_BLOCK = 512;
        static constexpr uint32_t BLOCK_PER_TEXTURE = 256;
        constexpr uint32_t BlockPerTexture = std::max(1u, BLOCK_PER_TEXTURE >> 1);

        uint32_t colorSpaceI = static_cast<uint32_t>(globalColorSpace);
        InvokeAt(colorSpaceI, ConverterLists,
        [&](auto&& ConvList) -> bool
        {
            // Get Compile Time Type
            using ConvListType = std::remove_cvref_t<decltype(ConvList)>;
            static constexpr auto Kernel = KCConvertColor<512, ConvListType>;
            // Find maximum block count for state allocation
            uint32_t blockCount = queue.RecommendedBlockCountDevice(Kernel,
                                                                    THREAD_PER_BLOCK, 0);

            using namespace std::string_view_literals;
            queue.IssueExactKernel<Kernel>
            (
                "KCConvertColorspace"sv,
                KernelExactIssueParams
                {
                    .gridSize = blockCount,
                    .blockSize = THREAD_PER_BLOCK
                },
                // I-O
                dSufViews,
                // Inputs
                dColorConvParams,
                // Constants
                globalColorSpace,
                i,
                BlockPerTexture
            );
            return true;
        });
    }
}

ColorConverter::ColorConverter(const GPUSystem& sys)
    : gpuSystem(sys)
{}

void ColorConverter::ConvertColor(std::vector<MipArray<SurfRefVariant>> textures,
                                  std::vector<ColorConvParams> colorConvParams,
                                  std::vector<GenericTexture*> bcTextures,
                                  MRayColorSpaceEnum globalColorSpace) const
{
    constexpr std::array A =
    {
        Vector4ui(0x47b22940, 0xfffe9d83, 0x56624343, 0xfc992220),
        Vector4ui(0x47b22940, 0xfffe9d83, 0x12235535, 0xcadf2112),
        Vector4ui(0xe6d4e540, 0xfffec54b, 0x7bddc86f, 0x78507dec),
        Vector4ui(0xe6d525c0, 0xfffebd23, 0x45126603, 0xbced9bcd),
        Vector4ui(0x27526840, 0xfffe894b, 0xefc85533, 0x14439b61),
        Vector4ui(0xa69425c0, 0xfffea51b, 0x7da70c91, 0xddcd8943),
        Vector4ui(0xa6342440, 0xfffea903, 0x5996067f, 0x9dfd7cca),
        Vector4ui(0x8613a440, 0xfffea0fb, 0x69a387d, 0xffee58bc),
        Vector4ui(0x85d3a240, 0xfffe9ceb, 0x9301a639, 0xfb9bfcad),
        Vector4ui(0x6d1e6c0, 0xfffe8133, 0x6bc7578d, 0x2561bfd3),
        Vector4ui(0xf6d1a640, 0xfffe7d32, 0xcc852631, 0x8656fb75),
        Vector4ui(0x6f1e740, 0xfffe7d3b, 0xb66ba401, 0xfdccebcf),
        Vector4ui(0x66332440, 0xfffe950b, 0x204b9b9d, 0xd424e229),
        Vector4ui(0x8653e4c0, 0xfffe9d0b, 0x22003, 0xeddd039d),
        Vector4ui(0x961423c0, 0xfffea103, 0x56609973, 0xecbc7776),
        Vector4ui(0x277268c0, 0xfffe8553, 0xcedd5309, 0x7510ffb9),
        Vector4ui(0x36f26740, 0xfffe893b, 0xfb9c136d, 0x4884f989),
        Vector4ui(0x6f1e6c0, 0xfffe853b, 0xffdc8651, 0x2a899dbc),
        Vector4ui(0xf6b0e4c0, 0xfffe813a, 0x6fd88839, 0x81289889),
        Vector4ui(0x6652e340, 0xfffe9d1b, 0x28ee8963, 0x10001698),
        Vector4ui(0x66136340, 0xfffe9cfb, 0x86877301, 0x669e4499),
        Vector4ui(0x763363c0, 0xfffe9d0b, 0xffdb3a7b, 0x3068e9ba)
    };

    //for(uint32_t i = 0; i < A.size(); i++)
    //{
    //    BlockCompressedIO::BC7 bcIO(A[i]);
    //    auto a = bcIO.ExtractColor(0);
    //    bcIO.InjectColor(0, a);
    //    auto b0 = bcIO.Block();
    //    assert(b0 == A[i]);
    //    //
    //    auto b = bcIO.ExtractColor(1);
    //    bcIO.InjectColor(1, b);
    //    auto b1 = bcIO.Block();
    //    assert(b1 == A[i]);
    //}


    //static constexpr
    //auto block = Vector4ui(0x37777991, 0xAEF0F115,
    //                      0xE47DC88C, 0xA4A89924);
    //BlockCompressedIO::BC7 bcIO(block);
    //auto a = bcIO.ExtractColor(0);
    //bcIO.InjectColor(0, a);
    //auto b0 = bcIO.Block();
    //assert(b0 == block);
    ////
    //auto b = bcIO.ExtractColor(1);
    //bcIO.InjectColor(1, b);
    //auto b1 = bcIO.Block();
    //assert(b1 == block);

    //auto c = bcIO.ExtractColor(2);
    //bcIO.InjectColor(2, c);
    //auto b2 = bcIO.Block();
    //assert(b2 == block);

    //auto d = bcIO.ExtractColor(3);
    //bcIO.InjectColor(3, d);
    //auto b3 = bcIO.Block();
    //assert(b3 == block);

    //auto e = bcIO.ExtractColor(4);
    //bcIO.InjectColor(4, e);
    //auto b4 = bcIO.Block();
    //assert(b4 == block);

    //auto f = bcIO.ExtractColor(5);
    //bcIO.InjectColor(5, f);
    //auto b5 = bcIO.Block();
    //assert(b5 == block);




    // TODO: Report bug on NVCC, "NVCC Language Frontend"
    // hangs (inf loop) when these literals are changed to "..."sv
    //using namespace std::string_view_literals;
    static const auto mainA = gpuSystem.CreateAnnotation("ConvertColor");
    static const auto normA = gpuSystem.CreateAnnotation("ConvertColor_N");
    static const auto bcA   = gpuSystem.CreateAnnotation("ConvertColor_BC");
    const auto _ = mainA.AnnotateScope();

    assert(textures.size() == colorConvParams.size());
    // TODO: Textures should be partitioned with respect to
    // devices, so that we can launch kernel from those devices
    const GPUDevice& bestDevice = gpuSystem.BestDevice();
    const GPUQueue& queue = bestDevice.GetComputeQueue(0);
    // We can temporarily allocate here. This will be done at
    // initialization time.
    DeviceMemory mem({&gpuSystem.BestDevice()}, 16_MiB, 512_MiB, true);
    Span<MipArray<SurfViewVariant>> dSufViews;
    Span<ColorConvParams> dColorConvParams;

    BCColorConverter bcColorConverter(std::move(bcTextures));
    size_t bcBufferSize = bcColorConverter.ScratchBufferSize();
    mem.ResizeBuffer(bcBufferSize);

    // Alias the memory, normal color conversion and BC color conversion
    // will be serialized
    MemAlloc::AllocateMultiData(std::tie(dSufViews, dColorConvParams),
                                mem, {textures.size(), textures.size()});

    //==============================//
    //       NORMAL TEXTURES        //
    //==============================//
    if(!textures.empty())
    {
        [[maybe_unused]]
        const auto normAnnotation = normA.AnnotateScope();

        // Copy references
        std::vector<MipArray<SurfViewVariant>> hSurfViews;
        hSurfViews.reserve(textures.size());
        for(const MipArray<SurfRefVariant>& surfRefs : textures)
        {
            MipArray<SurfViewVariant> mipViews;
            for(size_t i = 0; i < TracerConstants::MaxTextureMipCount; i++)
            {
                const SurfRefVariant& surf = surfRefs[i];
                mipViews[i] = std::visit([](auto&& v) -> SurfViewVariant
                {
                    using T = std::remove_cvref_t<decltype(v)>;
                    if constexpr(std::is_same_v<T, std::monostate>)
                        return std::monostate{};
                    else return v.View();

                }, surf);
            }
            hSurfViews.push_back(mipViews);
        }
        //
        auto hSurfViewSpan = Span<MipArray<SurfViewVariant>>(hSurfViews.begin(),
                                                             hSurfViews.end());
        auto hColorConvParams = Span<const ColorConvParams>(colorConvParams.cbegin(),
                                                            colorConvParams.cend());
        queue.MemcpyAsync(dSufViews, ToConstSpan(hSurfViewSpan));
        queue.MemcpyAsync(dColorConvParams, hColorConvParams);
        //
        // We will globally call a single kernel for each mip level
        uint8_t maxMipCount = std::transform_reduce(colorConvParams.cbegin(),
                                                    colorConvParams.cend(),
                                                    std::numeric_limits<uint8_t>::min(),
        [](uint8_t l, uint8_t r)
        {
            return std::max(l, r);
        },
        [](const ColorConvParams& p) -> uint8_t
        {
            return p.mipCount;
        });
        CallColorConvertKernel(dSufViews, dColorConvParams, maxMipCount,
                               globalColorSpace, queue);
    }
    //==============================//
    //       BC TEXTURES            //
    //==============================//
    {
        [[maybe_unused]]
        const auto bcAnnotation = bcA.AnnotateScope();

        Span<Byte> dBCScratchBuffer = Span<Byte>(static_cast<Byte*>(mem), mem.Size());
        bcColorConverter.CallBCColorConvertKernels(dBCScratchBuffer, globalColorSpace, queue);
    }
    // Wait for deallocation
    queue.Barrier().Wait();
}
