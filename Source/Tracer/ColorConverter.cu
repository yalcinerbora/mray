#include "ColorConverter.h"
#include "GenericTextureRW.h"
#include "BCColorIO.h"
#include "TextureMemory.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

#include "Core/ColorFunctions.h"
#include "Core/GraphicsFunctions.h"

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

//// Order is important here
//template<class R>
//constexpr auto CreateBCKernelList()
//{
//    using enum MRayColorSpaceEnum;
//    return Tuple
//    {
//        KCConvertColorBC<BC_CONV_TPB, R, ConverterList<MR_ACES2065_1>>,
//        KCConvertColorBC<BC_CONV_TPB, R, ConverterList<MR_ACES_CG>>,
//        KCConvertColorBC<BC_CONV_TPB, R, ConverterList<MR_REC_709>>,
//        KCConvertColorBC<BC_CONV_TPB, R, ConverterList<MR_REC_2020>>,
//        KCConvertColorBC<BC_CONV_TPB, R, ConverterList<MR_DCI_P3>>,
//        KCConvertColorBC<BC_CONV_TPB, R, ConverterList<MR_ADOBE_RGB>>
//    };
//}

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

//MRAY_GPU MRAY_GPU_INLINE
//Vector4ui ReadBlock(const SurfViewVariant& surfView, const Vector2ui& tileCoord)
//{
//    return DeviceVisit(surfView, [&](auto&& v)
//    {
//        using T = std::remove_cvref_t<decltype(v)>;
//        if constexpr(std::is_same_v<RWTextureView<2, Vector2ui>, T>)
//        {
//            return Vector4ui(v(tileCoord), 0, 0);
//        }
//        else if constexpr(std::is_same_v<RWTextureView<2, Vector4ui>, T>)
//        {
//            return v(tileCoord);
//        }
//        else return Vector4ui::Zero();
//    });
//}
//
//MRAY_GPU MRAY_GPU_INLINE
//void WriteBlock(SurfViewVariant& surfView,
//                const Vector4ui& block,
//                const Vector2ui& tileCoord)
//{
//    DeviceVisit(surfView, [&](auto&& v)
//    {
//        using T = std::remove_cvref_t<decltype(v)>;
//        if constexpr(std::is_same_v<RWTextureView<2, Vector2ui>, T>)
//        {
//            v(tileCoord) = Vector2ui(block[0], block[1]);
//        }
//        else if constexpr(std::is_same_v<RWTextureView<2, Vector4ui>, T>)
//        {
//            v(tileCoord) = block;
//        }
//    });
//}

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
        uint32_t localPI = procI % processorPerTexture;
        // Load to local space
        BCColorConvParams curParams = dTexConvParams[texI];
        uint32_t tileCount = curParams.blockRange[1] - curParams.blockRange[0];

        // Skip this mip if it is not available.
        // Mips may be partially available so we check all mips.
        bool noColorConvert = (curParams.fromColorSpace == MRayColorSpaceEnum::MR_DEFAULT ||
                               curParams.fromColorSpace == globalColorSpace);
        bool noGammaConvert = curParams.gamma == Float(1);

        if(noGammaConvert && noColorConvert) continue;

        // Loop over the blocks for this tex
        for(uint32_t tileI = localPI; tileI < tileCount; tileI += processorPerTexture)
        {
            using BlockType = typename BCReader::BlockType;
            using ColorPack = typename BCReader::ColorPack;
            BlockType block = dBCBlocks[curParams.blockRange[0] + tileI];
            ColorPack localPixels = BCReader::ExtractColors(block);

            // Convert each color one by one
            // TODO: This entire operation maybe
            // dedicated to a warp? then a thread maybe
            for(auto& localPixRGB : localPixels)
            {
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
            }

            // Write back
            block = BCReader::InjectColors(block, localPixels);
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
    void CallKernelForType(Span<Byte> dScratchBuffer,
                           // Constants
                           const Vector2ui& range,
                           MRayColorSpaceEnum globalColorSpace,
                           const GPUQueue& queue) const;

    public:
    // Constructors & Destructor
            BCColorConverter(std::vector<GenericTexture*>&& bcTextures);
    //
    size_t  ScratchBufferSize() const;
    void    CallBCColorConvertKernels(Span<Byte> dScratchBuffer,
                                      MRayColorSpaceEnum globalColorSpace,
                                     const GPUQueue& queue) const;
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
        Vector2ui r(std::distance(start, end),
                    std::distance(first, start));
        result.push_back(r);
        start = end;
    }
    return result;
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
            end = std::min(end, localTexCount);
            // "Size()" gives the aligned size (mutiple of 64k in CUDA)
            // so unnecessarily large maybe?
            // TODO: Profile and check this later
            size_t localSize = 0;
            for(uint32_t j = start; j < end; j++)
                localSize += bcTextures[i]->Size();

            bufferSize = std::max(bufferSize, localSize);
        }
    }
}

size_t BCColorConverter::ScratchBufferSize() const
{
    return bufferSize;
}

template<MRayPixelEnum E>
void BCColorConverter::CallKernelForType(Span<Byte> dScratchBuffer,
                                         // Constants
                                         const Vector2ui& range,
                                         MRayColorSpaceEnum globalColorSpace,
                                         const GPUQueue& queue) const
{

    using BCReaderType = typename BCReaderFinder::template Find<E>;
    //static constexpr auto KernelList = FindBCConvKernelList<E>();
    using BlockT = typename BCReaderType::BlockType;
    Span<BlockT> dBlocks = MemAlloc::RepurposeAlloc<BlockT>(dScratchBuffer);

    // Process batch by batch
    uint32_t localTexCount = uint32_t(range[1] - range[0]);
    uint32_t iterCount = MathFunctions::DivideUp(localTexCount, BC_TEX_PER_BATCH);
    for(uint32_t i = 0; i < iterCount; i++)
    {
        uint32_t  start = range[0] + i * BC_TEX_PER_BATCH;
        uint32_t  end = range[0] + (i + 1) * BC_TEX_PER_BATCH;
        end = std::min(end, localTexCount);
        uint32_t validTexCount = uint32_t(end - start);
        // "Size()" gives the aligned size (mutiple of 64k in CUDA)
        // so unnecessarily large maybe?
        // TODO: Profile and check this later
        //size_t offset = 0;
        BCColorConvParamList paramsList;
        for(uint32_t j = start; j < end; j++)
        {
            const auto* t = bcTextures[j];
            //uint32_t localI = j - start;

            //paramsList[localI] = BCColorConvParams
            //{
            //    .blockRange = Vector2ul(offset, t->Size()),
            //    .gamma = t->Gamma(),
            //    .fromColorSpace = t->ColorSpace()
            //};
            //offset += t->Size();

            // While calculating issue memcpy
            for(uint32_t mipLevel = 0; mipLevel < t->MipCount(); mipLevel++)
            {
                //t->CopyToAsync( queue);
            }
        }

        //================//
        //  Kernel Call!  //
        //================//
        //uint32_t colorSpaceI = static_cast<uint32_t>(globalColorSpace);
        //InvokeAt(colorSpaceI, ConverterLists,
        //[&](auto ConvList) -> bool
        //{
        //    static constexpr uint32_t PROCESSOR_PER_TEXTURE = 256;
        //    static constexpr uint32_t THREAD_PER_BLOCK = 512;
        //    // Get Compile Time Type
        //    using ConvListType = std::remove_cvref_t<decltype(ConvList)>;
        //    static constexpr auto Kernel = KCConvertColorBC<THREAD_PER_BLOCK, BCReaderType,
        //                                                    ConvListType>;
        //    // Find maximum block count for state allocation
        //    uint32_t blockCount = queue.RecommendedBlockCountDevice(Kernel,
        //                                                            THREAD_PER_BLOCK, 0);

        //    using namespace std::string_view_literals;
        //    queue.IssueExactKernel<Kernel>
        //    (
        //        "KCConvertColorspaceBC"sv,
        //        KernelExactIssueParams
        //        {
        //            .gridSize = blockCount,
        //            .blockSize = THREAD_PER_BLOCK
        //        },
        //        // I-O
        //        dBlocks,
        //        // Inputs
        //        paramsList,
        //        // Constants
        //        validTexCount,
        //        globalColorSpace,
        //        PROCESSOR_PER_TEXTURE
        //    );
        //    return true;
        //});

        //for(uint32_t j = start; j < end; j++)
        //{
        //    // While calculating issue memcpy
        //    for(uint32_t mipLevel = 0; mipLevel < t->MipCount(); mipLevel++)
        //    {
        //        //t->CopyFromAsync( queue);
        //    }
        //}

    }
}

void BCColorConverter::CallBCColorConvertKernels(Span<Byte> dScratchBuffer,
                                                 // Constants
                                                 MRayColorSpaceEnum globalColorSpace,
                                                 const GPUQueue& queue) const
{
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
    if(!bcTextures.empty())
    {
        Span<Byte> dBCScratchBuffer = Span<Byte>(static_cast<Byte*>(mem), mem.Size());
        bcColorConverter.CallBCColorConvertKernels(dBCScratchBuffer, globalColorSpace, queue);
    }
    // Wait for deallocation
    queue.Barrier().Wait();
}

//    std::vector<MipArray<SurfViewVariant>> hBCSurfViews;
//    hBCSurfViews.reserve(textures.size());
//    for(const MipArray<SurfRefVariant>& bcSurfRefs : bcTextures)
//    {
//        MipArray<SurfViewVariant> mipViews;
//        for(size_t i = 0; i < TracerConstants::MaxTextureMipCount; i++)
//        {
//            const SurfRefVariant& surf = bcSurfRefs[i];
//            mipViews[i] = std::visit([](auto&& v) -> SurfViewVariant
//            {
//                using T = std::remove_cvref_t<decltype(v)>;
//                if constexpr(std::is_same_v<T, std::monostate>)
//                    return std::monostate{};
//                else return v.View();
//            }, surf);
//        }
//        hBCSurfViews.push_back(mipViews);
//    }
//    //
//    auto hBCSurfViewSpan = Span<MipArray<SurfViewVariant>>(hBCSurfViews.begin(),
//                                                           hBCSurfViews.end());
//    auto hBCColorConvParams = Span<const BCColorConvParams>(bcColorConvParams.cbegin(),
//                                                            bcColorConvParams.cend());
//    queue.MemcpyAsync(dBCSufViews, ToConstSpan(hBCSurfViewSpan));
//    queue.MemcpyAsync(dBCColorConvParams, hBCColorConvParams);

//    uint8_t bcMaxMipCount = std::transform_reduce(bcColorConvParams.cbegin(),
//                                             bcColorConvParams.cend(),
//                                             std::numeric_limits<uint8_t>::min(),
//    [](uint8_t l, uint8_t r)
//    {
//        return std::max(l, r);
//    },
//    [](const ColorConvParams& p) -> uint8_t
//    {
//        return p.mipCount;
//    });


//if(isBlockCompressed)
//{
//    bcTextures.push_back();
//
//    auto [blockSize, tileSize] = std::visit
//    (
//        [](auto&& v) -> Pair<uint32_t, uint32_t>
//    {
//        using T = std::remove_cvref_t<decltype(v)>;
//        using Type = typename T::Type;
//        if constexpr(T::IsBCPixel)
//            return {uint32_t(Type::BlockSize), uint32_t(Type::TileSize)};
//        else return {0, 0};
//    }, pt
//    );
//    assert(blockSize != 0 && tileSize != 0);
//
//    BCColorConvParams bcP;
//    bcP.validMips = p.validMips;
//    bcP.mipCount = p.mipCount;
//    bcP.fromColorSpace = p.fromColorSpace;
//    bcP.gamma = p.gamma;
//    bcP.mipZeroRes = p.mipZeroRes;
//    //
//    bcP.pixelEnum = pt.Name();
//    bcP.blockSize = blockSize;
//    bcP.tileSize = tileSize;
//    bcColorConvParams.push_back(bcP);
//}
//else
