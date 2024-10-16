#include "ColorConverter.h"
#include "GenericTextureRW.h"
#include "BCColorIO.h"
#include "TextureMemory.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

#include "Core/ColorFunctions.h"
#include "Core/GraphicsFunctions.h"
#include "Core/Algorithm.h"

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
{
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

// Luminance Extract related

using ColorspaceList = Tuple
<
    Color::Colorspace<MRayColorSpaceEnum::MR_ACES2065_1>,
    Color::Colorspace<MRayColorSpaceEnum::MR_ACES_CG>,
    Color::Colorspace<MRayColorSpaceEnum::MR_REC_709>,
    Color::Colorspace<MRayColorSpaceEnum::MR_REC_2020>,
    Color::Colorspace<MRayColorSpaceEnum::MR_DCI_P3>,
    Color::Colorspace<MRayColorSpaceEnum::MR_ADOBE_RGB>
>;

struct LuminanceExtractParams
{
    Vector2ui           imgSize;
    MRayColorSpaceEnum  colorSpace;
};


template<uint32_t TPB>
MRAY_KERNEL
void KCExtractLuminance(// I-O
                        MRAY_GRID_CONSTANT const Span<const Span<Float>>,
                        // Inputs
                        MRAY_GRID_CONSTANT const Span<const LuminanceExtractParams>,
                        MRAY_GRID_CONSTANT const Span<const Variant<std::monostate, GenericTextureView>>,
                        // Constants
                        MRAY_GRID_CONSTANT const uint32_t);

template<uint32_t TPB, class ConverterTuple>
MRAY_KERNEL
void KCConvertColor(// I-O
                    MRAY_GRID_CONSTANT const Span<MipArray<SurfViewVariant>>,
                    // Inputs
                    MRAY_GRID_CONSTANT const Span<const ColorConvParams>,
                    // Constants
                    MRAY_GRID_CONSTANT const MRayColorSpaceEnum,
                    MRAY_GRID_CONSTANT const uint32_t,
                    MRAY_GRID_CONSTANT const uint32_t);

template<uint32_t TPB, class BCReader, class ConverterTuple>
MRAY_KERNEL
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
        using Math::Clamp;
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

MRAY_GPU MRAY_GPU_INLINE
Vector3 GenericReadFromView(const Vector2& uv, GenericTextureView& view)
{
    return DeviceVisit(view, [&](auto&& t) -> Vector3
    {
        using T = std::remove_cvref_t<decltype(t)>;
        if constexpr(T::Dimensions != 2)
            return Vector3::Zero();
        else if constexpr(T::Channels == 1)
            return Vector3(t(uv).value(), 0, 0);
        else if constexpr(T::Channels == 2)
            return Vector3(t(uv).value(), 0);
        else if constexpr(T::Channels == 3)
            return t(uv).value();
        else if constexpr(T::Channels == 4)
            return Vector3(t(uv).value());
        //
        return Vector3::Zero();
    });
}

template<uint32_t TPB>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCExtractLuminance(// I-O
                        MRAY_GRID_CONSTANT const Span<const Span<Float>> dLuminanceOutput,
                        // Inputs
                        MRAY_GRID_CONSTANT const Span<const LuminanceExtractParams> dLuminanceParams,
                        MRAY_GRID_CONSTANT const Span<const Variant<std::monostate, GenericTextureView>> dTextureViews,
                        // Constants
                        MRAY_GRID_CONSTANT const uint32_t blockPerTexture)
{
    static constexpr ColorspaceList colorspaceList = {};

    assert(dTextureViews.size() == dLuminanceParams.size() &&
           dTextureViews.size() == dLuminanceOutput.size());
    // Block-stride loop
    KernelCallParams kp;
    uint32_t textureCount = static_cast<uint32_t>(dTextureViews.size());
    uint32_t blockCount = blockPerTexture * textureCount;
    for(uint32_t bI = kp.blockId; bI < blockCount; bI += kp.gridSize)
    {
        uint32_t tI = bI / blockPerTexture;
        uint32_t localBI = bI % blockPerTexture;
        // Load to local space
        LuminanceExtractParams curParams = dLuminanceParams[tI];

        if(std::holds_alternative<std::monostate>(dTextureViews[tI]))
           continue;

        GenericTextureView texView = std::get<GenericTextureView>(dTextureViews[tI]);
        //
        Vector2ui res = curParams.imgSize;
        Vector2 resRecip = Vector2(1) / Vector2(res);
        //if(std::holds_alternative<std::monostate>(texView)) continue;

        // Loop over the blocks for this tex
        static constexpr Vector2ui TILE_SIZE = Vector2ui(32, 16);
        static_assert(TILE_SIZE.Multiply() == TPB);
        Vector2ui totalTiles = Math::DivideUp(res, TILE_SIZE);
        for(uint32_t tileI = localBI; tileI < totalTiles.Multiply();
            tileI += blockPerTexture)
        {
            Vector2ui localPI = Vector2ui(kp.threadId % TILE_SIZE[0],
                                          kp.threadId / TILE_SIZE[0]);
            Vector2ui tile2D = Vector2ui(tileI % totalTiles[0],
                                         tileI / totalTiles[0]);
            Vector2ui pixCoord = tile2D * TILE_SIZE + localPI;
            if(pixCoord[0] >= res[0] || pixCoord[1] >= res[1]) continue;

            // We assume 4 channel textures are RGB (first 3 channels)
            // and something else (A channel) so clamp to Vector3 and calculate
            // If it is two or one channel, other one/two parameter(s) will be
            // zeroed out.
            // We disregard if the pixel data is normalized or not.
            // It is user's problem.
            Vector2 uv = (Vector2(pixCoord) + Vector2(0.5)) * resRecip;
            Vector3 localPixRGB = GenericReadFromView(uv, texView);

            Float luminance;
            // Do the conversion
            uint32_t i = static_cast<uint32_t>(curParams.colorSpace);
            [[maybe_unused]]
            bool invoked = InvokeAt(i, colorspaceList, [i, &luminance, &localPixRGB](auto&& tupleElem)
            {
                luminance = Color::XYZToYxy(tupleElem.ToXYZ(localPixRGB))[0];
                return true;
            });
            assert(invoked);

            //
            Span<Float> dCurrentOutputSpan = dLuminanceOutput[tI];
            uint32_t linearPixelIndex =  pixCoord[1] * res[0] + pixCoord[0];
            dCurrentOutputSpan[linearPixelIndex] = luminance;
        }
    }
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
        Vector2ui totalTiles = Math::DivideUp(mipRes, TILE_SIZE);
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
                for(uint32_t i = 0; i < bcIO.ColorCount(); i++)
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
    std::vector<Vector2ul>          partitions;
    size_t                          bufferSize;

    template<MRayPixelEnum E>
    Pair<MipBlockCountList, uint64_t>
            FindTotalTilesOf(const GenericTexture* t) const;
    //
    template<MRayPixelEnum E>
    requires(E != MRayPixelEnum::MR_BC6H_UFLOAT &&
             E != MRayPixelEnum::MR_BC6H_SFLOAT)
    void    CallKernelForType(Span<Byte> dScratchBuffer,
                              // Constants
                              const Vector2ui& range,
                              MRayColorSpaceEnum globalColorSpace,
                              const GPUQueue& queue);
    template<MRayPixelEnum E>
    requires(E == MRayPixelEnum::MR_BC6H_UFLOAT ||
             E == MRayPixelEnum::MR_BC6H_SFLOAT)
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

template<MRayPixelEnum E>
Pair<MipBlockCountList, uint64_t>
BCColorConverter::FindTotalTilesOf(const GenericTexture* t) const
{
    static_assert(MRayPixelType<E>::IsBCPixel,
                  "This function can only be generated for BC types");
    using PixType = MRayPixelType<E>::Type;
    static constexpr uint32_t TILE_SIZE = PixType::TileSize;

    using Math::DivideUp;
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
requires(E == MRayPixelEnum::MR_BC6H_UFLOAT ||
         E == MRayPixelEnum::MR_BC6H_SFLOAT)
void BCColorConverter::CallKernelForType(Span<Byte>, const Vector2ui&,
                                         MRayColorSpaceEnum globalColorSpace,
                                         const GPUQueue&)
{
    MRAY_WARNING_LOG("[Tracer]: Scene has BC6H textures with a non \"MR_DEFAULT\" "
                     "color space. BC6H color space conversion is not supported."
                     "These textures will be treated as in Tracer's color space "
                     "(which is \"Linear/ACES_CG\")",
                     MRayColorSpaceStringifier::ToString(globalColorSpace));
}

template<MRayPixelEnum E>
requires(E != MRayPixelEnum::MR_BC6H_UFLOAT &&
         E != MRayPixelEnum::MR_BC6H_SFLOAT)
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
    uint32_t batchCount = Math::DivideUp(localTexCount, BC_TEX_PER_BATCH);
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
    partitions = Algo::PartitionRange(bcTextures.begin(), bcTextures.end(),
                                      PixTypeComp);
    // Find the memory
    for(const Vector2ul& range : partitions)
    {
        uint32_t localTexCount = uint32_t(range[1] - range[0]);
        uint32_t iterCount = Math::DivideUp(localTexCount, BC_TEX_PER_BATCH);
        for(uint32_t i = 0; i < iterCount; i++)
        {
            uint32_t  start = uint32_t(range[0]) + i * BC_TEX_PER_BATCH;
            uint32_t  end = uint32_t(range[0]) + (i + 1) * BC_TEX_PER_BATCH;
            end = std::min(end, uint32_t(range[0]) + localTexCount);
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
        const Vector2ul& range = partitions[pIndex];
        std::visit([&, this](auto&& v)
        {
            using PT = std::_Remove_cvref_t<decltype(v)>;
            constexpr MRayPixelEnum E = PT::Name;
            if constexpr(PT::IsBCPixel)
            {
                Vector2ui rangeI32(range[0], range[1]);
                CallKernelForType<E>(dScratchBuffer, rangeI32,
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
            static constexpr auto Kernel = KCConvertColor<THREAD_PER_BLOCK, ConvListType>;
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

void ColorConverter::ExtractLuminance(std::vector<Span<Float>> hLuminanceBuffers,
                                      std::vector<const GenericTexture*> textures,
                                      const GPUQueue& queue)
{
    std::vector<LuminanceExtractParams> hLuminanceExtractParams;
    std::vector<Variant<std::monostate, GenericTextureView>> hTextureViews;
    hLuminanceExtractParams.reserve(textures.size());
    hTextureViews.reserve(textures.size());
    size_t totalTexSize = textures.size();

    for(const GenericTexture* t : textures)
    {
        if(!t)
        {
            hTextureViews.push_back(std::monostate{});
            continue;
        }

        assert(t->DimensionCount() == 2);
        LuminanceExtractParams params =
        {
            .imgSize = Vector2ui(t->Extents()),
            .colorSpace = t->ColorSpace()
        };
        hLuminanceExtractParams.push_back(params);
        hTextureViews.push_back(t->View(TextureReadMode::DIRECT));
    }

    DeviceLocalMemory localMem(*queue.Device());
    //
    Span<Span<Float>> dLuminanceBuffers;
    Span<Variant<std::monostate, GenericTextureView>> dTextureViews;
    Span<LuminanceExtractParams> dLuminanceExtractParams;
    MemAlloc::AllocateMultiData(std::tie(dLuminanceExtractParams,
                                         dTextureViews,
                                         dLuminanceBuffers),
                                localMem,
                                {totalTexSize, totalTexSize, totalTexSize});

    queue.MemcpyAsync(dTextureViews,
                      Span<const Variant<std::monostate, GenericTextureView>>(hTextureViews));
    queue.MemcpyAsync(dLuminanceExtractParams,
                      Span<const LuminanceExtractParams>(hLuminanceExtractParams));
    queue.MemcpyAsync(dLuminanceBuffers, Span<const Span<Float>>(hLuminanceBuffers));

    static constexpr uint32_t BLOCK_PER_TEXTURE = 256;
    static constexpr uint32_t THREAD_PER_BLOCK = 512;
    // Get Compile Time Type
    static constexpr auto Kernel = KCExtractLuminance<THREAD_PER_BLOCK>;
    // Find maximum block count for state allocation
    uint32_t blockCount = queue.RecommendedBlockCountDevice(Kernel, THREAD_PER_BLOCK, 0);

    using namespace std::string_view_literals;
    queue.IssueExactKernel<Kernel>
    (
        "KCExtractLuminance"sv,
        KernelExactIssueParams
        {
            .gridSize = blockCount,
            .blockSize = THREAD_PER_BLOCK
        },
        ToConstSpan(dLuminanceBuffers),
        dLuminanceExtractParams,
        ToConstSpan(dTextureViews),
        BLOCK_PER_TEXTURE
    );

    queue.Barrier().Wait();
}