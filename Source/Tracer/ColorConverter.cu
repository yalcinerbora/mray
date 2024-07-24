#include "ColorConverter.h"
#include "GenericTextureRW.h"
#include "BCColorIO.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

#include "Core/ColorFunctions.h"
#include "Core/GraphicsFunctions.h"

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
                    MRAY_GRID_CONSTANT const Span<const ColorConvParams> dMipGenParamsList,
                    // Constants
                    MRAY_GRID_CONSTANT const ConverterTuple converterList,
                    MRAY_GRID_CONSTANT const MRayColorSpaceEnum globalColorSpace,
                    MRAY_GRID_CONSTANT const uint32_t currentMipLevel,
                    MRAY_GRID_CONSTANT const uint32_t blockPerTexture)
{
    assert(dSurfaces.size() == dMipGenParamsList.size());
    // Block-stride loop
    KernelCallParams kp;
    uint32_t textureCount = static_cast<uint32_t>(dSurfaces.size());
    uint32_t blockCount = blockPerTexture * textureCount;
    for(uint32_t bI = kp.blockId; bI < blockCount; bI += kp.gridSize)
    {
        uint32_t tI = bI / blockPerTexture;
        uint32_t localBI = bI % blockPerTexture;
        // Load to local space
        ColorConvParams curParams = dMipGenParamsList[tI];
        SurfViewVariant rwSurf = dSurfaces[tI][currentMipLevel];
        //
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

ColorConverter::ColorConverter(const GPUSystem& sys)
    : gpuSystem(sys)
{}

void ColorConverter::ConvertColor(const std::vector<MipArray<SurfRefVariant>>& textures,
                                  const std::vector<ColorConvParams>& colorConvParams,
                                  MRayColorSpaceEnum globalColorSpace) const
{
    assert(textures.size() == colorConvParams.size());
    // TODO: Textures should be partitioned with respect to
    // devices, so that we can launch kernel from those devices
    const GPUDevice& bestDevice = gpuSystem.BestDevice();
    const GPUQueue& queue = bestDevice.GetComputeQueue(0);
    // We can temporarily allocate here. This will be done at
    // initialization time.
    DeviceLocalMemory mem(gpuSystem.BestDevice());
    Span<MipArray<SurfViewVariant>> dSufViews;
    Span<ColorConvParams> dColorConvParams;
    MemAlloc::AllocateMultiData(std::tie(dSufViews, dColorConvParams),
                                mem, {textures.size(), textures.size()});

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

    auto hSurfViewSpan = Span<MipArray<SurfViewVariant>>(hSurfViews.begin(),
                                                         hSurfViews.end());
    auto hColorConvParams = Span<const ColorConvParams>(colorConvParams.cbegin(),
                                                        colorConvParams.cend());
    queue.MemcpyAsync(dSufViews, ToConstSpan(hSurfViewSpan));
    queue.MemcpyAsync(dColorConvParams, hColorConvParams);

    // Since texture writes are not coherent,
    // we need to call one kernel for each level of mips
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

    using enum MRayColorSpaceEnum;
    static constexpr Tuple ConverterKernels =
    {
        Pair{KCConvertColor<512, ConverterList<MR_ACES2065_1>>, ConverterList<MR_ACES2065_1>{}},
        Pair{KCConvertColor<512, ConverterList<MR_ACES_CG>>,    ConverterList<MR_ACES_CG>{}},
        Pair{KCConvertColor<512, ConverterList<MR_REC_709>>,    ConverterList<MR_REC_709>{}},
        Pair{KCConvertColor<512, ConverterList<MR_REC_2020>>,   ConverterList<MR_REC_2020>{}},
        Pair{KCConvertColor<512, ConverterList<MR_DCI_P3>>,     ConverterList<MR_DCI_P3>{}},
        Pair{KCConvertColor<512, ConverterList<MR_ADOBE_RGB>>,  ConverterList<MR_ADOBE_RGB>{}},

    };

    // Start from 1, we assume miplevel zero is already available
    for(uint16_t i = 0; i < maxMipCount; i++)
    {
        // We will dedicate N blocks for each texture.
        static constexpr uint32_t THREAD_PER_BLOCK = 512;
        static constexpr uint32_t BLOCK_PER_TEXTURE = 256;
        constexpr uint32_t BlockPerTexture = std::max(1u, BLOCK_PER_TEXTURE >> 1);

        uint32_t colorSpaceI = static_cast<uint32_t>(globalColorSpace);
        InvokeAt
        (
            colorSpaceI, ConverterKernels,
            [&](auto&& KernelTypePair) -> bool
            {
                // Find maximum block count for state allocation
                uint32_t blockCount = queue.RecommendedBlockCountDevice(KernelTypePair.first,
                                                                        THREAD_PER_BLOCK, 0);

                using namespace std::string_view_literals;
                // Find Compile Time Kernel
                constexpr auto KernelCT = std::get<std::remove_cvref_t<decltype(KernelTypePair)>>(ConverterKernels).first;
                queue.IssueExactKernel<KernelCT>
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
                    KernelTypePair.second,
                    globalColorSpace,
                    i,
                    BlockPerTexture
                );
                return true;
            }
        );
    }
    queue.Barrier().Wait();
}