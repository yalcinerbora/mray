#include "ImageLoader.h"

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/deepdata.h>

Expected<MRayPixelTypeRT> ImageFileOIIO::PixelFormatToMRay(const OIIO::ImageSpec& spec)
{
    using ResultT = Expected<MRayPixelTypeRT>;
    using enum MRayPixelEnum;
    switch(spec.format.basetype)
    {
        case OIIO::TypeDesc::UINT8:
        {
            switch(spec.nchannels)
            {
                case 1: return ResultT(MRayPixelType<MR_R8_UNORM>{});    break;
                case 2: return ResultT(MRayPixelType<MR_RG8_UNORM>{});   break;
                case 3: return ResultT(MRayPixelType<MR_RGB8_UNORM>{});  break;
                case 4: return ResultT(MRayPixelType<MR_RGBA8_UNORM>{}); break;
                default: break;
            }
            break;
        }
        case OIIO::TypeDesc::INT8:
        {
            switch(spec.nchannels)
            {
                case 1: return MRayPixelType<MR_R8_SNORM>{};    break;
                case 2: return MRayPixelType<MR_RG8_SNORM>{};   break;
                case 3: return MRayPixelType<MR_RGB8_SNORM>{};  break;
                case 4: return MRayPixelType<MR_RGBA8_SNORM>{}; break;
                default: break;
            }
            break;
        }
        case OIIO::TypeDesc::UINT16:
        {
            switch(spec.nchannels)
            {
                case 1: return MRayPixelType<MR_R16_UNORM>{};       break;
                case 2: return MRayPixelType<MR_RG16_UNORM>{};      break;
                case 3: return MRayPixelType<MR_RGB16_UNORM>{};     break;
                case 4: return MRayPixelType<MR_RGBA16_UNORM>{};    break;
                default: break;
            }
            break;
        }
        case OIIO::TypeDesc::INT16:
        {
            switch(spec.nchannels)
            {
                case 1: return MRayPixelType<MR_R16_SNORM>{};       break;
                case 2: return MRayPixelType<MR_RG16_SNORM>{};      break;
                case 3: return MRayPixelType<MR_RGB16_SNORM>{};     break;
                case 4: return MRayPixelType<MR_RGBA16_SNORM>{};    break;
                default: break;
            }
            break;
        }

        // TODO: Support reading these
        case OIIO::TypeDesc::UINT32:
        case OIIO::TypeDesc::INT32:
        case OIIO::TypeDesc::UINT64:
        case OIIO::TypeDesc::INT64:
        case OIIO::TypeDesc::DOUBLE:
            break;

        case OIIO::TypeDesc::HALF:
        {
            switch(spec.nchannels)
            {
                case 1: return MRayPixelType<MR_R_HALF>{};     break;
                case 2: return MRayPixelType<MR_RG_HALF>{};    break;
                case 3: return MRayPixelType<MR_RGB_HALF>{};   break;
                case 4: return MRayPixelType<MR_RGBA_HALF>{};  break;
                default: break;
            }
            break;
        }
        case OIIO::TypeDesc::FLOAT:
        {
            switch(spec.nchannels)
            {
                case 1: return MRayPixelType<MR_R_FLOAT>{};         break;
                case 2: return MRayPixelType<MR_RG_FLOAT>{};        break;
                case 3: return MRayPixelType<MR_RGB_FLOAT>{};       break;
                case 4: return MRayPixelType<MR_RGBA_FLOAT>{};      break;
                default: break;
            }
            break;
        }
        default: break;
    }
    return MRayError("Uknown image pixel type");
}

Expected<OIIO::ImageSpec> ImageFileOIIO::PixelFormatToOIIO(const ImageHeader& header)
{
    MRayPixelEnum pixType = header.pixelType.Name();
    int nChannels = static_cast<int>(header.pixelType.ChannelCount());

    OIIO::TypeDesc td;
    switch(pixType)
    {
        using enum MRayPixelEnum;
        case MR_R8_UNORM:
        case MR_RG8_UNORM:
        case MR_RGB8_UNORM:
        case MR_RGBA8_UNORM:
        {
            td = OIIO::TypeDesc::UINT8;
            break;
        }
        case MR_R16_UNORM:
        case MR_RG16_UNORM:
        case MR_RGB16_UNORM:
        case MR_RGBA16_UNORM:
        {
            td = OIIO::TypeDesc::UINT16;
            break;
        }
        case MR_R8_SNORM:
        case MR_RG8_SNORM:
        case MR_RGB8_SNORM:
        case MR_RGBA8_SNORM:
        {
            td = OIIO::TypeDesc::INT8;
            break;
        }
        case MR_R16_SNORM:
        case MR_RG16_SNORM:
        case MR_RGB16_SNORM:
        case MR_RGBA16_SNORM:
        {
            td = OIIO::TypeDesc::INT16;
            break;
        }
        case MR_R_HALF:
        case MR_RG_HALF:
        case MR_RGB_HALF:
        case MR_RGBA_HALF:
        {
            td = OIIO::TypeDesc::HALF;
            break;
        }
        case MR_R_FLOAT:
        case MR_RG_FLOAT:
        case MR_RGB_FLOAT:
        case MR_RGBA_FLOAT:
        {
            td = OIIO::TypeDesc::FLOAT;
            break;
        }
        case MR_BC1_UNORM:
        case MR_BC2_UNORM:
        case MR_BC3_UNORM:
        case MR_BC4_UNORM:
        case MR_BC4_SNORM:
        case MR_BC5_UNORM:
        case MR_BC5_SNORM:
        case MR_BC6H_UFLOAT:
        case MR_BC6H_SFLOAT:
        case MR_BC7_UNORM:
        default:
            return MRayError("Unable to convert pixel type to OIIO type {}",
                             MRayPixelTypeStringifier::ToString(pixType));
    }
    return OIIO::ImageSpec(static_cast<int32_t>(header.dimensions[0]),
                           static_cast<int32_t>(header.dimensions[1]),
                           nChannels, td);
}

Expected<std::string> ImageFileOIIO::ColorSpaceToOIIO(const ColorSpacePack& pack)
{
    const auto&[gamma, type] = pack;
    std::string suffix = (gamma == 1.0f) ? "" : MRAY_FORMAT("{:1}", gamma);

    // TODO: these are probably wrong check it
    std::string prefix;
    switch(pack.second)
    {
        case MRayColorSpaceEnum::MR_ACES2065_1: prefix = "ACES2065-1";      break;
        case MRayColorSpaceEnum::MR_ACES_CG:    prefix = "ACEScg";          break;
        case MRayColorSpaceEnum::MR_REC_709:    prefix = "Rec709";          break;
        case MRayColorSpaceEnum::MR_REC_2020:   prefix = "Rec2020";         break;
        case MRayColorSpaceEnum::MR_DCI_P3:     prefix = "DCIP3";           break;
        case MRayColorSpaceEnum::MR_DEFAULT:    prefix = "scene_linear";    break;
        default: return MRayError("Unable to convert color space type to OIIO type {}",
                                  MRayColorSpaceStringifier::ToString(type));
    }
    return prefix + suffix;
}

Expected<ColorSpacePack> ImageFileOIIO::ColorSpaceToMRay(const std::string& oiioString)
{
    using namespace std::literals;
    using enum MRayColorSpaceEnum;
    using MapType = std::tuple<std::string_view, MRayColorSpaceEnum, Float>;
    using ArrayType = std::array<MapType, 6>;

    // TODO: Not complete, add later
    static constexpr ArrayType LookupList =
    {
        MapType{"ACES2065-1"sv,     MR_ACES2065_1,  Float(1)},
        MapType{"ACEScg"sv,         MR_ACES_CG,     Float(1)},
        MapType{"Rec709"sv,         MR_REC_709,     Float(2.222)},
        MapType{"sRGB"sv,           MR_REC_709,     Float(2.2)},
        MapType{"lin_srgb"sv,       MR_REC_709,     Float(1)},
        MapType{"adobeRGB"sv,       MR_REC_709,     Float(2.19922)},
    };

    for(const auto& checkType : LookupList)
    {
        if(std::get<0>(checkType) != oiioString) continue;

        return ColorSpacePack
        {
            std::get<2>(checkType),
            std::get<1>(checkType)
        };
    }
    return MRayError("Unable to convert OIIO type to color space type \"{}\"",
                     oiioString);

}

Expected<MRayPixelTypeRT> ImageFileOIIO::ConvertFormatToRequested(MRayPixelTypeRT pixFormat,
                                                                  ImageSubChannelType subChannels)
{
    if(subChannels == ImageSubChannelType::ALL) return pixFormat;

    MRayPixelEnum e = pixFormat.Name();
    bool compatibleChannels = false;
    switch(e)
    {
        using enum MRayPixelEnum;
        // We can slice RGBA to any type
        case MR_RGBA8_UNORM:
        case MR_RGBA16_UNORM:
        case MR_RGBA8_SNORM:
        case MR_RGBA16_SNORM:
        case MR_RGBA_HALF:
        case MR_RGBA_FLOAT:
            compatibleChannels = true;
            break;

        // Only without A
        case MR_RGB8_UNORM:
        case MR_RGB16_UNORM:
        case MR_RGB8_SNORM:
        case MR_RGB16_SNORM:
        case MR_RGB_HALF:
        case MR_RGB_FLOAT:
            compatibleChannels = !(subChannels == ImageSubChannelType::A   ||
                                   subChannels == ImageSubChannelType::BA  ||
                                   subChannels == ImageSubChannelType::GBA ||
                                   subChannels == ImageSubChannelType::RGBA);
            break;

        // Only RG
        case MR_RG8_UNORM:
        case MR_RG16_UNORM:
        case MR_RG8_SNORM:
        case MR_RG16_SNORM:
        case MR_RG_HALF:
        case MR_RG_FLOAT:
            compatibleChannels = (subChannels == ImageSubChannelType::R ||
                                  subChannels == ImageSubChannelType::G ||
                                  subChannels == ImageSubChannelType::RG);
            break;

        // Only R
        case MR_R8_UNORM:
        case MR_R16_UNORM:
        case MR_R8_SNORM:
        case MR_R16_SNORM:
        case MR_R_HALF:
        case MR_R_FLOAT:
            compatibleChannels = (subChannels == ImageSubChannelType::R);
            break;

        // Probably compressed which should not be here
        default:
            compatibleChannels = false;
            break;
    }

    // TODO: maybe more verbose error
    if(!compatibleChannels)
        return MRayError("Unable to convert image channels to "
                         "requested channels");

    // Now the actual convertion.
    // There is a pattern here (we can abuse the enum order etc.)
    // but it will be hard to maintain so just yolo switch here
    switch(e)
    {
        using enum MRayPixelEnum;
        case MR_R8_UNORM:
        case MR_RG8_UNORM:
        case MR_RGB8_UNORM:
        case MR_RGBA8_UNORM:
        switch(subChannels)
        {
            using enum ImageSubChannelType;
            case R:     return MRayPixelTypeRT(MRayPixelType<MR_R8_UNORM>{});
            case G:     return MRayPixelTypeRT(MRayPixelType<MR_R8_UNORM>{});
            case B:     return MRayPixelTypeRT(MRayPixelType<MR_R8_UNORM>{});
            case A:     return MRayPixelTypeRT(MRayPixelType<MR_R8_UNORM>{});

            case RG:    return MRayPixelTypeRT(MRayPixelType<MR_RG8_UNORM>{});
            case GB:    return MRayPixelTypeRT(MRayPixelType<MR_RG8_UNORM>{});
            case BA:    return MRayPixelTypeRT(MRayPixelType<MR_RG8_UNORM>{});

            case RGB:   return MRayPixelTypeRT(MRayPixelType<MR_RGB8_UNORM>{});
            case GBA:   return MRayPixelTypeRT(MRayPixelType<MR_RGB8_UNORM>{});

            case RGBA:  return MRayPixelTypeRT(MRayPixelType<MR_RGBA8_UNORM>{});
            default:    break;
        }
        break;
        //
        case MR_R16_UNORM:
        case MR_RG16_UNORM:
        case MR_RGB16_UNORM:
        case MR_RGBA16_UNORM:
        switch(subChannels)
        {
            using enum ImageSubChannelType;
            case R:     return MRayPixelTypeRT(MRayPixelType<MR_R16_UNORM>{});
            case G:     return MRayPixelTypeRT(MRayPixelType<MR_R16_UNORM>{});
            case B:     return MRayPixelTypeRT(MRayPixelType<MR_R16_UNORM>{});
            case A:     return MRayPixelTypeRT(MRayPixelType<MR_R16_UNORM>{});

            case RG:    return MRayPixelTypeRT(MRayPixelType<MR_RG16_UNORM>{});
            case GB:    return MRayPixelTypeRT(MRayPixelType<MR_RG16_UNORM>{});
            case BA:    return MRayPixelTypeRT(MRayPixelType<MR_RG16_UNORM>{});

            case RGB:   return MRayPixelTypeRT(MRayPixelType<MR_RGB16_UNORM>{});
            case GBA:   return MRayPixelTypeRT(MRayPixelType<MR_RGB16_UNORM>{});

            case RGBA:  return MRayPixelTypeRT(MRayPixelType<MR_RGBA16_UNORM>{});
            default:    break;
        }
        break;
        //
        case MR_R8_SNORM:
        case MR_RG8_SNORM:
        case MR_RGB8_SNORM:
        case MR_RGBA8_SNORM:
        switch(subChannels)
        {
            using enum ImageSubChannelType;
            case R:     return MRayPixelTypeRT(MRayPixelType<MR_R8_SNORM>{});
            case G:     return MRayPixelTypeRT(MRayPixelType<MR_R8_SNORM>{});
            case B:     return MRayPixelTypeRT(MRayPixelType<MR_R8_SNORM>{});
            case A:     return MRayPixelTypeRT(MRayPixelType<MR_R8_SNORM>{});

            case RG:    return MRayPixelTypeRT(MRayPixelType<MR_RG8_SNORM>{});
            case GB:    return MRayPixelTypeRT(MRayPixelType<MR_RG8_SNORM>{});
            case BA:    return MRayPixelTypeRT(MRayPixelType<MR_RG8_SNORM>{});

            case RGB:   return MRayPixelTypeRT(MRayPixelType<MR_RGB8_SNORM>{});
            case GBA:   return MRayPixelTypeRT(MRayPixelType<MR_RGB8_SNORM>{});

            case RGBA:  return MRayPixelTypeRT(MRayPixelType<MR_RGBA8_SNORM>{});
            default:    break;
        }
        break;
        //
        case MR_R16_SNORM:
        case MR_RG16_SNORM:
        case MR_RGB16_SNORM:
        case MR_RGBA16_SNORM:
        switch(subChannels)
        {
            using enum ImageSubChannelType;
            case R:     return MRayPixelTypeRT(MRayPixelType<MR_R16_SNORM>{});
            case G:     return MRayPixelTypeRT(MRayPixelType<MR_R16_SNORM>{});
            case B:     return MRayPixelTypeRT(MRayPixelType<MR_R16_SNORM>{});
            case A:     return MRayPixelTypeRT(MRayPixelType<MR_R16_SNORM>{});

            case RG:    return MRayPixelTypeRT(MRayPixelType<MR_RG16_SNORM>{});
            case GB:    return MRayPixelTypeRT(MRayPixelType<MR_RG16_SNORM>{});
            case BA:    return MRayPixelTypeRT(MRayPixelType<MR_RG16_SNORM>{});

            case RGB:   return MRayPixelTypeRT(MRayPixelType<MR_RGB16_SNORM>{});
            case GBA:   return MRayPixelTypeRT(MRayPixelType<MR_RGB16_SNORM>{});

            case RGBA:  return MRayPixelTypeRT(MRayPixelType<MR_RGBA16_SNORM>{});
            default:    break;
        }
        break;
        //
        case MR_R_HALF:
        case MR_RG_HALF:
        case MR_RGB_HALF:
        case MR_RGBA_HALF:
        switch(subChannels)
        {
            using enum ImageSubChannelType;
            case R:     return MRayPixelTypeRT(MRayPixelType<MR_R_HALF>{});
            case G:     return MRayPixelTypeRT(MRayPixelType<MR_R_HALF>{});
            case B:     return MRayPixelTypeRT(MRayPixelType<MR_R_HALF>{});
            case A:     return MRayPixelTypeRT(MRayPixelType<MR_R_HALF>{});

            case RG:    return MRayPixelTypeRT(MRayPixelType<MR_RG_HALF>{});
            case GB:    return MRayPixelTypeRT(MRayPixelType<MR_RG_HALF>{});
            case BA:    return MRayPixelTypeRT(MRayPixelType<MR_RG_HALF>{});

            case RGB:   return MRayPixelTypeRT(MRayPixelType<MR_RGB_HALF>{});
            case GBA:   return MRayPixelTypeRT(MRayPixelType<MR_RGB_HALF>{});

            case RGBA:  return MRayPixelTypeRT(MRayPixelType<MR_RGBA_HALF>{});
            default:    break;
        }
        break;
        //
        case MRayPixelEnum::MR_R_FLOAT:
        case MRayPixelEnum::MR_RG_FLOAT:
        case MRayPixelEnum::MR_RGB_FLOAT:
        case MRayPixelEnum::MR_RGBA_FLOAT:
        switch(subChannels)
        {
            using enum ImageSubChannelType;
            case R:     return MRayPixelTypeRT(MRayPixelType<MR_R_HALF>{});
            case G:     return MRayPixelTypeRT(MRayPixelType<MR_R_HALF>{});
            case B:     return MRayPixelTypeRT(MRayPixelType<MR_R_HALF>{});
            case A:     return MRayPixelTypeRT(MRayPixelType<MR_R_HALF>{});

            case RG:    return MRayPixelTypeRT(MRayPixelType<MR_RG_HALF>{});
            case GB:    return MRayPixelTypeRT(MRayPixelType<MR_RG_HALF>{});
            case BA:    return MRayPixelTypeRT(MRayPixelType<MR_RG_HALF>{});

            case RGB:   return MRayPixelTypeRT(MRayPixelType<MR_RGB_HALF>{});
            case GBA:   return MRayPixelTypeRT(MRayPixelType<MR_RGB_HALF>{});

            case RGBA:  return MRayPixelTypeRT(MRayPixelType<MR_RGBA_HALF>{});
            default:    break;
        }
        break;
        default: break;
    }
    // Code should not come here
    return MRayError("Unable to convert image channels to "
                     "requested channels");
}

ImageFileOIIO::ImageFileOIIO(const std::string& filePath,
              ImageSubChannelType subChannels,
              ImageIOFlags flags)
    : ImageFileBase(filePath, subChannels, flags)
{}

Expected<ImageHeader> ImageFileOIIO::ReadHeader()
{
    oiioFile = OIIO::ImageInput::open(filePath);
    if(!oiioFile) return MRayError("OIIO Error ({})", OIIO::geterror());

    OIIO::ImageSpec spec = oiioFile->spec();
    // Check sign conversion only uint types can be converted to signed
    // This is usefull for normal maps, we can directly utilize SNORM conversion
    // of the hardware instead of doing "r * 2 - 1" (where r is [0, 1))
    bool isSignConvertible = (spec.format == OIIO::TypeDesc::UINT16 ||
                              spec.format == OIIO::TypeDesc::UINT8);
    if(flags[ImageFlagTypes::LOAD_AS_SIGNED] && !isSignConvertible)
        return MRayError("Image type (OIIO \"{}\") is not sign convertible.({})",
                         spec.format, filePath);
    else if(flags[ImageFlagTypes::LOAD_AS_SIGNED])
    {
        spec.format = (spec.format == OIIO::TypeDesc::UINT16)
                        ? OIIO::TypeDesc::INT16
                        : OIIO::TypeDesc::INT8;
    }

    // Determine pixel format
    Expected<MRayPixelTypeRT> pixFormatE = PixelFormatToMRay(spec);
    if(pixFormatE.has_error()) return pixFormatE.error();
    const MRayPixelTypeRT& pixFormat = pixFormatE.value();
    originalPixType = pixFormat;

    Expected<MRayPixelTypeRT> convPixFormatE = ConvertFormatToRequested(pixFormat, subChannels);
    if(convPixFormatE.has_error()) return convPixFormatE.error();
    const MRayPixelTypeRT& convPixFormat = convPixFormatE.value();

    // Determine dimension
    auto dimension = Vector3ui(spec.width, spec.height, spec.depth);

    // Determine color space
    OIIO::ustring colorSpaceOIIO;
    bool hasColorSpace = spec.getattribute("oiio:ColorSpace", OIIO::TypeString,
                                           &colorSpaceOIIO);

    Expected<ColorSpacePack> colorSpaceE = ColorSpaceToMRay(std::string(colorSpaceOIIO));
    if(hasColorSpace && !colorSpaceE.has_value())
    {
        MRayError e = colorSpaceE.error();
        e.AppendInfo(MRAY_FORMAT("({})", filePath));
        return e;
    }
    // If "oiio:ColorSpace" query is push MR_DEFAULT color space with linear gamma
    // SceneLoader may check user defined color space if applicable
    Pair defaultColorSpace = Pair(Float(1), MRayColorSpaceEnum::MR_DEFAULT);
    const auto& colorSpace = colorSpaceE.value_or(defaultColorSpace);

    // TODO: Support tiled images
    if(spec.tile_width != 0 || spec.tile_height != 0)
        return MRayError("Tiled images are not currently supported ({}).",
                         filePath);

    // TODO: Do a conversion to an highest precision channel for these
    if(!spec.channelformats.empty())
        return MRayError("Channel-specific formats are not supported ({}).",
                         filePath);

    // We do not support texture arrays
    if(spec.format.is_array())
        return MRayError("Arrayed per-pixel formats are not supported ({}).",
                         filePath);

    // Mipmap determination
    // Iterate through mips and find the amount
    int32_t mipCount = 0;
    for(; oiioFile->seek_subimage(0, mipCount); mipCount++);

    if(mipCount > Image::MAX_MIP_COUNT)
        return MRayError("Mip map count exceeds the maximum amount ({}) on file \"{}\"",
                         Image::MAX_MIP_COUNT, filePath);

    bool isDefaultColorSpace = flags[ImageFlagTypes::DISREGARD_COLOR_SPACE];
    using enum MRayColorSpaceEnum;
    header = ImageHeader
    {
        .dimensions = dimension,
        .mipCount = static_cast<uint32_t>(mipCount),
        .pixelType = convPixFormat,
        .colorSpace = Pair((isDefaultColorSpace) ? Float(1) : colorSpace.first,
                           (isDefaultColorSpace) ? MR_DEFAULT : colorSpace.second)
    };
    return header;
}

Expected<Image> ImageFileOIIO::ReadImage()
{
    Image result;
    result.header = header;

    // Pre-check the spec
    const OIIO::ImageSpec& spec = oiioFile->spec(0, 0);

    // Again determine the read format
    OIIO::TypeDesc readFormat = spec.format;
    if(flags[ImageFlagTypes::LOAD_AS_SIGNED])
    {
        readFormat = (readFormat == OIIO::TypeDesc::UINT16)
                        ? OIIO::TypeDesc::INT16
                        : OIIO::TypeDesc::INT8;
    }

    // Calculate read channel count according to channel expansion
    size_t readChannelCount = header.pixelType.ChannelCount();
    bool doChannelExpand = (flags[ImageFlagTypes::TRY_3C_4C_CONVERSION] &&
                            readChannelCount == 3u);
    readChannelCount = (doChannelExpand) ? (readChannelCount + 1) : readChannelCount;
    MRayPixelTypeRT dataPixelType = (doChannelExpand)
                                        ? ImageLoaderI::TryExpandTo4CFormat(header.pixelType)
                                        : header.pixelType;

    // Finally find the channel range
    Vector2i channelRange = ImageLoaderI::CalculateChannelRange(subChannels);
    // Put [0,0) when it is the exact pixel read,
    // OIIO may have different (optimized) path for this
    channelRange = (header.pixelType == originalPixType)
                    ? Vector2i(0)
                    : channelRange;

    // Read mip by mip
    for(uint32_t mipLevel = 0; mipLevel < header.mipCount; mipLevel++)
    {
        OIIO::ImageSpec mipSpec = oiioFile->spec(0, mipLevel);
        assert(mipSpec.format == spec.format);

        // Calculate the read buffer stride
        auto channelSize = static_cast<OIIO::stride_t>(mipSpec.channel_bytes());
        OIIO::stride_t xStride = (doChannelExpand)
                                    ? (4 * channelSize)
                                    : OIIO::AutoStride;

        // Allocate the expanded (or non-expanded) buffer and directly load into it
        size_t scanLineSize = (static_cast<size_t>(mipSpec.width) *
                               readChannelCount * mipSpec.channel_bytes());
        size_t totalSize = scanLineSize * static_cast<size_t>(spec.height);

        // Due to type safety we need to construct pixels using the formatted
        // type, we need to "visit" the variant to correctly construct the
        // type-ereased transient data
        TransientData pixels = std::visit([totalSize](auto&& v)
        {
            using T = std::remove_cvref_t<decltype(v)>::Type;
            return TransientData(std::in_place_type_t<T>{}, totalSize);
        }, dataPixelType);

        // Read inverted, OIIO has DirectX
        // (or most image processing literature) style
        // Point towards the last scanline, and put stride as negative
        Byte* lastScanlinePtr = (pixels.AccessAs<Byte>().data() +
                                 (spec.height - 1) * scanLineSize);
        // Now we can read the file directly flipped and with proper format etc. etc.
        OIIO::stride_t oiioScanLineSize = static_cast<OIIO::stride_t>(scanLineSize);

        // Directly read the data
        if(!oiioFile->read_image(0, mipLevel,
                                 channelRange[0], channelRange[1],
                                 readFormat, lastScanlinePtr,
                                 xStride, -oiioScanLineSize))
            return MRayError("OIIO Error ({})", oiioFile->geterror());

        // Push the mip data to vector
        result.imgData.emplace_back(ImageMip
        {
            .mipSize = Vector3ui(mipSpec.width, mipSpec.height, mipSpec.depth),
            .pixels = std::move(pixels)
        });
    };
    return result;
}