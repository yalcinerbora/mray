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

ImageFileOIIO::ImageFileOIIO(const std::string& filePath,
              ImageSubChannelType subChannels,
              ImageIOFlags flags)
    : ImageFileBase(filePath, subChannels, flags)
{}

Expected<ImageHeader> ImageFileOIIO::ReadHeader()
{
    oiioFile = OIIO::ImageInput::open(filePath);
    if(!oiioFile) return MRayError("OIIO Error ({})", OIIO::geterror());

    const OIIO::ImageSpec& spec = oiioFile->spec();

    // Determine color space
    Expected<MRayPixelTypeRT> pixFormatE = PixelFormatToMRay(spec);
    if(!pixFormatE.has_value()) return pixFormatE.error();
    const MRayPixelTypeRT& pixFormat = pixFormatE.value();

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
    const auto& colorSpace = colorSpaceE.value_or(Pair(Float(1),
                                                       MRayColorSpaceEnum::MR_DEFAULT));

    // TODO: Support tiled images
    if(spec.tile_width != 0 || spec.tile_height != 0)
        return MRayError("Tiled images are not currently supported ({}).",
                         filePath);

    // TODO: Do a conversion to an highest precision channel for these
    if(!spec.channelformats.empty())
        return MRayError("Channel-specific formats are not supported ({}).",
                         filePath);
    // Is this for deep images??
    // Or mip
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
    return ImageHeader
    {
        .dimensions = dimension,
        .mipCount = static_cast<uint32_t>(mipCount),
        .pixelType = pixFormat,
        .colorSpace = Pair((isDefaultColorSpace) ? Float(1) : colorSpace.first,
                           (isDefaultColorSpace) ? MR_DEFAULT : colorSpace.second)
    };
}

Expected<Image> ImageFileOIIO::ReadImage()
{
    Image result;
    result.header = header;
    for(uint32_t mipLevel = 0; mipLevel < header.mipCount; mipLevel++)
    {
        // Calculate the final spec according to the flags
        // channel expand and sign convert..
        OIIO::ImageSpec spec = oiioFile->spec(0, mipLevel);
        OIIO::ImageSpec finalSpec = spec;

        // Check if sign convertible
        OIIO::TypeDesc readFormat = spec.format;
        if(flags[ImageFlagTypes::LOAD_AS_SIGNED] && !ImageLoaderI::IsSignConvertible(header.pixelType))
            return MRayError("Image type is not sign convertible.({})", filePath);
        else if(flags[ImageFlagTypes::LOAD_AS_SIGNED])
        {
            readFormat = (spec.format == OIIO::TypeDesc::UINT16)
                            ? OIIO::TypeDesc::INT16
                            : OIIO::TypeDesc::INT8;
            finalSpec.format = readFormat;
        }

        // Find the x stride to do a channel expand
        bool doChannelExpand = (header.pixelType.ChannelCount() == 3 &&
                                flags[ImageFlagTypes::TRY_3C_4C_CONVERSION]);
        int nChannels = (doChannelExpand) ? (spec.nchannels + 1) : (spec.nchannels);
        OIIO::stride_t xStride = (doChannelExpand)
                                    ? (nChannels * static_cast<OIIO::stride_t>(readFormat.size()))
                                    : (OIIO::AutoStride);
        // Change the final spec as well for color convert
        if(doChannelExpand) finalSpec.nchannels = nChannels;

        // Allocate the expanded (or non-expanded) buffer and directly load into it
        size_t scanLineSize = (static_cast<size_t>(spec.width) *
                               static_cast<size_t>(nChannels) *
                               readFormat.size());
        size_t totalSize = scanLineSize * static_cast<size_t>(spec.height);
        TransientData pixels(std::in_place_type_t<Byte>{}, totalSize);
        Byte* dataLastElement = (pixels.AccessAs<Byte>().data() +
                                 (spec.height - 1) * scanLineSize);
        // Now we can read the file directly flipped and with proper format etc. etc.
        OIIO::stride_t oiioScanLineSize = static_cast<OIIO::stride_t>(scanLineSize);

        // Directly read the compressed data,
        // OIIO uncompress DDS when anything is not auto stride
        // This means we need to flip it manually (maybe attaching a view parameter)
        if(!oiioFile->read_image(0, mipLevel, readFormat,
                                 dataLastElement, xStride,
                                 -oiioScanLineSize))
            return MRayError("OIIO Error ({})", oiioFile->geterror());

        // Re-adjust the pixelFormat (we may have done channel expand and sign convert
        Expected<MRayPixelTypeRT> pixFormatFinalE = PixelFormatToMRay(finalSpec);
        if(!pixFormatFinalE.has_value()) return pixFormatFinalE.error();

        result.imgData.emplace_back(ImageMip
        {
            .mipSize = Vector3ui(spec.width, spec.height, spec.depth),
            .pixels = std::move(pixels)
        });
    };
    return result;
}
