#include "ImageLoader.h"
#include "Core/MRayDataType.h"

#include <execution>
#include <algorithm>
#include <array>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

bool IsSignConvertible(MRayPixelTypeRT pf)
{
    MRayPixelEnum pixType = pf.Name();
    switch(pixType)
    {
        using enum MRayPixelEnum;
        case MR_R8_UNORM:
        case MR_RG8_UNORM:
        case MR_RGB8_UNORM:
        case MR_RGBA8_UNORM:
        case MR_R16_UNORM:
        case MR_RG16_UNORM:
        case MR_RGB16_UNORM:
        case MR_RGBA16_UNORM:
            return true;
        default:
            return false;
    }
}

Expected<MRayPixelTypeRT> ImageLoader::OIIOImageSpecToPixelFormat(const OIIO::ImageSpec& spec)
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

Expected<OIIO::ImageSpec> ImageLoader::PixelFormatToOIIOImageSpec(const ImageHeader<2>& header)
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

Expected<Image<2>> ImageLoader::ReadImage2D(const std::string& filePath,
                                            const ImageIOFlags flags) const
{
    auto inFile = OIIO::ImageInput::open(filePath);
    if(!inFile) return MRayError("OIIO Error ({})", OIIO::geterror());

    // Check the spec
    const OIIO::ImageSpec& spec = inFile->spec();

    Expected<MRayPixelTypeRT> pixFormatE = OIIOImageSpecToPixelFormat(spec);
    if(!pixFormatE.has_value()) return pixFormatE.error();

    const MRayPixelTypeRT& pixFormat = pixFormatE.value();
    auto dimension = Vector2ui(spec.width, spec.height);

    // TODO: Do a conversion to an highest precision channel for these
    if(!spec.channelformats.empty())
        return MRayError("Channel-specific formats are not supported.({})",
                         filePath);
    // Is this for deep images??
    if(spec.format.is_array())
        return MRayError("Arrayed per-pixel formats are not supported.({})",
                         filePath);

    // Calculate the final spec according to the flags
    // channel expand and sign convert..
    OIIO::ImageSpec finalSpec = spec;

    // Check if sign convertible
    OIIO::TypeDesc readFormat = spec.format;
    if(flags[ImageFlagTypes::LOAD_AS_SIGNED] && !IsSignConvertible(pixFormat))
        return MRayError("Image type is not sign convertible.({})",
                         filePath);
    else if(flags[ImageFlagTypes::LOAD_AS_SIGNED])
    {
        readFormat = (spec.format == OIIO::TypeDesc::UINT16)
                        ? OIIO::TypeDesc::INT16
                        : OIIO::TypeDesc::INT8;
        finalSpec.format = readFormat;
    }

    // Find the x stride to do a channel expand
    bool doChannelExpand = (pixFormat.ChannelCount() == 3 && flags[ImageFlagTypes::TRY_3C_4C_CONVERSION]);
    int nChannels = (doChannelExpand) ? (spec.nchannels + 1) : (spec.nchannels);
    OIIO::stride_t xStride = (doChannelExpand)
                                ? (nChannels * readFormat.size())
                                : (OIIO::AutoStride);
    // Change the final spec as well for color convert
    if(doChannelExpand) finalSpec.nchannels = nChannels;

    // Allocate the expanded (or non-expanded) buffer and directly load into it
    MRayInput input(std::in_place_type_t<Byte>{},
                    spec.width * spec.height * nChannels * readFormat.size());
    OIIO::stride_t scanLineSize = spec.width * nChannels * readFormat.size();
    Byte* dataLastElement = input.AccessAs<Byte>().data() + (dimension[1] - 1) * scanLineSize;
    // Now we can read the file directly flipped and with proper format etc. etc.
    if(!inFile->read_image(readFormat, dataLastElement, xStride, -scanLineSize))
        return MRayError("OIIO Error ({})", inFile->geterror());

    // Re-adjust the pixelFormat (we may have done channel expand and sign convert
    Expected<MRayPixelTypeRT> pixFormatFinalE = OIIOImageSpecToPixelFormat(finalSpec);
    if(!pixFormatFinalE.has_value())
        return pixFormatFinalE.error();

    return Image<2>
    {
        .header = ImageHeader<2>
        {
            .dimensions = dimension,
            // TODO: Mip count
            .mipCount = 1,
            .pixelType = pixFormatFinalE.value()
        },
        .pixels = std::move(input)
    };
}

Expected<Image<2>> ImageLoader::ReadImageSubChannel(const std::string&,
                                                    ImageChannelType,
                                                    const ImageIOFlags) const
{
    return MRayError::OK;
}

Expected<ImageHeader<2>> ImageLoader::ReadImageHeader2D(const std::string&,
                                                        const ImageIOFlags) const
{
    return MRayError::OK;
}

MRayError ImageLoader::WriteImage2D(const Image<2>& imgIn,
                                    const std::string& filePath,
                                    ImageType extension,
                                    const ImageIOFlags flags) const
{
    auto out = OIIO::ImageOutput::create(filePath);

    Expected<OIIO::ImageSpec> specE = PixelFormatToOIIOImageSpec(imgIn.header);
    if(!specE.has_value()) return specE.error();

    const OIIO::ImageSpec& spec = specE.value();
    OIIO::stride_t scanLineSize = static_cast<OIIO::stride_t>(spec.scanline_bytes());

    const Byte* dataLastElement = imgIn.pixels.AccessAs<Byte>().data();
    dataLastElement += (imgIn.header.dimensions[1] - 1) * scanLineSize;

    // TODO: properly write an error check/out code for these.
    if(!out->open(filePath, spec))
        return MRayError("OIIO Error ({})", out->geterror());
    if(!out->write_image(spec.format, dataLastElement, OIIO::AutoStride, -scanLineSize))
        return MRayError("OIIO Error ({})", out->geterror());
    if(!out->close())
        return MRayError("OIIO Error ({})", out->geterror());

    return MRayError::OK;
}
