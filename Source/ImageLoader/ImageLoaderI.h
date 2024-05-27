#pragma once

#pragma once

#include <vector>
#include <string>
#include <memory>

#include "Core/DataStructures.h"
#include "Core/Vector.h"
#include "Core/Flag.h"
#include "Core/MRayDataType.h"
#include "TransientPool/TransientPool.h"

enum class ImageType
{
    PNG,
    JPG,
    BMP,
    HDR,
    EXR
};

enum class ImageChannelType
{
    R, G, B, A,
    RG, GB, BA,
    RGB, GBA,
    RGBA
};

enum class ImageFlagTypes
{
    // Enforce singed loading for unsigned types
    // Useful of normal maps, given a uint8_t 3 channel type
    // It should be considered as unsigned type,
    // Data will be converted as if the conversion is as follows: (uint8_t)
    //
    //  using T = uint8_t;
    //  constexpr T mid = static_cast<T>(0x1u << ((sizeof(T) * BYTE_BITS) - 1));
    //  t -= mid;
    //  return int8_t(t);
    //
    // This should be equavilent of the conversion of "r * 255 - 128"
    // The difference is that 2's complement integers are not represented like this
    // Devices (CUDA currently) does not proparly normalize these integers
    LOAD_AS_SIGNED = 0,
    // Due to HW limitations (CUDA does not directly support 3 channel textures),
    // We might need to pad a 4th channel to the image.
    // This enables that
    TRY_3C_4C_CONVERSION = 1,
    // Data is probably encoded exactly and/or raw data
    // (probably normal map, roughness etc.)
    // Do not try to convert to the system defined common renderer color space
    DISREGARD_COLOR_SPACE = 2,

    END
};

using ImageIOFlags = Flag<ImageFlagTypes>;

// TODO: Images are just as complex as the renderer itself...
// (Too many conventions standarts etc..)
// OIIO helps tramendously however, it is just a passthrough.
//
// So we restrict the data to some basic formats
//  - Deep images are ignored (or maybe only the 1st depth is read)
//  - If image is tiled hopefully it is read via scanline methods
template<uint32_t D>
requires(D == 2 || D == 3)
struct ImageHeader
{
    Vector<D, uint32_t>     dimensions;
    uint32_t                mipCount;
    MRayPixelTypeRT         pixelType;
    //
    MRayColorSpaceEnum      colorSpace;
    Float                   gamma;
};

template<uint32_t D>
requires(D == 2 || D == 3)
struct ImageMip
{
    Vector<D, uint32_t>     mipSize;
    TransientData           pixels;
};

template<uint32_t D>
requires(D == 2 || D == 3)
struct Image
{
    // This array is just to elide heap, so max 1Mx1M texture size
    static constexpr int32_t MAX_MIP_COUNT = 20;
    using MipMapArray = StaticVector<ImageMip<D>, MAX_MIP_COUNT>;

    ImageHeader<D>  header;
    MipMapArray     imgData;
};

template<uint32_t D>
requires(D == 2 || D == 3)
struct WriteImage
{
    ImageHeader<D>      header;
    uint32_t            depth;
    Span<const Byte>    pixels;
};

class ImageLoaderI
{
    public:
    static MRayPixelTypeRT      TryExpandTo4CFormat(MRayPixelTypeRT pf);

    public:
        virtual                 ~ImageLoaderI() = default;

        // Read Functions
        // This will be expanded later
        // (it will have a similar
        // interface of OIIO)
        virtual Expected<Image<2>>          ReadImage2D(const std::string& filePath,
                                                        ImageIOFlags = ImageIOFlags()) const = 0;
        // Read subportion of the image
        // i.e., given RGBA image, you can read RG, portion as a 2 channel image
        // Only contiguous channels are supported
        virtual Expected<Image<2>>          ReadImageSubChannel(const std::string& filePath,
                                                                ImageChannelType,
                                                                ImageIOFlags = ImageIOFlags()) const = 0;

        // Lightweight version (if OIIO supports), image files are read and only size/type
        // information will be provided. Thus, tracer can preallocate the required memory.
        virtual Expected<ImageHeader<2>>    ReadImageHeader2D(const std::string& filePath,
                                                              ImageIOFlags = ImageIOFlags()) const = 0;
        // Write Functions
        virtual MRayError           WriteImage2D(const WriteImage<2>&,
                                                 const std::string& filePath,
                                                 ImageType extension,
                                                 ImageIOFlags = ImageIOFlags()) const = 0;
        // TODO: Add 3D variants

};

inline MRayPixelTypeRT ImageLoaderI::TryExpandTo4CFormat(MRayPixelTypeRT pt)
{
    MRayPixelEnum pf = pt.Name();
    using enum MRayPixelEnum;
    switch(pf)
    {
        case MR_RGB8_UNORM:     return MRayPixelTypeRT(MRayPixelType<MR_RGBA8_UNORM>{});
        case MR_RGB16_UNORM:    return MRayPixelTypeRT(MRayPixelType<MR_RGBA16_UNORM>{});
        case MR_RGB8_SNORM:     return MRayPixelTypeRT(MRayPixelType<MR_RGBA8_SNORM>{});
        case MR_RGB16_SNORM:    return MRayPixelTypeRT(MRayPixelType<MR_RGBA16_SNORM>{});
        case MR_RGB_HALF:       return MRayPixelTypeRT(MRayPixelType<MR_RGBA_HALF>{});
        case MR_RGB_FLOAT:      return MRayPixelTypeRT(MRayPixelType<MR_RGBA_FLOAT>{});
        default:                return pt;
    }
}

//inline int8_t ImageLoaderI::ChannelTypeToChannelIndex(ImageChannelType channel)
//{
//    return static_cast<int8_t>(channel);
//}
//
//inline int8_t ImageLoaderI::FormatToChannelCount(PixelFormat pf)
//{
//    switch(pf)
//    {
//        case PixelFormat::MR_UNORM_4x8:
//        case PixelFormat::R16_UNORM:
//        case PixelFormat::R8_SNORM:
//        case PixelFormat::R16_SNORM:
//        case PixelFormat::R_HALF:
//        case PixelFormat::R_FLOAT:
//            return 1;
//        case PixelFormat::RG8_UNORM:
//        case PixelFormat::RG16_UNORM:
//        case PixelFormat::RG8_SNORM:
//        case PixelFormat::RG16_SNORM:
//        case PixelFormat::RG_HALF:
//        case PixelFormat::RG_FLOAT:
//            return 2;
//        case PixelFormat::RGB8_UNORM:
//        case PixelFormat::RGB16_UNORM:
//        case PixelFormat::RGB8_SNORM:
//        case PixelFormat::RGB16_SNORM:
//        case PixelFormat::RGB_HALF:
//        case PixelFormat::RGB_FLOAT:
//            return 3;
//        case PixelFormat::RGBA8_UNORM:
//        case PixelFormat::RGBA16_UNORM:
//        case PixelFormat::RGBA8_SNORM:
//        case PixelFormat::RGBA16_SNORM:
//        case PixelFormat::RGBA_HALF:
//        case PixelFormat::RGBA_FLOAT:
//            return 4;
//        default:
//            // TODO: Add bc compression channels
//            return std::numeric_limits<int8_t>::max();
//    }
//}
//
//inline size_t ImageLoaderI::FormatToPixelSize(PixelFormat pf)
//{
//    size_t channelSize = FormatToChannelSize(pf);
//    // Yolo switch
//    switch(pf)
//    {
//        // 1 Channels
//        case PixelFormat::R8_UNORM:
//        case PixelFormat::R8_SNORM:
//        case PixelFormat::R16_UNORM:
//        case PixelFormat::R16_SNORM:
//        case PixelFormat::R_HALF:
//        case PixelFormat::R_FLOAT:
//            return channelSize * 1;
//        // 2 Channels
//        case PixelFormat::RG8_UNORM:
//        case PixelFormat::RG8_SNORM:
//        case PixelFormat::RG16_UNORM:
//        case PixelFormat::RG16_SNORM:
//        case PixelFormat::RG_HALF:
//        case PixelFormat::RG_FLOAT:
//            return channelSize * 2;
//        // 3 Channels
//        case PixelFormat::RGB8_UNORM:
//        case PixelFormat::RGB8_SNORM:
//        case PixelFormat::RGB16_UNORM:
//        case PixelFormat::RGB16_SNORM:
//        case PixelFormat::RGB_HALF:
//        case PixelFormat::RGB_FLOAT:
//            return channelSize * 3;
//                // 3 Channels
//        case PixelFormat::RGBA8_UNORM:
//        case PixelFormat::RGBA8_SNORM:
//        case PixelFormat::RGBA16_UNORM:
//        case PixelFormat::RGBA16_SNORM:
//        case PixelFormat::RGBA_HALF:
//        case PixelFormat::RGBA_FLOAT:
//            return channelSize * 4;
//        // BC Types
//        // TODO: Implement these
//        case PixelFormat::BC1_U:    return 0;
//        case PixelFormat::BC2_U:    return 0;
//        case PixelFormat::BC3_U:    return 0;
//        case PixelFormat::BC4_U:    return 0;
//        case PixelFormat::BC4_S:    return 0;
//        case PixelFormat::BC5_U:    return 0;
//        case PixelFormat::BC5_S:    return 0;
//        case PixelFormat::BC6H_U:   return 0;
//        case PixelFormat::BC6H_S:   return 0;
//        case PixelFormat::BC7_U:    return 0;
//        // Unknown Type
//        case PixelFormat::END:
//        default:
//            return 0;
//
//    }
//}
//
//inline size_t ImageLoaderI::FormatToChannelSize(PixelFormat pf)
//{
//    // Yolo switch
//    switch(pf)
//    {
//        // SNORM & UNORM INT8 Types
//        case PixelFormat::R8_UNORM:
//        case PixelFormat::R8_SNORM:     return sizeof(uint8_t);
//        case PixelFormat::RG8_UNORM:
//        case PixelFormat::RG8_SNORM:    return sizeof(uint8_t);
//        case PixelFormat::RGB8_UNORM:
//        case PixelFormat::RGB8_SNORM:   return sizeof(uint8_t);
//        case PixelFormat::RGBA8_UNORM:
//        case PixelFormat::RGBA8_SNORM:  return sizeof(uint8_t);
//        // SNORM & UNORM INT16 Types
//        case PixelFormat::R16_UNORM:
//        case PixelFormat::R16_SNORM:    return sizeof(uint16_t);
//        case PixelFormat::RG16_UNORM:
//        case PixelFormat::RG16_SNORM:   return sizeof(uint16_t);
//        case PixelFormat::RGB16_UNORM:
//        case PixelFormat::RGB16_SNORM:  return sizeof(uint16_t);
//        case PixelFormat::RGBA16_UNORM:
//        case PixelFormat::RGBA16_SNORM: return sizeof(uint16_t);
//        // Half Types
//        case PixelFormat::R_HALF:       return sizeof(uint16_t);
//        case PixelFormat::RG_HALF:      return sizeof(uint16_t);
//        case PixelFormat::RGB_HALF:     return sizeof(uint16_t);
//        case PixelFormat::RGBA_HALF:    return sizeof(uint16_t);
//
//        case PixelFormat::R_FLOAT:      return sizeof(float);
//        case PixelFormat::RG_FLOAT:     return sizeof(float);
//        case PixelFormat::RGB_FLOAT:    return sizeof(float);
//        case PixelFormat::RGBA_FLOAT:   return sizeof(float);
//        // BC Types
//        // TODO: Implement these
//        case PixelFormat::BC1_U:    return 0;
//        case PixelFormat::BC2_U:    return 0;
//        case PixelFormat::BC3_U:    return 0;
//        case PixelFormat::BC4_U:    return 0;
//        case PixelFormat::BC4_S:    return 0;
//        case PixelFormat::BC5_U:    return 0;
//        case PixelFormat::BC5_S:    return 0;
//        case PixelFormat::BC6H_U:   return 0;
//        case PixelFormat::BC6H_S:   return 0;
//        case PixelFormat::BC7_U:    return 0;
//        // Unknown Type
//        case PixelFormat::END:
//        default:
//            return 0;
//
//    }
//}
//
//inline PixelFormat ImageLoaderI::SignConvertedFormat(PixelFormat pf)
//{
//    switch(pf)
//    {
//        case PixelFormat::R8_UNORM:     return PixelFormat::R8_SNORM;
//        case PixelFormat::RG8_UNORM:    return PixelFormat::RG8_SNORM;
//        case PixelFormat::RGB8_UNORM:   return PixelFormat::RGB8_SNORM;
//        case PixelFormat::RGBA8_UNORM:  return PixelFormat::RGBA8_SNORM;
//        case PixelFormat::R16_UNORM:    return PixelFormat::R16_SNORM;
//        case PixelFormat::RG16_UNORM:   return PixelFormat::RG16_SNORM;
//        case PixelFormat::RGB16_UNORM:  return PixelFormat::RGB16_SNORM;
//        case PixelFormat::RGBA16_UNORM: return PixelFormat::RGBA16_SNORM;
//
//        case PixelFormat::R8_SNORM:     return PixelFormat::R8_UNORM;
//        case PixelFormat::RG8_SNORM:    return PixelFormat::RG8_UNORM;
//        case PixelFormat::RGB8_SNORM:   return PixelFormat::RGB8_UNORM;
//        case PixelFormat::RGBA8_SNORM:  return PixelFormat::RGBA8_UNORM;
//        case PixelFormat::R16_SNORM:    return PixelFormat::R16_UNORM;
//        case PixelFormat::RG16_SNORM:   return PixelFormat::RG16_UNORM;
//        case PixelFormat::RGB16_SNORM:  return PixelFormat::RGB16_UNORM;
//        case PixelFormat::RGBA16_SNORM: return PixelFormat::RGBA16_UNORM;
//
//        default: return PixelFormat::END;
//    }
//}
//
//inline bool ImageLoaderI::IsSignConvertible(PixelFormat pf)
//{
//    switch(pf)
//    {
//        case PixelFormat::R8_UNORM:
//        case PixelFormat::RG8_UNORM:
//        case PixelFormat::RGB8_UNORM:
//        case PixelFormat::RGBA8_UNORM:
//        case PixelFormat::R16_UNORM:
//        case PixelFormat::RG16_UNORM:
//        case PixelFormat::RGB16_UNORM:
//        case PixelFormat::RGBA16_UNORM:
//        case PixelFormat::R8_SNORM:
//        case PixelFormat::RG8_SNORM:
//        case PixelFormat::RGB8_SNORM:
//        case PixelFormat::RGBA8_SNORM:
//        case PixelFormat::R16_SNORM:
//        case PixelFormat::RG16_SNORM:
//        case PixelFormat::RGB16_SNORM:
//        case PixelFormat::RGBA16_SNORM:
//            return true;
//        default: return false;
//    }
//}
//
//inline bool ImageLoaderI::HasSignConversion(PixelFormat toFormat, PixelFormat fromFormat)
//{
//    switch(toFormat)
//    {
//        case PixelFormat::R8_UNORM:
//        case PixelFormat::RG8_UNORM:
//        case PixelFormat::RGB8_UNORM:
//        case PixelFormat::RGBA8_UNORM:
//        case PixelFormat::R16_UNORM:
//        case PixelFormat::RG16_UNORM:
//        case PixelFormat::RGB16_UNORM:
//        case PixelFormat::RGBA16_UNORM:
//        {
//            return (fromFormat == PixelFormat::R8_SNORM     ||
//                    fromFormat == PixelFormat::RG8_SNORM    ||
//                    fromFormat == PixelFormat::RGB8_SNORM   ||
//                    fromFormat == PixelFormat::RGBA8_SNORM  ||
//                    fromFormat == PixelFormat::R16_SNORM    ||
//                    fromFormat == PixelFormat::RG16_SNORM   ||
//                    fromFormat == PixelFormat::RGB16_SNORM  ||
//                    fromFormat == PixelFormat::RGBA16_SNORM);
//        }
//
//        case PixelFormat::R8_SNORM:
//        case PixelFormat::RG8_SNORM:
//        case PixelFormat::RGB8_SNORM:
//        case PixelFormat::RGBA8_SNORM:
//        case PixelFormat::R16_SNORM:
//        case PixelFormat::RG16_SNORM:
//        case PixelFormat::RGB16_SNORM:
//        case PixelFormat::RGBA16_SNORM:
//        {
//            return (fromFormat == PixelFormat::R8_UNORM     ||
//                    fromFormat == PixelFormat::RG8_UNORM    ||
//                    fromFormat == PixelFormat::RGB8_UNORM   ||
//                    fromFormat == PixelFormat::RGBA8_UNORM  ||
//                    fromFormat == PixelFormat::R16_UNORM    ||
//                    fromFormat == PixelFormat::RG16_UNORM   ||
//                    fromFormat == PixelFormat::RGB16_UNORM  ||
//                    fromFormat == PixelFormat::RGBA16_UNORM);
//        }
//        default: return false;
//    }
//}
//

//
//inline bool ImageLoaderI::Is4CExpandable(PixelFormat pf)
//{
//    switch(pf)
//    {
//        case PixelFormat::RGB8_UNORM:
//        case PixelFormat::RGB16_UNORM:
//        case PixelFormat::RGB8_SNORM:
//        case PixelFormat::RGB16_SNORM:
//        case PixelFormat::RGB_HALF:
//        case PixelFormat::RGB_FLOAT:
//            return true;
//        default: return false;
//    }
//}
