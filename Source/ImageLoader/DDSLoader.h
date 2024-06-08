#include "ImageLoaderI.h"

#include "Core/BitFunctions.h"
#include "Core/Flag.h"
#include "Core/DataStructures.h"

#include <OpenImageIO/imageio.h>

#include <filesystem>

// Unfortunately, OIIO does not load DDS in raw format :(
// So writing a DDS Reader
// Subset only, only compressed formats
// Rest will be delegated to the OIIO
// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide

namespace DDSLoader
{

enum class FormatDXGI : uint32_t
{
    UNKNOWN = 0,
    R32G32B32A32_TYPELESS = 1,
    R32G32B32A32_FLOAT = 2,
    R32G32B32A32_UINT = 3,
    R32G32B32A32_SINT = 4,
    R32G32B32_TYPELESS = 5,
    R32G32B32_FLOAT = 6,
    R32G32B32_UINT = 7,
    R32G32B32_SINT = 8,
    R16G16B16A16_TYPELESS = 9,
    R16G16B16A16_FLOAT = 10,
    R16G16B16A16_UNORM = 11,
    R16G16B16A16_UINT = 12,
    R16G16B16A16_SNORM = 13,
    R16G16B16A16_SINT = 14,
    R32G32_TYPELESS = 15,
    R32G32_FLOAT = 16,
    R32G32_UINT = 17,
    R32G32_SINT = 18,
    R32G8X24_TYPELESS = 19,
    D32_FLOAT_S8X24_UINT = 20,
    R32_FLOAT_X8X24_TYPELESS = 21,
    X32_TYPELESS_G8X24_UINT = 22,
    R10G10B10A2_TYPELESS = 23,
    R10G10B10A2_UNORM = 24,
    R10G10B10A2_UINT = 25,
    R11G11B10_FLOAT = 26,
    R8G8B8A8_TYPELESS = 27,
    R8G8B8A8_UNORM = 28,
    R8G8B8A8_UNORM_SRGB = 29,
    R8G8B8A8_UINT = 30,
    R8G8B8A8_SNORM = 31,
    R8G8B8A8_SINT = 32,
    R16G16_TYPELESS = 33,
    R16G16_FLOAT = 34,
    R16G16_UNORM = 35,
    R16G16_UINT = 36,
    R16G16_SNORM = 37,
    R16G16_SINT = 38,
    R32_TYPELESS = 39,
    D32_FLOAT = 40,
    R32_FLOAT = 41,
    R32_UINT = 42,
    R32_SINT = 43,
    R24G8_TYPELESS = 44,
    D24_UNORM_S8_UINT = 45,
    R24_UNORM_X8_TYPELESS = 46,
    X24_TYPELESS_G8_UINT = 47,
    R8G8_TYPELESS = 48,
    R8G8_UNORM = 49,
    R8G8_UINT = 50,
    R8G8_SNORM = 51,
    R8G8_SINT = 52,
    R16_TYPELESS = 53,
    R16_FLOAT = 54,
    D16_UNORM = 55,
    R16_UNORM = 56,
    R16_UINT = 57,
    R16_SNORM = 58,
    R16_SINT = 59,
    R8_TYPELESS = 60,
    R8_UNORM = 61,
    R8_UINT = 62,
    R8_SNORM = 63,
    R8_SINT = 64,
    A8_UNORM = 65,
    R1_UNORM = 66,
    R9G9B9E5_SHAREDEXP = 67,
    R8G8_B8G8_UNORM = 68,
    G8R8_G8B8_UNORM = 69,
    BC1_TYPELESS = 70,
    BC1_UNORM = 71,
    BC1_UNORM_SRGB = 72,
    BC2_TYPELESS = 73,
    BC2_UNORM = 74,
    BC2_UNORM_SRGB = 75,
    BC3_TYPELESS = 76,
    BC3_UNORM = 77,
    BC3_UNORM_SRGB = 78,
    BC4_TYPELESS = 79,
    BC4_UNORM = 80,
    BC4_SNORM = 81,
    BC5_TYPELESS = 82,
    BC5_UNORM = 83,
    BC5_SNORM = 84,
    B5G6R5_UNORM = 85,
    B5G5R5A1_UNORM = 86,
    B8G8R8A8_UNORM = 87,
    B8G8R8X8_UNORM = 88,
    R10G10B10_XR_BIAS_A2_UNORM = 89,
    B8G8R8A8_TYPELESS = 90,
    B8G8R8A8_UNORM_SRGB = 91,
    B8G8R8X8_TYPELESS = 92,
    B8G8R8X8_UNORM_SRGB = 93,
    BC6H_TYPELESS = 94,
    BC6H_UF16 = 95,
    BC6H_SF16 = 96,
    BC7_TYPELESS = 97,
    BC7_UNORM = 98,
    BC7_UNORM_SRGB = 99,
    AYUV = 100,
    Y410 = 101,
    Y416 = 102,
    NV12 = 103,
    P010 = 104,
    P016 = 105,
    _420_OPAQUE = 106,
    YUY2 = 107,
    Y210 = 108,
    Y216 = 109,
    NV11 = 110,
    AI44 = 111,
    IA44 = 112,
    P8 = 113,
    A8P8 = 114,
    B4G4R4A4_UNORM = 115,
    P208 = 130,
    V208 = 131,
    V408 = 132,
    SAMPLER_FEEDBACK_MIN_MIP_OPAQUE,
    SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE,
    FORCE_UINT = 0xffffffff
};

enum class ResourceType : uint32_t
{
    UNKNOWN = 0,
    BUFFER = 1,
    TEXTURE1D = 2,
    TEXTURE2D = 3,
    TEXTURE3D = 4
};

enum class MiscFlagBitsDX10 : uint32_t
{
    GENERATE_MIPS = 0,
    SHARED = 1,
    TEXTURECUBE = 2,
    DRAWINDIRECT_ARGS = 3,
    BUFFER_ALLOW_RAW_VIEWS = 4,
    BUFFER_STRUCTURED = 5,
    RESOURCE_CLAMP = 6,
    SHARED_KEYEDMUTEX = 7,
    GDI_COMPATIBLE = 8,
    SHARED_NTHANDLE = 9,
    RESTRICTED_CONTENT = 10,
    RESTRICT_SHARED_RESOURCE = 11,
    RESTRICT_SHARED_RESOURCE_DRIVER = 12,
    GUARDED = 13,
    TILE_POOL = 14,
    TILED = 15,
    HW_PROTECTED = 16,
    // ???
    SHARED_DISPLAYABLE,
    SHARED_EXCLUSIVE_WRITER,
    END
};

using MiscFlagsDX10 = Flag<MiscFlagBitsDX10>;

struct PixFormat
{
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwFourCC;
    uint32_t dwRGBBitCount;
    uint32_t dwRBitMask;
    uint32_t dwGBitMask;
    uint32_t dwBBitMask;
    uint32_t dwABitMask;
};

struct HeaderBase
{
    uint32_t        fourcc;
    uint32_t        dwSize;
    uint32_t        dwFlags;
    uint32_t        dwHeight;
    uint32_t        dwWidth;
    uint32_t        dwPitchOrLinearSize;
    uint32_t        dwDepth;
    uint32_t        dwMipMapCount;
    uint32_t        dwReserved1[11];
    PixFormat       ddspf;
    uint32_t        dwCaps;
    uint32_t        dwCaps2;
    uint32_t        dwCaps3;
    uint32_t        dwCaps4;
    uint32_t        dwReserved2;
};

struct HeaderExtended
{
    FormatDXGI      dxgiFormat;
    ResourceType    resourceDimension;
    uint32_t        miscFlag;
    uint32_t        arraySize;
    uint32_t        miscFlags2;
};

// Sanity Checks
// https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-file-layout-for-textures
static_assert(sizeof(HeaderBase) == 128, "DDS Header size is wrong!");
static_assert(sizeof(HeaderExtended) == 20, "DDS Header size is wrong!");

static constexpr uint32_t DDS_FOURCC = BitFunctions::GenerateFourCC('D', 'D', 'S', ' ');
static constexpr uint32_t DX10_FOURCC = BitFunctions::GenerateFourCC('D', 'X', '1', '0');
static constexpr uint32_t NON_NATIVE_DDS_FOURCC = BitFunctions::GenerateFourCC('_', 'S', 'D', 'D');

size_t MipSizeToLinearByteSize(MRayPixelTypeRT pf, const Vector2ui& mipSize)
{
    return 512;
}

Expected<ImageHeader<2>> ReadHeaderInternal(bool& isGoodDDS, std::ifstream& ddsFile,
                                            size_t fileSize, const std::string& filePath)
{
    isGoodDDS = false;
    // Now we can load the header
    HeaderBase header;
    char* headerPtr = reinterpret_cast<char*>(&header);
    ddsFile.read(headerPtr, sizeof(header));

    if(header.fourcc != DDS_FOURCC)
        return MRayError("File \"{}\" is not a DDS file!", filePath);
    // Try these and inform the user
    if(header.fourcc == NON_NATIVE_DDS_FOURCC)
        return MRayError("WRONG FOURCC: File \"{}\" has filpped fourcc code. "
                         "It may be saved in different endian platform maybe?",
                         filePath);
    if(header.ddspf.dwFourCC != DX10_FOURCC)
        return MRayError("File \"{}\" is not a DX10-DDS "
                         "file, old DDS files are not supported!", filePath);


    ImageHeader<2> outputHeader = {};

    HeaderExtended headerDX10;
    if(fileSize < sizeof(HeaderExtended) + sizeof(HeaderBase))
        return MRayError("File \"{}\" is too small to "
                            "be a DX10-DDS file!", filePath);

    char* headerPtrDX10 = reinterpret_cast<char*>(&headerDX10);
    ddsFile.read(headerPtrDX10, sizeof(HeaderExtended));

    MiscFlagsDX10 miscFlags = static_cast<MiscFlagBitsDX10>(headerDX10.miscFlag);

    if(headerDX10.arraySize != 1)
    {
        return MRayError("File \"{}\" has array textures, it is not "
                            "supported yet!", filePath);
    }

    if(headerDX10.resourceDimension != ResourceType::TEXTURE2D)
    {
        return MRayError("File \"{}\" has non-2D texture, it is not "
                            "supported yet!", filePath);
    }

    if(miscFlags[MiscFlagsDX10::F::TEXTURECUBE])
    {
        return MRayError("File \"{}\" has cube texture, it is not "
                            "supported yet!", filePath);
    }

    // All fine go!
    outputHeader.mipCount = std::max(header.dwMipMapCount, 1u);
    outputHeader.dimensions = Vector2ui(header.dwWidth,
                                        header.dwHeight);
    using enum FormatDXGI;
    using enum MRayPixelEnum;
    // Check Format
    switch(headerDX10.dxgiFormat)
    {
        case BC1_TYPELESS:
        case BC1_UNORM:
        case BC1_UNORM_SRGB:
        {
            outputHeader.pixelType = MRayPixelType<MR_BC1_UNORM>{};
            break;
        }
        case BC2_TYPELESS:
        case BC2_UNORM:
        case BC2_UNORM_SRGB:
        {
            outputHeader.pixelType = MRayPixelType<MR_BC2_UNORM>{};
            break;
        }
        case BC3_TYPELESS:
        case BC3_UNORM:
        case BC3_UNORM_SRGB:
        {
            outputHeader.pixelType = MRayPixelType<MR_BC3_UNORM>{};
            break;
        }
        case BC4_TYPELESS:
        case BC4_UNORM:
        {
            outputHeader.pixelType = MRayPixelType<MR_BC4_UNORM>{};
            break;
        }
        case BC4_SNORM:
        {
            outputHeader.pixelType = MRayPixelType<MR_BC4_SNORM>{};
            break;
        }
        case BC5_TYPELESS:
        case BC5_UNORM:
        {
            outputHeader.pixelType = MRayPixelType<MR_BC5_UNORM>{};
            break;
        }
        case BC5_SNORM:
        {
            outputHeader.pixelType = MRayPixelType<MR_BC5_SNORM>{};
            break;
        }
        case BC6H_TYPELESS:
        case BC6H_UF16:
        {
            outputHeader.pixelType = MRayPixelType<MR_BC6H_UFLOAT>{};
            break;
        }
        case BC6H_SF16:
        {
            outputHeader.pixelType = MRayPixelType<MR_BC6H_SFLOAT>{};
            break;
        }
        case BC7_TYPELESS:
        case BC7_UNORM:
        case BC7_UNORM_SRGB:
        {
            outputHeader.pixelType = MRayPixelType<MR_BC7_UNORM>{};
            break;
        }
        default:
        {
            // Probably good DDS texture
            // let OIIO handle it.
            isGoodDDS = true;
            return ImageHeader<2>{};
        }
    }
    // Check Color Space
    switch(headerDX10.dxgiFormat)
    {
        case BC1_UNORM_SRGB:
        case BC2_UNORM_SRGB:
        case BC3_UNORM_SRGB:
        case BC7_UNORM_SRGB:
        {
            outputHeader.colorSpace = MRayColorSpaceEnum::MR_DEFAULT;
            outputHeader.gamma = Float(2.2);
            break;
        }
        default:
        {
            outputHeader.colorSpace = MRayColorSpaceEnum::MR_DEFAULT;
            outputHeader.gamma = Float(1);
            break;
        }
    }
    return outputHeader;
}

}

Expected<ImageHeader<2>> ReadHeaderDDS(bool& isGoodDDS,
                                       const std::string& filePath)
{
    using namespace DDSLoader;
    size_t fileSize = std::filesystem::file_size(std::filesystem::path(filePath));
    if(fileSize < sizeof(HeaderBase))
        return MRayError("File \"{}\" is too small to "
                         "be a DDS file!", filePath);

    std::ifstream ddsFile = std::ifstream(filePath, std::ios::binary);
    if(!ddsFile.is_open())
        return MRayError("File \"{}\" is not found",
                         filePath);

    return ReadHeaderInternal(isGoodDDS, ddsFile, fileSize, filePath);
}

Expected<Image<2>> ReadImageDDS(const std::string& filePath)
{
    using namespace DDSLoader;
    size_t fileSize = std::filesystem::file_size(std::filesystem::path(filePath));
    if(fileSize < sizeof(HeaderBase))
        return MRayError("File \"{}\" is too small to "
                         "be a DDS file!", filePath);

    std::ifstream ddsFile = std::ifstream(filePath, std::ios::binary);
    if(!ddsFile.is_open())
        return MRayError("File \"{}\" is not found",
                         filePath);

    Image<2> result = {};
    bool isGoodDDS;
    auto headerE = ReadHeaderInternal(isGoodDDS, ddsFile, fileSize, filePath);
    if(headerE.has_error())
        return headerE.error();

    result.header = headerE.value();
    for(uint32_t i = 0; i < result.header.mipCount; i++)
    {
        Vector2ui mipSize = Vector2ui(result.header.dimensions[0] >> i,
                                      result.header.dimensions[1] >> i);
        mipSize = Vector2ui::Max(mipSize, Vector2ui(1));
        size_t mipByteSize = MipSizeToLinearByteSize(result.header.pixelType, mipSize);

        TransientData data(std::in_place_type_t<Byte>{}, mipByteSize);
        auto dataSpan = data.AccessAs<Byte>();
        ddsFile.read(reinterpret_cast<char*>(dataSpan.data()),
                     dataSpan.size_bytes());
        if(!ddsFile)
            return MRayError("File \"{}\" does not have enough data to "
                             "fulfill its header!", filePath);
        result.imgData.emplace_back(mipSize, std::move(data));

    }
    return result;
}