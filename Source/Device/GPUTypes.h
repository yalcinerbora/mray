#pragma once

#include "Core/Vector.h"
#include "Core/NormConvFunctions.h"

#include <type_traits>

using GPUThreadInitFunction = void(*)();

template <class T> struct IntegralOf
{
    static_assert(!std::is_same_v<T, T>,
                    "Type does not have a proper integral wrapper!");
};

template <class T>
requires (sizeof(T) == 8 && alignof(T) >= alignof(uint64_t))
struct IntegralOf<T> { using type = uint64_t; };

template <class T>
requires (sizeof(T) == 4 && alignof(T) >= alignof(uint32_t))
struct IntegralOf<T> { using type = uint32_t; };

template <class T>
requires (sizeof(T) == 2 && alignof(T) >= alignof(uint16_t))
struct IntegralOf<T> { using type = uint16_t; };

static constexpr uint32_t ComputeQueuePerDevice = 4;
static_assert(ComputeQueuePerDevice > 0,
              "At least one compute queue must "
              "be present on a Device!");

enum class DeviceQueueType
{
    NORMAL,
    FIRE_AND_FORGET,
    TAIL_LAUNCH
};

struct KernelAttributes
{
    size_t  localMemoryPerThread;
    size_t  constantMemorySize;
    int     maxDynamicSharedMemorySize;
    int     maxTBP;
    int     registerCountPerThread;
    size_t  staticSharedMemorySize;
};

struct DeviceWorkIssueParams
{
    uint32_t workCount;
    uint32_t sharedMemSize = 0;
};

struct DeviceBlockIssueParams
{
    uint32_t gridSize;
    uint32_t blockSize;
    uint32_t sharedMemSize = 0;
};

// Texture Size Type Metaprogramming
template <uint32_t D, std::integral T, class = void> struct TextureExtentT;
template <uint32_t D, std::integral T> requires(D == 1)
struct TextureExtentT<D, T> { using type = T; };
template <uint32_t D, std::integral T> requires(D > 1 && D < 4)
struct TextureExtentT<D, T> { using type = Vector<D, T>; };
template <uint32_t D>
using TextureExtent = typename TextureExtentT<D, uint32_t>::type;
template <uint32_t D>
using TextureSignedExtent = typename TextureExtentT<D, int32_t>::type;

// Padded channel type metaprogramming
template <uint32_t C, class T, class = void> struct PaddedChannelT;
template <uint32_t C, class T> requires(C != 3u)
struct PaddedChannelT<C, T> { using type = T; };
template <uint32_t C, class T> requires(C == 3u)
struct PaddedChannelT<C, T> { using type = Vector<C + 1, typename T::InnerType>; };
template <uint32_t C, class T>
using PaddedChannel = typename PaddedChannelT<C, T>::type;

// UV type metaprogramming
template <uint32_t D, class = void> struct UVTypeT;
template <uint32_t D> requires(D == 1)
struct UVTypeT<D> { using type = Float; };
template <uint32_t D> requires(D > 1 && D < 4)
struct UVTypeT<D> { using type = Vector<D, Float>; };
template <uint32_t D>
using UVType = typename UVTypeT<D>::type;

// Find channel count
template<class T>
requires (std::integral<T> || std::floating_point<T> || VectorC<T>)
constexpr uint32_t VectorTypeToChannels()
{
    if constexpr(std::is_integral_v<T>||
                 std::is_floating_point_v<T>)
    {
        return 1u;
    }
    else
    {
        return T::Dims;
    }
}

template <class T>
constexpr uint32_t BCTypeToChannels()
{
    // https://developer.nvidia.com/blog/revealing-new-features-in-the-cuda-11-5-toolkit/
    if constexpr(std::is_same_v<T, PixelBC1> ||
                 std::is_same_v<T, PixelBC2> ||
                 std::is_same_v<T, PixelBC3> ||
                 std::is_same_v<T, PixelBC7>)
    {
        return 4;
    }
    else if constexpr(std::is_same_v<T, PixelBC4U> ||
                      std::is_same_v<T, PixelBC4S>)
    {
        return 1;
    }
    else if constexpr(std::is_same_v<T, PixelBC5U> ||
                      std::is_same_v<T, PixelBC5S>)
    {
        return 2;
    }
    else if constexpr(std::is_same_v<T, PixelBC6U> ||
                      std::is_same_v<T, PixelBC6S>)
    {
        return 3;
    }
    else static_assert(std::is_same_v<T, PixelBC1>,
                       "Unknown Block Compressed Format!");
}

template <class T>
constexpr uint32_t PixelTypeToChannels()
{
    if constexpr(IsBlockCompressedPixel<T>)
        return BCTypeToChannels<T>();
    else
        return VectorTypeToChannels<T>();
}

template <class T>
constexpr bool IsNormConvertible()
{
    // 32-bit types are not norm convertible,
    // so removed these from this function
    //
    // YOLO
    return (std::is_same_v<T, uint16_t>     ||
            std::is_same_v<T, Vector2us>    ||
            std::is_same_v<T, Vector3us>    ||
            std::is_same_v<T, Vector4us>    ||

            std::is_same_v<T, int16_t>      ||
            std::is_same_v<T, Vector2s>     ||
            std::is_same_v<T, Vector3s>     ||
            std::is_same_v<T, Vector4s>     ||

            std::is_same_v<T, uint8_t>      ||
            std::is_same_v<T, Vector2uc>    ||
            std::is_same_v<T, Vector3uc>    ||
            std::is_same_v<T, Vector4uc>    ||

            std::is_same_v<T, int8_t>       ||
            std::is_same_v<T, Vector2c>     ||
            std::is_same_v<T, Vector3c>     ||
            std::is_same_v<T, Vector4c>);
}

template <class T>
constexpr uint32_t BCTypeToBlockSize()
{
    // https://developer.nvidia.com/blog/revealing-new-features-in-the-cuda-11-5-toolkit/
    if constexpr(std::is_same_v<T, PixelBC1>  ||
                 std::is_same_v<T, PixelBC4U> ||
                 std::is_same_v<T, PixelBC4S>)
    {
        return 8;
    }
    else if constexpr(std::is_same_v<T, PixelBC2>  ||
                      std::is_same_v<T, PixelBC3>  ||
                      std::is_same_v<T, PixelBC5U> ||
                      std::is_same_v<T, PixelBC5S> ||
                      std::is_same_v<T, PixelBC6U> ||
                      std::is_same_v<T, PixelBC6S> ||
                      std::is_same_v<T, PixelBC7>)
    {
        return 16;
    }
    else static_assert(std::is_same_v<T, PixelBC1>,
                       "Unknown Block Compressed Format!");
}

// Texture initialization parameters
// Defaults are for x -> normalized float conversion
template <uint32_t D>
struct TextureInitParams
{
    uint32_t    maxAnisotropy   = 16;
    Float       mipmapBias      = 0.0f;
    Float       minMipmapClamp  = -100.0f;
    Float       maxMipmapClamp  = 100.0f;
    // Dimension Related (must be set)
    uint32_t            mipCount        = 1;
    TextureExtent<D>    size            = TextureExtent<D>(0);
    bool                normIntegers    = true;
    bool                normCoordinates = true;
    bool                convertSRGB     = false;
    //
    MRayTextureInterpEnum       interp      = MRayTextureInterpEnum::MR_NEAREST;
    MRayTextureEdgeResolveEnum  eResolve    = MRayTextureEdgeResolveEnum::MR_WRAP;
};

template<uint32_t D>
MR_HF_DECL constexpr
UVType<D> LinearToTexelCoordinates(const TextureExtent<D>& extents,
                                   uint32_t linearIndex)
{
    if constexpr(D == 1)
        return UVType<D>(Float(linearIndex) + Float(0.5));
    else if constexpr(D == 2)
        return UVType<D>(Float(linearIndex % extents[0]) + Float(0.5),
                         Float(linearIndex / extents[0]) + Float(0.5));
    else if constexpr(D == 3)
        return UVType<D>(Float(linearIndex % extents[0]) + Float(0.5),
                         Float(linearIndex / extents[0]) + Float(0.5),
                         Float(linearIndex / (extents[0] * extents[1])) + Float(0.5));
    else static_assert(D <= 3, "Only up to 3D textures are supported!");
}

template<uint32_t D>
MR_HF_DECL constexpr
UVType<D> LinearToUV(const TextureExtent<D>& extents,
                     uint32_t linearIndex)
{
    UVType<D> indicesFloat = LinearToFloatIndex<D>(extents, linearIndex);
    UVType<D> extentsFloat = UVType<D>(extents);
    return indicesFloat / extentsFloat;
}

// Given properly aligned (256bytes) byte array, fetch data from
// it with the given runtime pixel type "pixType".
// This is used for volumetric data currently. It is put here since
// we may abuse GPU intrinsics to reduce the complexity.
MR_GF_DEF
constexpr Vector4
ReadGenericTexelData(const Byte* data, uint32_t index,
                     MRayPixelEnum pixType)
{
    using namespace NormConversion;
    const Byte* aData = std::assume_aligned<256>(data);
    //
    #define RIP(TYPE) reinterpret_cast<const TYPE*>(aData)[index]

    Vector4 r;
    switch(pixType)
    {
        using enum MRayPixelEnum;
        // 1-Channel
        case MR_R8_UNORM:  r[0] = FromUNorm<Float>(RIP(uint8_t));  break;
        case MR_R16_UNORM: r[0] = FromUNorm<Float>(RIP(uint16_t)); break;
        case MR_R8_SNORM:  r[0] = FromSNorm<Float>(RIP(int8_t));   break;
        case MR_R16_SNORM: r[0] = FromSNorm<Float>(RIP(int16_t));  break;
        //case MR_R_HALF:
        case MR_R_FLOAT:   r[0] = Float(RIP(float));               break;

        // 2-Channel
        case MR_RG8_UNORM:  r[0] = FromUNorm<Float>(RIP(Vector2uc)[0]);
                            r[1] = FromUNorm<Float>(RIP(Vector2uc)[1]); break;
        case MR_RG16_UNORM: r[0] = FromUNorm<Float>(RIP(Vector2us)[0]);
                            r[1] = FromUNorm<Float>(RIP(Vector2us)[1]); break;
        case MR_RG8_SNORM:  r[0] = FromSNorm<Float>(RIP(Vector2c)[0]);
                            r[1] = FromSNorm<Float>(RIP(Vector2c)[1]);  break;
        case MR_RG16_SNORM: r[0] = FromSNorm<Float>(RIP(Vector2s)[0]);
                            r[1] = FromSNorm<Float>(RIP(Vector2s)[1]);  break;
        //case MR_RG_HALF:
        case MR_RG_FLOAT:   r = Vector4(RIP(Vector2f), 0, 0);           break;

        // 3-Channel
        case MR_RGB8_UNORM:  r[0] = FromUNorm<Float>(RIP(Vector3uc)[0]);
                             r[1] = FromUNorm<Float>(RIP(Vector3uc)[1]);
                             r[2] = FromUNorm<Float>(RIP(Vector3uc)[2]); break;
        case MR_RGB16_UNORM: r[0] = FromUNorm<Float>(RIP(Vector3us)[0]);
                             r[1] = FromUNorm<Float>(RIP(Vector3us)[1]);
                             r[2] = FromUNorm<Float>(RIP(Vector3us)[2]); break;
        case MR_RGB8_SNORM:  r[0] = FromSNorm<Float>(RIP(Vector3c)[0]);
                             r[1] = FromSNorm<Float>(RIP(Vector3c)[1]);
                             r[2] = FromSNorm<Float>(RIP(Vector3c)[2]);  break;
        case MR_RGB16_SNORM: r[0] = FromSNorm<Float>(RIP(Vector3s)[0]);
                             r[1] = FromSNorm<Float>(RIP(Vector3s)[1]);
                             r[2] = FromSNorm<Float>(RIP(Vector3s)[2]);  break;
        //case MR_RGB_HALF:
        case MR_RGB_FLOAT:  r = Vector4(RIP(Vector3f), Float(0));        break;

        // 4-Channel
        case MR_RGBA8_UNORM:  r = Vector4(RIP(UNorm4x8));  break;
        case MR_RGBA16_UNORM: r = Vector4(RIP(UNorm4x16)); break;
        case MR_RGBA8_SNORM:  r = Vector4(RIP(SNorm4x8));  break;
        case MR_RGBA16_SNORM: r = Vector4(RIP(SNorm4x16)); break;
        case MR_RGBA_FLOAT:   r = Vector4(RIP(Vector4f));  break;

        // TODO:
        //case MR_RGBA_HALF:
        default: r = Vector4::Zero();
    }
    #undef RIP
    //
    return r;
}

