#pragma once

#include "Core/Vector.h"

#include <type_traits>

using GPUThreadInitFunction = void(*)();

static constexpr uint32_t WarpSize();
static constexpr uint32_t StaticThreadPerBlock1D();
static constexpr uint32_t TotalQueuePerDevice();

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
MRAY_HYBRID MRAY_CGPU_INLINE
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
MRAY_HYBRID MRAY_CGPU_INLINE
UVType<D> LinearToUV(const TextureExtent<D>& extents,
                     uint32_t linearIndex)
{
    UVType<D> indicesFloat = LinearToFloatIndex<D>(extents, linearIndex);
    UVType<D> extentsFloat = UVType<D>(extents);
    return indicesFloat / extentsFloat;
}
