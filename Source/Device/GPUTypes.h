#pragma once

#include "Core/Vector.h"
#include "Core/MemAlloc.h"
#include "TransientPool/TransientPool.h"

#include <type_traits>

static constexpr uint32_t WarpSize();
static constexpr uint32_t StaticThreadPerBlock1D();
static constexpr uint32_t TotalQueuePerDevice();

static constexpr uint32_t QueuePerDevice = 4;

static_assert(QueuePerDevice > 0, "At least one queue must be present on a Device!");

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

struct KernelIssueParams
{
    uint32_t workCount;
    uint32_t sharedMemSize = 0;
};

struct KernelExactIssueParams
{
    uint32_t gridSize;
    uint32_t blockSize;
    uint32_t sharedMemSize = 0;
};

// Texture Related
enum class InterpolationType
{
    NEAREST,
    LINEAR
};

enum class EdgeResolveType
{
    WRAP,
    CLAMP,
    MIRROR
    // Border does not work properly
};

// Texture Size Type Metaprogramming
template <uint32_t D, class = void> struct TextureExtentT;
template <uint32_t D> requires(D == 1)
struct TextureExtentT<D> { using type = uint32_t; };
template <uint32_t D> requires(D > 1 && D < 4)
struct TextureExtentT<D> { using type = Vector<D, uint32_t>; };
template <uint32_t D>
using TextureExtent = typename TextureExtentT<D>::type;

template <uint32_t C, class T, class = void> struct PaddedChannelT;
template <uint32_t C, class T> requires(C != 3)
struct PaddedChannelT<C, T> { using type = T; };
template <uint32_t C, class T> requires(C == 3)
struct PaddedChannelT<C, T> { using type = Vector<C + 1, typename T::InnerType>; };
template <uint32_t C, class T>
using PaddedChannel = typename PaddedChannelT<C, T>::type;

template <uint32_t D, class = void> struct UVTypeT;
template <uint32_t D> requires(D == 1)
struct UVTypeT<D> { using type = Float; };
template <uint32_t D> requires(D > 1 && D < 4)
struct UVTypeT<D> { using type = Vector<D, Float>; };
template <uint32_t D>
using UVType = typename UVTypeT<D>::type;

// Texture initialization parameters
// Defaults are for x -> normalized float conversion
template <uint32_t D>
struct TextureInitParams
{
    bool                    normIntegers    = true;
    bool                    normCoordinates = true;
    bool                    convertSRGB     = false;

    InterpolationType       interp      = InterpolationType::NEAREST;
    EdgeResolveType         eResolve    = EdgeResolveType::WRAP;

    uint32_t                maxAnisotropy   = 16;
    Float                   mipmapBias      = 0.0f;
    Float                   minMipmapClamp  = -100.0f;
    Float                   maxMipmapClamp  = 100.0f;

    // Dimension Related (must be set)
    TextureExtent<D>        size            = TextureExtent<D>(0);
    uint32_t                mipCount        = 0;
};

template<uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
UVType<D> LinearToFloatIndex(const TextureExtent<D>& extents,
                             uint32_t linearIndex)
{
    if constexpr(D == 1)
        return linearIndex;
    else if constexpr(D == 2)
        return UVType<D>(linearIndex % extents[0] + Float{0.5},
                         linearIndex / extents[0] + Float{0.5});
    else if constexpr(D == 3)
        return UVType<D>(linearIndex % extents[0] + Float{0.5},
                         linearIndex / extents[0] + Float{0.5},
                         linearIndex / extents[0] * extents[1] + Float{0.5});
    else static_assert(D <= 3, "Only up to 3D textures are supported!");
    return UVType<D>(std::numeric_limits<Float>::max());
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