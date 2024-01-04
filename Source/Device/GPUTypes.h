#pragma once

#include "Core/Vector.h"

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

// Generic Call Parameters
struct KernelCallParams
{
    uint32_t gridSize;
    uint32_t blockSize;
    uint32_t blockId;
    uint32_t threadId;

    MRAY_HYBRID uint32_t GlobalId() const;
    MRAY_HYBRID uint32_t TotalSize() const;
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
template <uint32_t D, class = void> struct TextureSizeT;
template <uint32_t D> requires(D == 1)
struct TextureSizeT<D> { using type = uint32_t; };
template <uint32_t D> requires(D > 1 && D < 4)
struct TextureSizeT<D> { using type = Vector<D, uint32_t>; };

template <uint32_t D>
using TextureDim = typename TextureSizeT<D>::type;

// Texture initialization parameters
// Defaults are for x -> normalized float conversion
template <uint32_t D, class UnderlyingT>
struct TextureInitParams
{
    using UnderlyingType    = UnderlyingT;

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
    TextureDim<D>           size            = TextureDim<D>(0);
    uint32_t                mipCount        = 0;
};

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t KernelCallParams::GlobalId() const
{
    return blockId * blockSize + threadId;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t KernelCallParams::TotalSize() const
{
    return gridSize * blockSize;
}