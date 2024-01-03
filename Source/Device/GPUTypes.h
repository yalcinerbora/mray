#pragma once

#include "Core/Vector.h"

static constexpr uint32_t WarpSize();
static constexpr uint32_t StaticThreadPerBlock1D();
static constexpr uint32_t TotalQueuePerDevice();

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

// Generic Call Parameters
struct KernelCallParameters1D
{
    uint32_t gridSize;
    uint32_t blockSize;
    uint32_t blockId;
    uint32_t threadId;
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


// Texture initialization parameters
// Defaults are for x -> normalized float conversion
template <uint32_t D, class UnderlyingT>
struct TextureInitParams
{
    using UnderlyingType = UnderlyingT;

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
    Vector<D, uint32_t>     size            = Vector<D, uint32_t>::Zero();
    uint32_t                mipCount        = 0;
};