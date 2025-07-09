#pragma once

#include "Core/Types.h"
#include "Core/Vector.h"
#include "Core/AABB.h"

#include <vector>
#include <string>

// TODO: Move this
struct CameraTransform
{
    Vector3 position    = Vector3::Zero();
    Vector3 gazePoint   = Vector3(0, 0, -1);
    Vector3 up          = Vector3::YAxis();
};

struct TracerAnalyticData
{
    using TypeCountPair = Pair<std::string, uint32_t>;

    std::vector<std::string> camTypes;
    std::vector<std::string> lightTypes;
    std::vector<std::string> primTypes;
    std::vector<std::string> mediumTypes;
    std::vector<std::string> materialTypes;
    std::vector<std::string> rendererTypes;

    MRayColorSpaceEnum  tracerColorSpace;
    size_t              totalGPUMemoryBytes;
};

struct RendererAnalyticData
{
    // Performance
    double              throughput;
    std::string         throughputSuffix;
    //
    double              workPerPixel;
    double              wppLimit;
    std::string         workPerPixelSuffix;
    // Timings
    float               iterationTimeMS;
    // Image related
    Vector2ui           renderResolution;
    MRayColorSpaceEnum  outputColorSpace;
    // Memory
    size_t              usedGPUMemoryBytes;
    // Custom Input Related
    uint32_t            customLogicSize0 = 0;
    uint32_t            customLogicSize1 = 0;
};

struct SceneAnalyticData
{
    // Generic
    std::string sceneName       = "";
    // Timings
    double      sceneLoadTimeS  = 0.0;
    // Amounts
    uint32_t    mediumCount     = 0;
    uint32_t    primCount       = 0;
    uint32_t    textureCount    = 0;
    uint32_t    surfaceCount    = 0;
    uint32_t    lightCount      = 0;
    uint32_t    cameraCount     = 0;
    //
    AABB3       sceneExtent     = AABB3::Negative();
};

// TODO: Move this somewhere proper later
#include "Core/MemAlloc.h"
inline Pair<double, std::string_view> ConvertMemSizeToString(size_t size)
{
    // This function is overengineered for a GUI operation.
    // This probably has better precision? (probably not)
    // has high amount memory (TiB++ of memory).
    Pair<double, std::string_view> result;
    using namespace std::string_view_literals;
    size_t shiftVal = 0;
    if(size >= 1_TiB)
    {
        result.second = "TiB"sv;
        shiftVal = 40;
    }
    else if(size >= 1_GiB)
    {
        result.second = "GiB"sv;
        shiftVal = 30;
    }
    else if(size >= 1_MiB)
    {
        result.second = "MiB"sv;
        shiftVal = 20;
    }
    else if(size >= 1_KiB)
    {
        result.second = "KiB"sv;
        shiftVal = 10;
    }
    else
    {
        result.second = "Bytes"sv;
        shiftVal = 0;
    }

    size_t mask = ((size_t(1) << shiftVal) - 1);
    size_t integer = size >> shiftVal;
    size_t decimal = mask & size;
    // Sanity check
    static_assert(std::numeric_limits<double>::is_iec559,
                  "This overengineered function requires "
                  "IEEE-754 floats.");
    static constexpr size_t DOUBLE_MANTISSA = 52;
    static constexpr size_t MANTISSA_MASK = (size_t(1) << DOUBLE_MANTISSA) - 1;
    size_t bitCount = Bit::RequiredBitsToRepresent(decimal);
    if(bitCount > DOUBLE_MANTISSA)
        decimal >>= (bitCount - DOUBLE_MANTISSA);
    else
        decimal <<= (DOUBLE_MANTISSA - bitCount);


    uint64_t dblFrac = std::bit_cast<uint64_t>(1.0);
    dblFrac |= decimal & MANTISSA_MASK;
    result.first = std::bit_cast<double>(dblFrac);
    result.first += static_cast<double>(integer) - 1.0;
    return result;
}
