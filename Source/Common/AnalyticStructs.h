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
    //using TypeCountPair = Tuple<std::string, uint32_t>;

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

