#pragma once

#include "Core/Types.h"
#include "Core/Vector.h"
#include "Core/AABB.h"

#include <vector>
#include <string>

// TODO: Move this
struct CameraTransform
{
    Vector3 position;
    Vector3 gazePoint;
    Vector3 up;
};

struct TracerAnalyticData
{
    using TypeCountPair = Pair<std::string, uint32_t>;

    std::vector<TypeCountPair> camTypes;
    std::vector<TypeCountPair> lightTypes;
    std::vector<TypeCountPair> primTypes;
    std::vector<TypeCountPair> mediumTypes;
    std::vector<TypeCountPair> materialTypes;

    std::vector<std::string> rendererTypes;

    MRayColorSpaceEnum  tracerColorSpace;

    // Memory Related
    size_t      totalGPUMemoryBytes;
    size_t      usedGPUMemoryBytes;
};

struct RendererAnalyticData
{
    // Performance
    double          throughput;
    std::string     throughputSuffix;
    //
    double          workPerPixel;
    std::string     workPerPixelSuffix;
    // Timings
    float           iterationTimeMS;
    // Image related
    Vector2ui           renderResolution;
    MRayColorSpaceEnum  outputColorSpace;

    // Custom Input Related
    uint32_t            customLogicSize0 = 0;
    uint32_t            customLogicSize1 = 0;
};

struct SceneAnalyticData
{
    // Generic
    std::string     sceneName       = "";
    // Timings
    double          sceneLoadTimeS  = 0.0;
    // Amounts
    uint32_t        mediumCount     = 0;
    uint32_t        primCount       = 0;
    uint32_t        textureCount    = 0;
    uint32_t        surfaceCount    = 0;
    uint32_t        cameraCount     = 0;
    //
    AABB3f          sceneExtent     = AABB3f::Negative();
    Vector2         timeRange       = Vector2::Zero();
};
