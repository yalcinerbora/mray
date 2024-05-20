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
    std::vector<TypeCountPair> rendererTypes;

    MRayColorSpaceEnum  tracerColorSpace;

    // Memory Related
    double      totalGPUMemoryMiB;
    double      usedGPUMemoryMiB;
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
    Vector2i            renderResolution;
    MRayColorSpaceEnum  outputColorSpace;
};

struct SceneAnalyticData
{
    // Generic
    std::string     sceneName;
    // Timings
    double          sceneLoadTime;  // secs

    // Amounts
    uint32_t        mediumCount;
    uint32_t        primCount;
    uint32_t        textureCount;
    uint32_t        surfaceCount;
    uint32_t        cameraCount;

    AABB3f          sceneExtent;
    Vector2         timeRange;
};
