#pragma once

#include <string>
#include <vector>
#include "Core/Types.h"
#include "Core/Vector.h"

struct VisorAnalyticData
{
    double  frameTime;
    double  usedGPUMemoryMiB;
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
    const Vector2i  renderResolution;
};

struct TracerAnalyticData
{
    using TypeCountPair = Pair<std::string, uint32_t>;

    std::vector<TypeCountPair> camTypes;
    std::vector<TypeCountPair> lightTypes;
    std::vector<TypeCountPair> primTypes;
    std::vector<TypeCountPair> mediumTypes;
    std::vector<TypeCountPair> materialTypeList;

    uint32_t    textureCountList;
    uint32_t    surfaceCount;

    // Memory Related
    double          totalGPUMemoryMiB;
    double          usedGPUMemoryMiB;
};

struct SceneAnalyticData
{
    // Generic
    std::string sceneName;
    // Timings
    double      sceneLoadTime;      // secs
};

struct VisorState
{
    SceneAnalyticData       scene;
    TracerAnalyticData      tracer;
    VisorAnalyticData       visor;
    RendererAnalyticData    renderer;
};