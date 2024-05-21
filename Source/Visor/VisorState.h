#pragma once

#include <string>
#include <vector>
#include "Core/Types.h"
#include "Core/Vector.h"

//
#include "Common/AnalyticStructs.h"

struct VisorAnalyticData
{
    double  frameTime;
    double  usedGPUMemoryMiB;
};

struct VisorState
{
    CameraTransform         transform;
    SceneAnalyticData       scene;
    TracerAnalyticData      tracer;
    VisorAnalyticData       visor;
    RendererAnalyticData    renderer;

    // Internal state
    int32_t                currentCameraIndex = 0;
};