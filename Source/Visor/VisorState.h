#pragma once

#include <string>
#include <vector>
#include "Core/Types.h"
#include "Core/Vector.h"

//
#include "MRay/TransferQueue.h"

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
};