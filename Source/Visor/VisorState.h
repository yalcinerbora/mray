#pragma once

#include <string>
#include <vector>
#include "Core/Types.h"
#include "Core/Vector.h"

//
#include "Common/AnalyticStructs.h"

enum class TracerRunState
{
    RUNNING,
    STOPPED,
    PAUSED,

    END
};

struct VisorAnalyticData
{
    float   frameTime;
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
    TracerRunState          currentRendererState = TracerRunState::STOPPED;
    int32_t                 currentCameraIndex = 0;
    int32_t                 currentRenderIndex = 0;
    int32_t                 currentRenderLogic0 = 0;
    int32_t                 currentRenderLogic1 = 0;
};