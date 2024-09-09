#pragma once

#include <string>
#include <vector>
#include "Core/Types.h"
#include "Core/Vector.h"

//
#include "Common/AnalyticStructs.h"

#include <vulkan/vulkan.h>

enum class TracerRunState
{
    RUNNING,
    STOPPED,
    PAUSED,

    END
};

struct SwapchainInfo
{
    VkPresentModeKHR    presentMode = VK_PRESENT_MODE_FIFO_KHR;
    VkColorSpaceKHR     colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkFormat            format = VK_FORMAT_UNDEFINED;
    VkExtent2D          extent = VkExtent2D{0, 0};
};

struct VisorAnalyticData
{
    float               frameTime;
    size_t              usedGPUMemory;
    SwapchainInfo       swapchainInfo;
};

struct VisorState
{
    CameraTransform         transform;
    SceneAnalyticData       scene;
    TracerAnalyticData      tracer;
    VisorAnalyticData       visor;
    RendererAnalyticData    renderer;

    // Throughput
    using ThroughputAverage = Math::MovingAverage<16>;
    ThroughputAverage       renderThroughputAverage;

    // This comes after scene change
    uint64_t                tracerUsedGPUMemBytes;

    // Internal state
    TracerRunState          currentRendererState = TracerRunState::STOPPED;
    int32_t                 currentCameraIndex = 0;
    int32_t                 currentRenderIndex = 0;
    int32_t                 currentRenderLogic0 = 0;
    int32_t                 currentRenderLogic1 = 0;
};