#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include "Core/Types.h"
#include "VulkanTypes.h"

struct MRayError;
class Swapchain;
struct VulkanSystemView;

struct FramebufferPack
{
    VkExtent2D      extent;
    VkImage         img;
    VkImageView     imgView;
    VkFramebuffer   fbo;
    VkRenderPass    renderPass;
};

struct FrameSemaphorePair
{
    VulkanBinarySemaphore imageAvailableSignal;
    VulkanBinarySemaphore commandsExecutedSignal;
};

struct FramePack : public FramebufferPack
{
    VkCommandBuffer commandBuffer = nullptr;
    float           prevFrameTime = 0.0f;
};

class FramePool
{
    private:
    //VkCommandPool       commandPool = nullptr;
    VkDevice            deviceVk    = nullptr;
    VkQueue             mainQueueVk = nullptr;
    VulkanFence         fence;
    VulkanCommandBuffer cBuffer;
    FrameSemaphorePair  semaphores;

    public:
                    FramePool() = default;
                    FramePool(const FramePool&) = delete;
                    FramePool(FramePool&&) = default;
    FramePool&      operator=(const FramePool&) = delete;
    FramePool&      operator=(FramePool&&) = default;
                    ~FramePool() = default;

    MRayError       Initialize(const VulkanSystemView& handlesVk);

    FramePack       AcquireNextFrame(Swapchain& swapchain);
    void            PresentThisFrame(Swapchain& swapchain,
                                     const VulkanTimelineSemaphore* extraWaitSem);
};