#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include "Core/Types.h"

struct SemaphoreVariant;
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
    VkSemaphore imageAvailableSignal    = nullptr;
    VkSemaphore commandsRecordedSignal  = nullptr;
};

struct FramePack : public FramebufferPack
{
    VkCommandBuffer commandBuffer = nullptr;
    float           prevFrameTime = 0.0f;
};

class FramePool
{
    public:
    static constexpr int32_t FRAME_COUNT = 1;
    // TODO: This is little bit memory hog for extra peformance
    // also too long to implement
    // currently we have single "frame-in-fligh" for simplicity
    // CPU thread will wait rendering to issue another batch of commands.
    // This probably will not be a performance bottleneck anyway due to
    // interactive rendering paradigm of the current visor.
    // Visor itself is a secondary system, and it may hog the GPU if
    // there is no iGPU on the system.
    static_assert(FRAME_COUNT == 1, "Currently, single frame in flight is used");

    private:
    template<class T>
    using FrameArray = std::array<T, FRAME_COUNT>;

    VkCommandPool                   commandPool = nullptr;
    VkDevice                        deviceVk    = nullptr;
    VkQueue                         mainQueueVk = nullptr;
    int32_t                         frameIndex  = 0;
    FrameArray<VkFence>             fences      = {};
    FrameArray<VkCommandBuffer>     cBuffers    = {};
    FrameArray<FrameSemaphorePair>  semaphores  = {};

    void                Clear();
    public:
                    FramePool() = default;
                    FramePool(const FramePool&) = delete;
                    FramePool(FramePool&&);
    FramePool&      operator=(const FramePool&) = delete;
    FramePool&      operator=(FramePool&&);
                    ~FramePool();

    MRayError       Initialize(const VulkanSystemView& handlesVk);
    VkCommandBuffer AllocateCommandBuffer() const;

    FramePack       AcquireNextFrame(Swapchain& swapchain);
    void            PresentThisFrame(Swapchain& swapchain,
                                     const SemaphoreVariant& waitSemaphore);
    VkSemaphore     PrevFrameFinishSignal();
};