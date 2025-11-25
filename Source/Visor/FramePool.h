#pragma once

#include <vulkan/vulkan.h>
#include "VulkanTypes.h"

struct MRayError;
class Swapchain;
struct VulkanSystemView;

static constexpr size_t MAX_WINDOW_FBO_COUNT = 8;

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
    // !!!!
    //
    // Most of the uniform data is nauively memcopied to
    // a device-visible host buffer every frame (for simplicity
    // this is not a game afterall)
    //
    // So be careful changing this and sending multiple frames.
    // Afaik all transient buffers should be duplicated for each
    // launched frame.
    //
    // !!!!
    static constexpr uint32_t FRAMES_IN_FLIGHT = 1;
    using CommandSubmitSems = std::array<VulkanBinarySemaphore, MAX_WINDOW_FBO_COUNT>;
    using ImgAvailSems = std::array<VulkanBinarySemaphore, FRAMES_IN_FLIGHT>;

    private:
    VkDevice            deviceVk    = nullptr;
    VkQueue             mainQueueVk = nullptr;
    VulkanFence         fence;
    VulkanCommandBuffer cBuffer;
    CommandSubmitSems   commandsSubmittedSems;
    ImgAvailSems        imageAvailableSems;
    uint32_t            launchFrameIndex = 0;

    public:
                    FramePool() = default;
                    FramePool(const FramePool&) = delete;
                    FramePool(FramePool&&) = default;
    FramePool&      operator=(const FramePool&) = delete;
    FramePool&      operator=(FramePool&&) = default;
                    ~FramePool();// = default;

    MRayError       Initialize(const VulkanSystemView& handlesVk);

    FramePack       AcquireNextFrame(Swapchain& swapchain);
    void            PresentThisFrame(Swapchain& swapchain,
                                     const VulkanTimelineSemaphore* extraWaitSem);
};