#pragma once

#include <vulkan/vulkan.h>
#include <array>

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
    static constexpr uint32_t FRAME_COUNT = 1;
    static_assert(FRAME_COUNT == 1, "Current single frame in flight is used");

    private:
    template<class T>
    using FrameArray = std::array<T, FRAME_COUNT>;

    VkCommandPool                   commandPool = nullptr;
    VkDevice                        deviceVk    = nullptr;
    VkQueue                         mainQueueVk = nullptr;
    uint32_t                        frameIndex  = 0;
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
    void            PresentThisFrame(Swapchain& swapchain);
    VkSemaphore     PrevFrameFinishSignal();
};