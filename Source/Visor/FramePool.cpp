#include "FramePool.h"
#include "VisorWindow.h"
#include "VulkanTypes.h"
#include "VulkanAllocators.h"
#include "Core/Error.h"

void FramePool::Clear()
{
    if(!commandPool) return;

    vkDestroyCommandPool(deviceVk, commandPool,
                         VulkanHostAllocator::Functions());
    for(size_t i = 0; i < FRAME_COUNT; i++)
    {
        vkDestroyFence(deviceVk, fences[i],
                       VulkanHostAllocator::Functions());
        vkDestroySemaphore(deviceVk, semaphores[i].imageAvailableSignal,
                           VulkanHostAllocator::Functions());
        vkDestroySemaphore(deviceVk, semaphores[i].commandsRecordedSignal,
                           VulkanHostAllocator::Functions());
    }
}

FramePool::FramePool(FramePool&& other)
    : commandPool(std::exchange(other.commandPool, nullptr))
    , deviceVk(other.deviceVk)
    , mainQueueVk(other.mainQueueVk)
    , frameIndex(other.frameIndex)
{
    for(size_t i = 0; i < FRAME_COUNT; i++)
    {
        fences[i] = std::exchange(other.fences[i], nullptr);
        cBuffers[i] = std::exchange(other.cBuffers[i], nullptr);
        semaphores[i] = std::exchange(other.semaphores[i], FrameSemaphorePair{});
    }
}

FramePool& FramePool::operator=(FramePool&& other)
{
    Clear();

    commandPool = std::exchange(other.commandPool, nullptr);
    deviceVk = other.deviceVk;
    mainQueueVk = other.mainQueueVk;
    frameIndex = other.frameIndex;

    for(size_t i = 0; i < FRAME_COUNT; i++)
    {
        fences[i] = std::exchange(other.fences[i], nullptr);
        cBuffers[i] = std::exchange(other.cBuffers[i], nullptr);
        semaphores[i] = std::exchange(other.semaphores[i], FrameSemaphorePair{});
    }
    return *this;
}

FramePool::~FramePool()
{
    Clear();
}

MRayError FramePool::Initialize(const VulkanSystemView& handlesVk)
{
    deviceVk = handlesVk.deviceVk;
    mainQueueVk = handlesVk.mainQueueVk;

    // Command buffers
    VkCommandPoolCreateInfo cpCreateInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = handlesVk.queueIndex
    };
    vkCreateCommandPool(deviceVk, &cpCreateInfo,
                        VulkanHostAllocator::Functions(),
                        &commandPool);


    // Allocate all in one pass
    VkCommandBufferAllocateInfo cbuffAllocInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = FRAME_COUNT
    };
    vkAllocateCommandBuffers(handlesVk.deviceVk,
                             &cbuffAllocInfo, cBuffers.data());


    // Semaphores and fences
    VkSemaphoreCreateInfo semCreateInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0
    };
    VkFenceCreateInfo fenceCreateInfo =
    {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };
    for(size_t i = 0; i < FRAME_COUNT; i++)
    {
        vkCreateSemaphore(handlesVk.deviceVk, &semCreateInfo,
                          VulkanHostAllocator::Functions(),
                          &semaphores[i].imageAvailableSignal);
        vkCreateSemaphore(handlesVk.deviceVk, &semCreateInfo,
                          VulkanHostAllocator::Functions(),
                          &semaphores[i].commandsRecordedSignal);
        vkCreateFence(handlesVk.deviceVk, &fenceCreateInfo,
                      VulkanHostAllocator::Functions(),
                      &fences[i]);
    }
    return MRayError::OK;
}

FramePack FramePool::AcquireNextFrame(Swapchain& swapchain)
{
    VkSemaphore imgAvailSem = semaphores[frameIndex].imageAvailableSignal;
    FramebufferPack fbPack = swapchain.NextFrame(imgAvailSem);

    vkWaitForFences(deviceVk, 1, &fences[frameIndex],
                    VK_TRUE, std::numeric_limits<uint64_t>::max());
    vkResetFences(deviceVk, 1, &fences[frameIndex]);

    FramePack framePack = {};
    framePack.extent = fbPack.extent;
    framePack.img = fbPack.img;
    framePack.imgView = fbPack.imgView;
    framePack.fbo = fbPack.fbo;
    framePack.renderPass = fbPack.renderPass;
    framePack.commandBuffer = cBuffers[frameIndex];
    return framePack;
}

void FramePool::PresentThisFrame(Swapchain& swapchain)
{
    VkSemaphore imgAvailSem = semaphores[frameIndex].imageAvailableSignal;
    VkSemaphore comRecordSem = semaphores[frameIndex].commandsRecordedSignal;
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &imgAvailSem,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cBuffers[frameIndex],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &comRecordSem,
    };
    vkQueueSubmit(mainQueueVk, 1, &submitInfo, fences[frameIndex]);
    swapchain.PresentFrame(comRecordSem);
    frameIndex = (frameIndex + 1) % FRAME_COUNT;
}