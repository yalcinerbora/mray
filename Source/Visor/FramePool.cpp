#include "FramePool.h"
#include "VisorWindow.h"
#include "VulkanTypes.h"
#include "VulkanAllocators.h"
#include "Core/Error.h"
#include "Core/DataStructures.h"

void FramePool::Clear()
{
    if(!deviceVk) return;

    for(size_t i = 0; i < FRAME_COUNT; i++)
    {
        vkDestroyFence(deviceVk, fences[i],
                       VulkanHostAllocator::Functions());
        vkDestroySemaphore(deviceVk, semaphores[i].imageAvailableSignal,
                           VulkanHostAllocator::Functions());
        vkDestroySemaphore(deviceVk, semaphores[i].commandsExecutedSignal,
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

    commandPool = other.commandPool;
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
    commandPool = handlesVk.mainCommandPool;

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
                          &semaphores[i].commandsExecutedSignal);
        vkCreateFence(handlesVk.deviceVk, &fenceCreateInfo,
                      VulkanHostAllocator::Functions(),
                      &fences[i]);
    }
    return MRayError::OK;
}

FramePack FramePool::AcquireNextFrame(Swapchain& swapchain)
{
    assert(frameIndex >= 0 && frameIndex < FRAME_COUNT);
    uint32_t frameIndexUInt = static_cast<uint32_t>(frameIndex);
    vkWaitForFences(deviceVk, 1, &fences[frameIndexUInt],
                    VK_TRUE, std::numeric_limits<uint64_t>::max());

    VkSemaphore imgAvailSem = semaphores[frameIndexUInt].imageAvailableSignal;
    FramebufferPack fbPack = swapchain.NextFrame(imgAvailSem);

    vkResetFences(deviceVk, 1, &fences[frameIndexUInt]);

    FramePack framePack = {};
    framePack.extent = fbPack.extent;
    framePack.img = fbPack.img;
    framePack.imgView = fbPack.imgView;
    framePack.fbo = fbPack.fbo;
    framePack.renderPass = fbPack.renderPass;
    framePack.commandBuffer = cBuffers[frameIndexUInt];
    return framePack;
}

void FramePool::PresentThisFrame(Swapchain& swapchain,
                                 const Optional<SemaphoreVariant>& extraWaitSem)
{
    assert(frameIndex >= 0 && frameIndex < FRAME_COUNT);
    uint32_t frameIndexUInt = static_cast<uint32_t>(frameIndex);
    VkSemaphore comExecSem = semaphores[frameIndexUInt].commandsExecutedSignal;
    VkSemaphore imgReadySem = semaphores[frameIndexUInt].imageAvailableSignal;

    // ============= //
    //   SUBMISSON   //
    // ============= //
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    StaticVector<VkSemaphoreSubmitInfo, 2> waitSemaphores;
    waitSemaphores.push_back(VkSemaphoreSubmitInfo
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = imgReadySem,
        .value = 0,
        // TODO change this to more fine-grained later maybe?
        .stageMask = waitStage,
        .deviceIndex = 0
    });
    if(extraWaitSem)
    {
        waitSemaphores.push_back(waitSemaphores.back());
        waitSemaphores.back().value = extraWaitSem.value().Value();
        waitSemaphores.back().semaphore = extraWaitSem.value().semHandle;
    }
    StaticVector<VkSemaphoreSubmitInfo, 2> signalSemaphores = waitSemaphores;
    signalSemaphores[0].value = 0;
    signalSemaphores[0].semaphore = comExecSem;
    if(extraWaitSem)
    {
        signalSemaphores[1].value = extraWaitSem.value().Value() + 1;
    }

    VkCommandBufferSubmitInfo commandSubmitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = cBuffers[frameIndexUInt],
        .deviceMask = 0

    };
    VkSubmitInfo2 submitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = static_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphoreInfos = waitSemaphores.data(),
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandSubmitInfo,
        .signalSemaphoreInfoCount = static_cast<uint32_t>(signalSemaphores.size()),
        .pSignalSemaphoreInfos = signalSemaphores.data()
    };
    // Finally submit!
    vkQueueSubmit2(mainQueueVk, 1, &submitInfo, fences[frameIndexUInt]);
    swapchain.PresentFrame(comExecSem);
    frameIndex = (frameIndex + 1) % FRAME_COUNT;
}

VkCommandBuffer FramePool::AllocateCommandBuffer() const
{
    VkCommandBuffer buffer;
    VkCommandBufferAllocateInfo cbuffAllocInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(deviceVk, &cbuffAllocInfo, &buffer);
    return buffer;
}

VkSemaphore FramePool::PrevFrameFinishSignal()
{
    int32_t prevFrame = MathFunctions::Roll(frameIndex - 1, 0, FRAME_COUNT);
    return semaphores[static_cast<size_t>(prevFrame)].imageAvailableSignal;
}