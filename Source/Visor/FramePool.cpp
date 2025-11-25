#include "FramePool.h"
#include "VisorWindow.h"
#include "VulkanTypes.h"
#include "VulkanAllocators.h"
#include "Core/Error.h"
#include "Core/DataStructures.h"

MRayError FramePool::Initialize(const VulkanSystemView& handlesVk)
{
    deviceVk = handlesVk.deviceVk;
    mainQueueVk = handlesVk.mainQueueVk;

    cBuffer = VulkanCommandBuffer(handlesVk);
    fence = VulkanFence(handlesVk);

    for(uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++)
        imageAvailableSems[i] = VulkanBinarySemaphore(handlesVk);

    // TODO: We are allocating extra semaphores
    // which may not be optimal
    for(uint32_t i = 0; i < MAX_WINDOW_FBO_COUNT; i++)
        commandsSubmittedSems[i] = VulkanBinarySemaphore(handlesVk);

    return MRayError::OK;
}

FramePool::~FramePool()
{
    for(uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++)
        imageAvailableSems[i] = VulkanBinarySemaphore();
    for(uint32_t i = 0; i < MAX_WINDOW_FBO_COUNT; i++)
        commandsSubmittedSems[i] = VulkanBinarySemaphore();
}

FramePack FramePool::AcquireNextFrame(Swapchain& swapchain)
{
    VkFence waitFence = fence;
    vkWaitForFences(deviceVk, 1, &waitFence, VK_TRUE,
                    std::numeric_limits<uint64_t>::max());

    const auto& imgAvailSem = imageAvailableSems[launchFrameIndex];
    FramebufferPack fbPack = swapchain.NextFrame(imgAvailSem);

    vkResetFences(deviceVk, 1, &waitFence);

    FramePack framePack = {};
    framePack.extent = fbPack.extent;
    framePack.img = fbPack.img;
    framePack.imgView = fbPack.imgView;
    framePack.fbo = fbPack.fbo;
    framePack.renderPass = fbPack.renderPass;
    framePack.commandBuffer = cBuffer;
    return framePack;
}

void FramePool::PresentThisFrame(Swapchain& swapchain,
                                 const VulkanTimelineSemaphore* extraWaitSem)
{
    const auto& comExecSem = commandsSubmittedSems[swapchain.IndexFBO()];
    const auto& imgReadySem = imageAvailableSems[launchFrameIndex];

    // ============== //
    //   SUBMISSION   //
    // ============== //
    VkPipelineStageFlags waitStage = (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    StaticVector<VkSemaphoreSubmitInfo, 2> waitSemaphores;
    waitSemaphores.push_back(imgReadySem.WaitInfo(waitStage));
    if(extraWaitSem)
        waitSemaphores.push_back(extraWaitSem->WaitInfo(waitStage));

    StaticVector<VkSemaphoreSubmitInfo, 2> signalSemaphores;
    signalSemaphores.push_back(comExecSem.SignalInfo(waitStage));
    if(extraWaitSem)
        signalSemaphores.push_back(extraWaitSem->SignalInfo(waitStage, 1));

    VkCommandBufferSubmitInfo commandSubmitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = cBuffer,
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
    vkQueueSubmit2(mainQueueVk, 1, &submitInfo, fence);
    swapchain.PresentFrame(comExecSem);

    launchFrameIndex = (launchFrameIndex + 1) % FRAMES_IN_FLIGHT;
}