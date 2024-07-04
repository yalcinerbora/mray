#include "RenderImagePool.h"
#include "Core/Log.h"

#include <BS/BS_thread_pool.hpp>
#include <array>

void RenderImagePool::Clear()
{
    if(!imgMemory.Memory()) return;

    // Now there may be a save process happening when a new
    // Image is requested, we need to wait it to finish.
    // We launched via "detach" so we dont have the
    VkSemaphoreWaitInfo semWait =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext = nullptr,
        .flags = 0,
        .semaphoreCount = 1,
        .pSemaphores = &saveSemaphore.semHandle,
        .pValues = &saveSemaphore.value
    };
    vkWaitSemaphores(handlesVk->deviceVk, &semWait,
                     std::numeric_limits<uint64_t>::max());


    // Now we can unmap and free memory
    vkUnmapMemory(handlesVk->deviceVk, stageMemory.Memory());
    imgMemory = VulkanDeviceMemory(handlesVk->deviceVk);
    stageMemory = VulkanDeviceMemory(handlesVk->deviceVk);

    vkDestroySemaphore(handlesVk->deviceVk, saveSemaphore.semHandle,
                       VulkanHostAllocator::Functions());
    vkDestroySemaphore(handlesVk->deviceVk, clearSemaphore.semHandle,
                       VulkanHostAllocator::Functions());

    using CList = std::array<VkCommandBuffer, 3>;
    CList commands = { hdrCopyCommand, sdrCopyCommand, clearCommand };
    vkFreeCommandBuffers(handlesVk->deviceVk, handlesVk->mainCommandPool, 3,
                         commands.data());
}

RenderImagePool::RenderImagePool(BS::thread_pool* threadPool,
                                 const VulkanSystemView& handles)
    : hdrImage(handles)
    , hdrSampleImage(handles)
    , sdrImage(handles)
    , outStageBuffer(handles)
    , stageMemory(handles.deviceVk)
    , imgMemory(handles.deviceVk)
    , handlesVk(&handles)
    , threadPool(threadPool)
    , imgLoader(CreateImageLoader())
{}

RenderImagePool::RenderImagePool(BS::thread_pool* threadPool,
                                 const VulkanSystemView& handles,
                                 const RenderImageInitInfo& initInfoIn)
    : RenderImagePool(threadPool, handles)
{
    initInfo = initInfoIn;
    // Work with floating point
    // TODO: Reduce floating point usage, currently formats are
    // technically a protocol between tracer and
    // visor, (we assume it is always XXX_FLOAT).
    auto sdrImgUsageFlags = (VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_STORAGE_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    auto hdrImgUsageFlags = sdrImgUsageFlags;

    hdrImage = (initInfo.isSpectralPack)
        ? VulkanImage(*handlesVk, VulkanSamplerMode::NEAREST,
                      VK_FORMAT_R32_SFLOAT, hdrImgUsageFlags,
                      initInfo.extent, initInfo.depth)
        : VulkanImage(*handlesVk, VulkanSamplerMode::NEAREST,
                      VK_FORMAT_R32G32B32A32_SFLOAT,
                      hdrImgUsageFlags, initInfo.extent);
    hdrSampleImage = VulkanImage(*handlesVk, VulkanSamplerMode::NEAREST,
                                 VK_FORMAT_R32_SFLOAT,
                                 hdrImgUsageFlags, initInfo.extent);
    sdrImage = VulkanImage(*handlesVk, VulkanSamplerMode::NEAREST,
                           VK_FORMAT_R32G32B32A32_SFLOAT,
                           sdrImgUsageFlags, initInfo.extent);

    size_t outBufferSize = std::max(hdrImage.MemRequirements().second,
                                    sdrImage.MemRequirements().second);
    outStageBuffer = VulkanBuffer(*handlesVk,
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                  outBufferSize);

    auto& deviceAllocator = VulkanDeviceAllocator::Instance();
    imgMemory = deviceAllocator.AllocateMultiObject(std::tie(hdrImage,
                                                             sdrImage,
                                                             hdrSampleImage),
                                                    VulkanDeviceAllocator::DEVICE);
    // If we create view before attaching memory,
    // validation layer whines. So creating views after allocation.
    hdrImage.CreateView();
    sdrImage.CreateView();
    hdrSampleImage.CreateView();

    stageMemory = deviceAllocator.AllocateMultiObject(std::tie(outStageBuffer),
                                                      VulkanDeviceAllocator::HOST_VISIBLE);

    void* ptr;
    vkMapMemory(handlesVk->deviceVk, stageMemory.Memory(),
                0, VK_WHOLE_SIZE, 0, &ptr);
    hStagePtr = static_cast<Byte*>(ptr);

    std::array<VkCommandBuffer, 3>  commands;
    VkCommandBufferAllocateInfo cBuffAllocInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = handlesVk->mainCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 3
    };
    vkAllocateCommandBuffers(handlesVk->deviceVk, &cBuffAllocInfo,
                             commands.data());
    vkAllocateCommandBuffers(handlesVk->deviceVk, &cBuffAllocInfo,
                             &sdrCopyCommand);
    hdrCopyCommand = commands[0];
    sdrCopyCommand = commands[1];
    clearCommand = commands[2];

    VkCommandBufferBeginInfo cBuffBeginInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
        .pInheritanceInfo = nullptr
    };
    // HDR
    vkBeginCommandBuffer(hdrCopyCommand, &cBuffBeginInfo);
    VkBufferImageCopy hdrCopyParams = hdrImage.FullCopyParams();
    vkCmdCopyImageToBuffer(hdrCopyCommand, hdrImage.Image(),
                           VK_IMAGE_LAYOUT_GENERAL,
                           outStageBuffer.Buffer(), 1,
                           &hdrCopyParams);
    vkEndCommandBuffer(hdrCopyCommand);

    // SDR
    vkBeginCommandBuffer(sdrCopyCommand, &cBuffBeginInfo);
    VkBufferImageCopy sdrCopyParams = hdrImage.FullCopyParams();
    vkCmdCopyImageToBuffer(sdrCopyCommand, sdrImage.Image(),
                           VK_IMAGE_LAYOUT_GENERAL,
                           outStageBuffer.Buffer(), 1,
                           &sdrCopyParams);
    vkEndCommandBuffer(sdrCopyCommand);

    // Clear
    VkImageSubresourceRange clearRange
    {
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 0u,
        .levelCount     = 1u,
        .baseArrayLayer = 0u,
        .layerCount     = 1u
    };
    VkClearColorValue clearColorValue = {};
    vkBeginCommandBuffer(clearCommand, &cBuffBeginInfo);
    // Change the HDR image state to writable
    std::array<VkImageMemoryBarrier, 3> imgBarrierInfo = {};
    imgBarrierInfo[0] =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_NONE,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = handlesVk->queueIndex,
        .dstQueueFamilyIndex = handlesVk->queueIndex,
        .image = hdrImage.Image(),
        .subresourceRange = VkImageSubresourceRange
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
    };
    imgBarrierInfo[1] = imgBarrierInfo[0];
    imgBarrierInfo[1].image = sdrImage.Image();
    imgBarrierInfo[2] = imgBarrierInfo[0];
    imgBarrierInfo[2].image = hdrSampleImage.Image();
    vkCmdPipelineBarrier(clearCommand,
                         VK_PIPELINE_STAGE_NONE,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr,
                         0, nullptr,
                         3, imgBarrierInfo.data());

    vkCmdClearColorImage(clearCommand, hdrImage.Image(),
                         VK_IMAGE_LAYOUT_GENERAL,
                         &clearColorValue, 1u,
                         &clearRange);
    vkCmdClearColorImage(clearCommand, sdrImage.Image(),
                         VK_IMAGE_LAYOUT_GENERAL,
                         &clearColorValue, 1u,
                         &clearRange);
    vkCmdClearColorImage(clearCommand, hdrSampleImage.Image(),
                         VK_IMAGE_LAYOUT_GENERAL,
                         &clearColorValue, 1u,
                         &clearRange);
    vkEndCommandBuffer(clearCommand);

    // Semaphore
    VkSemaphoreTypeCreateInfo semTypeCInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = saveSemaphore.value
    };
    VkSemaphoreCreateInfo semCInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &semTypeCInfo,
        .flags = 0
    };
    vkCreateSemaphore(handlesVk->deviceVk, &semCInfo,
                      VulkanHostAllocator::Functions(),
                      &saveSemaphore.semHandle);
    semTypeCInfo.initialValue = clearSemaphore.value;
    vkCreateSemaphore(handlesVk->deviceVk, &semCInfo,
                      VulkanHostAllocator::Functions(),
                      &clearSemaphore.semHandle);
}

RenderImagePool::RenderImagePool(RenderImagePool&& other)
    : hdrImage(std::move(other.hdrImage))
    , hdrSampleImage(std::move(other.hdrSampleImage))
    , sdrImage(std::move(other.sdrImage))
    , outStageBuffer(std::move(other.outStageBuffer))
    , stageMemory(std::move(other.stageMemory))
    , imgMemory(std::move(other.imgMemory))
    , handlesVk(other.handlesVk)
    , hStagePtr(other.hStagePtr)
    , hdrCopyCommand(std::exchange(other.hdrCopyCommand, nullptr))
    , sdrCopyCommand(std::exchange(other.sdrCopyCommand, nullptr))
    , clearCommand(std::exchange(other.clearCommand, nullptr))
    , threadPool(other.threadPool)
    , imgLoader(std::move(other.imgLoader))
    , saveSemaphore(std::exchange(other.saveSemaphore, {}))
    , clearSemaphore(std::exchange(other.clearSemaphore, {}))
    , initInfo(other.initInfo)
{}

RenderImagePool& RenderImagePool::operator=(RenderImagePool&& other)
{
    assert(this != &other);
    Clear();

    hdrImage = std::move(other.hdrImage);
    hdrSampleImage = std::move(other.hdrSampleImage);
    sdrImage = std::move(other.sdrImage);
    outStageBuffer = std::move(other.outStageBuffer);
    stageMemory = std::move(other.stageMemory);
    imgMemory = std::move(other.imgMemory);
    handlesVk = other.handlesVk;
    hStagePtr = other.hStagePtr;
    hdrCopyCommand = std::exchange(other.hdrCopyCommand, nullptr);
    sdrCopyCommand = std::exchange(other.sdrCopyCommand, nullptr);
    clearCommand = std::exchange(other.clearCommand, nullptr);
    threadPool = other.threadPool;
    imgLoader = std::move(other.imgLoader);
    saveSemaphore = std::exchange(other.saveSemaphore, {});
    clearSemaphore = std::exchange(other.clearSemaphore, {});
    initInfo = other.initInfo;
    return *this;
}

RenderImagePool::~RenderImagePool()
{
    Clear();
}

SemaphoreVariant RenderImagePool::SaveImage(IsHDRImage t, const RenderImageSaveInfo& fileOutInfo)
{
    // ============= //
    //   SUBMISSON   //
    // ============= //
    VkSemaphoreSubmitInfo waitSemaphore =
    {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = saveSemaphore.semHandle,
            .value = saveSemaphore.Value(),
            // TODO change this to more fine-grained later maybe?
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0
    };
    VkSemaphoreSubmitInfo signalSemaphore = waitSemaphore;
    signalSemaphore.value = saveSemaphore.Value() + 1;
    VkCommandBufferSubmitInfo commandSubmitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = (t == HDR) ? hdrCopyCommand : sdrCopyCommand,
        .deviceMask = 0
    };
    VkSubmitInfo2 submitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &waitSemaphore,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandSubmitInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalSemaphore
    };
    // Finally submit!
    vkQueueSubmit2(handlesVk->mainQueueVk, 1, &submitInfo, nullptr);

    // Copy everything on the stack.
    // Also copy semaphore and its value
    // Now during this thread's process we may encounter
    threadPool->detach_task([this, fileOutInfo, t, sem = saveSemaphore]()
    {
        // Wait for the copy to staging buffer to finish
        uint64_t waitValue = sem.Value() + 1;
        VkSemaphoreWaitInfo waitInfo =
        {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext = nullptr,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &sem.semHandle,
            .pValues = &waitValue
        };
        vkWaitSemaphores(handlesVk->deviceVk, &waitInfo,
                         std::numeric_limits<uint64_t>::max());

        // TODO: Find the proper tight size
        using enum MRayPixelEnum;
        size_t imgTightSize = 0;
        auto pixelType = (t == HDR) ? ((initInfo.isSpectralPack)
            ? MRayPixelTypeRT(MRayPixelType<MR_R_FLOAT>{})
            : MRayPixelTypeRT(MRayPixelType<MR_RGB_FLOAT>{}))
            : MRayPixelTypeRT(MRayPixelType<MR_RGB16_UNORM>{});
        auto imgFileType = (t == HDR)
            ? ImageType::EXR
            : ImageType::PNG;
        auto colorSpace = (t == HDR)
            ? initInfo.hdrColorSpace
            : initInfo.sdrColorSpace.first;
        Float gamma = (initInfo.isSpectralPack || t == HDR)
            ? Float(1.0)
            : initInfo.sdrColorSpace.second;
        //
        WriteImageParams imgInfo =
        {
            .header =
            {
                .dimensions = Vector3ui(initInfo.extent, 1u),
                .mipCount = 1,
                .pixelType = pixelType,
                .colorSpace = Pair(gamma, colorSpace)
            },
            .depth = initInfo.depth,
            .pixels = Span<const Byte>(hStagePtr, imgTightSize)
        };

        using namespace std::string_literals;
        std::string filePath = (fileOutInfo.prefix + "_"s +
                                std::to_string(fileOutInfo.sample) + "spp"s +
                                std::to_string(fileOutInfo.time) + "sec"s);

        MRayError e = imgLoader->WriteImage(imgInfo, filePath,
                                            imgFileType);

        // Signal ready for next command
        VkSemaphoreSignalInfo signalInfo =
        {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
            .pNext = nullptr,
            .semaphore = sem.semHandle,
            .value = sem.Value() + 2
        };
        // Signal the next save state, even if there is an error
        vkSignalSemaphore(handlesVk->deviceVk, &signalInfo);

        // Log the error
        if(e) MRAY_ERROR_LOG("{}", e.GetError());
    });

    // Our next wait cycle will start from +3.
    // - Copy to staging buffer (signalled +1)
    // - Main command buffer executed (signalled +2)
    // - Copy thread write finished signal (signalled +3)

    // Main command buffer signals us because I'm lazy
    //  to make an indirect synchronization

    //  will be twice as high from prev
    // Increment once more
    saveSemaphore.value += 3;
    // However next command (probably render) should only wait for the
    // memory transfer from device memory to host memory
    return SemaphoreVariant{saveSemaphore.value + 1, saveSemaphore.semHandle};
}

SemaphoreVariant RenderImagePool::IssueClear()
{
    // ============= //
    //   SUBMISSON   //
    // ============= //
    VkSemaphoreSubmitInfo waitSemaphore =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = clearSemaphore.semHandle,
        .value = clearSemaphore.Value(),
        // TODO change this to more fine-grained later maybe?
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0
    };
    VkSemaphoreSubmitInfo signalSemaphore =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = clearSemaphore.semHandle,
        .value = clearSemaphore.Value() + 1,
        // TODO change this to more fine-grained later maybe?
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0
    };
    VkCommandBufferSubmitInfo commandSubmitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = clearCommand,
        .deviceMask = 0
    };
    VkSubmitInfo2 submitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &waitSemaphore,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandSubmitInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalSemaphore
    };
    // Finally submit!
    vkQueueSubmit2(handlesVk->mainQueueVk, 1, &submitInfo, nullptr);

    clearSemaphore.value += 2;
    return clearSemaphore;
}