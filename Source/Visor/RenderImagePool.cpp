#include "RenderImagePool.h"
#include <BS/BS_thread_pool.hpp>
#include <array>

void RenderImagePool::Clear()
{
    if(!imgMemory) return;

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
    vkUnmapMemory(handlesVk->deviceVk, stageMemory);
    vkFreeMemory(handlesVk->deviceVk, imgMemory,
                 VulkanHostAllocator::Functions());
    vkFreeMemory(handlesVk->deviceVk, stageMemory,
                 VulkanHostAllocator::Functions());

    vkDestroySemaphore(handlesVk->deviceVk, saveSemaphore.semHandle,
                       VulkanHostAllocator::Functions());
    vkFreeCommandBuffers(handlesVk->deviceVk, handlesVk->mainCommandPool, 1,
                         &hdrCopyCommand);
    vkFreeCommandBuffers(handlesVk->deviceVk, handlesVk->mainCommandPool, 1,
                         &sdrCopyCommand);
}

RenderImagePool::RenderImagePool(BS::thread_pool* threadPool,
                                 const VulkanSystemView& handles)
    : hdrImage(handles)
    , hdrSampleImage(handles)
    , sdrImage(handles)
    , outStageBuffer(handles)
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
    hdrImage = (initInfo.isSpectralPack)
        ? VulkanImage(*handlesVk, VK_FORMAT_R32_SFLOAT,
                      initInfo.extent, initInfo.depth)
        : VulkanImage(*handlesVk, VK_FORMAT_R32G32B32_SFLOAT,
                      initInfo.extent);
    hdrSampleImage = VulkanImage(*handlesVk, VK_FORMAT_R32_SFLOAT,
                                 initInfo.extent);
    sdrImage = VulkanImage(*handlesVk, VK_FORMAT_R32G32B32_SFLOAT,
                           initInfo.extent);

    size_t outBufferSize = std::max(hdrImage.MemRequirements().first,
                                    sdrImage.MemRequirements().first);
    outStageBuffer = VulkanBuffer(*handlesVk,
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                  outBufferSize);

    auto& deviceAllocator = VulkanDeviceAllocator::Instance();
    imgMemory = deviceAllocator.AllocateMultiObject(std::tie(hdrImage,
                                                             sdrImage,
                                                             hdrSampleImage),
                                                    VulkanDeviceAllocator::DEVICE);

    stageMemory = deviceAllocator.AllocateMultiObject(std::tie(outStageBuffer),
                                                      VulkanDeviceAllocator::HOST_VISIBLE);

    void* ptr;
    vkMapMemory(handlesVk->deviceVk, stageMemory, 0, VK_WHOLE_SIZE, 0,
                &ptr);
    hStagePtr = static_cast<Byte*>(ptr);


    VkCommandBufferAllocateInfo cBuffAllocInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = handlesVk->mainCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(handlesVk->deviceVk, &cBuffAllocInfo,
                             &hdrCopyCommand);
    vkAllocateCommandBuffers(handlesVk->deviceVk, &cBuffAllocInfo,
                             &sdrCopyCommand);

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
}

RenderImagePool::RenderImagePool(RenderImagePool&& other)
    : hdrImage(std::move(other.hdrImage))
    , hdrSampleImage(std::move(other.hdrSampleImage))
    , sdrImage(std::move(other.sdrImage))
    , outStageBuffer(std::move(other.outStageBuffer))
    , stageMemory(std::exchange(other.stageMemory, nullptr))
    , imgMemory(std::exchange(other.imgMemory, nullptr))
    , handlesVk(other.handlesVk)
    , hStagePtr(other.hStagePtr)
    , hdrCopyCommand(std::exchange(other.hdrCopyCommand, nullptr))
    , sdrCopyCommand(std::exchange(other.sdrCopyCommand, nullptr))
    , threadPool(other.threadPool)
    , imgLoader(std::move(other.imgLoader))
{}

RenderImagePool& RenderImagePool::operator=(RenderImagePool&& other)
{
    assert(this != &other);
    Clear();

    hdrImage = std::move(other.hdrImage);
    hdrSampleImage = std::move(other.hdrSampleImage);
    sdrImage = std::move(other.sdrImage);
    outStageBuffer = std::move(other.outStageBuffer);
    stageMemory = std::exchange(other.stageMemory, nullptr);
    imgMemory = std::exchange(other.imgMemory, nullptr);
    handlesVk = other.handlesVk;
    return *this;
}

RenderImagePool::~RenderImagePool()
{
    Clear();
}

void RenderImagePool::SaveImage(VkSemaphore prevCmdSignal,
                                IsHDRImage t, const RenderImageSaveInfo& fileOutInfo)

{
    // ============= //
    //   SUBMISSON   //
    // ============= //
    uint64_t waitCounter = saveSemaphore.value;
    uint64_t signalCounter = waitCounter + 1;
    std::array<VkSemaphoreSubmitInfo, 2> waitSemaphores =
    {
        VkSemaphoreSubmitInfo
        {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = prevCmdSignal,
            .value = 0,
            // TODO change this to more fine-grained later maybe?
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0
        },
        VkSemaphoreSubmitInfo
        {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = saveSemaphore.semHandle,
            .value = waitCounter,
            // TODO change this to more fine-grained later maybe?
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0
        }
    };
    VkSemaphoreSubmitInfo signalSemaphores = waitSemaphores[1];
    signalSemaphores.value = signalCounter;
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
        .waitSemaphoreInfoCount = 2,
        .pWaitSemaphoreInfos = waitSemaphores.data(),
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandSubmitInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalSemaphores
    };
    // Finally submit!
    vkQueueSubmit2(handlesVk->mainQueueVk, 1, &submitInfo, nullptr);
    saveSemaphore.value++;

    // Copy everything on the stack.
    // Also copy semaphore and its value
    // Now during this thread's process we may encounter
    threadPool->detach_task([this, fileOutInfo, t, sem = saveSemaphore]()
    {
        VkSemaphoreWaitInfo waitInfo =
        {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext = nullptr,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &sem.semHandle,
            .pValues = &sem.value
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
        WriteImage<2> imgInfo =
        {
            .header =
            {
                .dimensions = Vector2ui(initInfo.extent),
                .mipCount = 1,
                .pixelType = pixelType,
                .colorSpace = colorSpace,
                .gamma = gamma
            },
            .depth = initInfo.depth,
            .pixels = Span<const Byte>(hStagePtr, imgTightSize)
        };

        using namespace std::string_literals;
        std::string filePath = (fileOutInfo.prefix + "_"s +
                                std::to_string(fileOutInfo.sample) + "spp"s +
                                std::to_string(fileOutInfo.time) + "sec"s);

        MRayError e = imgLoader->WriteImage2D(imgInfo, filePath,
                                              imgFileType);

        // Signal ready for next command
        VkSemaphoreSignalInfo signalInfo =
        {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
            .pNext = nullptr,
            .semaphore = sem.semHandle,
            .value = sem.value + 1
        };
        // Signal the next save state, even if there is an error
        vkSignalSemaphore(handlesVk->deviceVk, &signalInfo);

        // Log the error
        if(e) MRAY_ERROR_LOG("{}", e.GetError());
    });

    // Our next wait will be twice as high from prev
    // Increment once more
    saveSemaphore.value++;
}