#include "RenderImagePool.h"
#include "Core/Log.h"

#include <BS/BS_thread_pool.hpp>
#include <array>

constexpr VkImageMemoryBarrier GENERIC_IMG_MEM_BARRIER =
{
    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_NONE,
    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
    .srcQueueFamilyIndex = 0,
    .dstQueueFamilyIndex = 0,
    .image = nullptr,
    .subresourceRange = VkImageSubresourceRange
    {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
    },
};

void RenderImagePool::GenerateSDRCopyCommand()
{
    // SDR image should be in read only optimal
    // from the fragment shader
    VkImageMemoryBarrier imgBarrierInfo =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = handlesVk->queueIndex,
        .dstQueueFamilyIndex = handlesVk->queueIndex,
        .image = sdrImage.Image(),
        .subresourceRange = VkImageSubresourceRange
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    vkCmdPipelineBarrier(sdrCopyCommand,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr,
                         0, nullptr,
                         1, &imgBarrierInfo);

    VkBufferImageCopy sdrCopyParams = hdrImage.FullCopyParams();
    vkCmdCopyImageToBuffer(sdrCopyCommand, sdrImage.Image(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           outStageBuffer.Buffer(), 1,
                           &sdrCopyParams);

    // Convert it back
    std::swap(imgBarrierInfo.srcAccessMask,
              imgBarrierInfo.dstAccessMask);
    std::swap(imgBarrierInfo.oldLayout,
              imgBarrierInfo.newLayout);
    vkCmdPipelineBarrier(sdrCopyCommand,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr,
                         0, nullptr,
                         1, &imgBarrierInfo);
}

void RenderImagePool::GenerateHDRCopyCommand()
{
    // HDR image should be in general format
    // from a compute shader
    VkImageMemoryBarrier imgBarrierInfo =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
        }
    };
    vkCmdPipelineBarrier(hdrCopyCommand,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr,
                         0, nullptr,
                         1, &imgBarrierInfo);

    VkBufferImageCopy hdrCopyParams = hdrImage.FullCopyParams();
    vkCmdCopyImageToBuffer(hdrCopyCommand, hdrImage.Image(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           outStageBuffer.Buffer(), 1,
                           &hdrCopyParams);

    // Convert it back
    std::swap(imgBarrierInfo.srcAccessMask,
              imgBarrierInfo.dstAccessMask);
    std::swap(imgBarrierInfo.oldLayout,
              imgBarrierInfo.newLayout);
    vkCmdPipelineBarrier(hdrCopyCommand,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr,
                         0, nullptr,
                         1, &imgBarrierInfo);
}

void RenderImagePool::GenerateClearCommand()
{
    // Clear
    VkImageSubresourceRange clearRange
    {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0u,
        .levelCount = 1u,
        .baseArrayLayer = 0u,
        .layerCount = 1u
    };
    VkClearColorValue clearColorValue = {};
    // Change the HDR image state to writable
    std::array<VkImageMemoryBarrier, 3> imgBarrierInfoList = {};
    imgBarrierInfoList[0] =
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
        }
    };
    // For sample count image make it writable as well
    imgBarrierInfoList[1] = imgBarrierInfoList[0];
    imgBarrierInfoList[1].image = hdrSampleImage.Image();
    // And for the sdr image
    imgBarrierInfoList[2] = imgBarrierInfoList[0];
    imgBarrierInfoList[2].image = sdrImage.Image();
    // Finally the barrier
    vkCmdPipelineBarrier(clearCommand,
                         VK_PIPELINE_STAGE_NONE,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr,
                         0, nullptr,
                         3, imgBarrierInfoList.data());

    //vkCmdClearColorImage(clearCommand, hdrImage.Image(),
    //                     VK_IMAGE_LAYOUT_GENERAL,
    //                     &clearColorValue, 1u,
    //                     &clearRange);
    vkCmdClearColorImage(clearCommand, hdrSampleImage.Image(),
                         VK_IMAGE_LAYOUT_GENERAL,
                         &clearColorValue, 1u,
                         &clearRange);
    //vkCmdClearColorImage(clearCommand, sdrImage.Image(),
    //                     VK_IMAGE_LAYOUT_GENERAL,
    //                     &clearColorValue, 1u,
    //                     &clearRange);

    // Make SDR image read optimal, since tonemap stage
    // expects the image to be previously read optimal.
    imgBarrierInfoList[0].image = sdrImage.Image();
    imgBarrierInfoList[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imgBarrierInfoList[0].dstAccessMask = (VK_ACCESS_SHADER_READ_BIT |
                                           VK_ACCESS_SHADER_WRITE_BIT);
    imgBarrierInfoList[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarrierInfoList[0].newLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
    // Finally the barrier
    vkCmdPipelineBarrier(clearCommand,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT), 0,
                         0, nullptr,
                         0, nullptr,
                         1, imgBarrierInfoList.data());
}

RenderImagePool::RenderImagePool()
    : imgLoader(nullptr, nullptr)
{}

RenderImagePool::RenderImagePool(BS::thread_pool* tp,
                                 const VulkanSystemView& handles,
                                 const RenderImageInitInfo& initInfoIn)
    : handlesVk(&handles)
    , threadPool(tp)
    , imgLoader(CreateImageLoader())
    , initInfo(initInfoIn)
{
    VkImageMemoryBarrier imgBarrierInfo = GENERIC_IMG_MEM_BARRIER;
    imgBarrierInfo.srcQueueFamilyIndex = handlesVk->queueIndex;
    imgBarrierInfo.dstQueueFamilyIndex = handlesVk->queueIndex;

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
                           VK_FORMAT_R16G16B16A16_UNORM,
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

    // Staging memory related
    stageMemory = deviceAllocator.AllocateMultiObject(std::tie(outStageBuffer),
                                                      VulkanDeviceAllocator::HOST_VISIBLE);
    void* ptr;
    vkMapMemory(handlesVk->deviceVk, stageMemory.Memory(),
                0, VK_WHOLE_SIZE, 0, &ptr);
    hStagePtr = static_cast<Byte*>(ptr);

    // Command related
    hdrCopyCommand = VulkanCommandBuffer(*handlesVk);
    sdrCopyCommand = VulkanCommandBuffer(*handlesVk);
    clearCommand = VulkanCommandBuffer(*handlesVk);

    VkCommandBufferBeginInfo cBuffBeginInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
        .pInheritanceInfo = nullptr
    };
    // HDR
    vkBeginCommandBuffer(hdrCopyCommand, &cBuffBeginInfo);
    GenerateHDRCopyCommand();
    vkEndCommandBuffer(hdrCopyCommand);

    // SDR
    vkBeginCommandBuffer(sdrCopyCommand, &cBuffBeginInfo);
    GenerateSDRCopyCommand();
    vkEndCommandBuffer(sdrCopyCommand);

    // Clear
    vkBeginCommandBuffer(clearCommand, &cBuffBeginInfo);
    GenerateClearCommand();
    vkEndCommandBuffer(clearCommand);
}

void RenderImagePool::SaveImage(IsHDRImage t, const RenderImageSaveInfo& fileOutInfo,
                                const VulkanTimelineSemaphore& imgSem)
{
    // ============= //
    //   SUBMISSON   //
    // ============= //
    auto allStages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo waitSemaphore = imgSem.WaitInfo(allStages);
    VkSemaphoreSubmitInfo signalSemaphore = imgSem.SignalInfo(allStages, 1);
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
    threadPool->detach_task([this, fileOutInfo, t, &imgSem]()
    {
        // Wait for the copy to staging buffer to finish
        imgSem.HostWait(1);
        // TODO: Find the proper tight size
        using enum MRayPixelEnum;
        auto outPixelType = (t == HDR) ? ((initInfo.isSpectralPack)
            ? MRayPixelTypeRT(MRayPixelType<MR_R_FLOAT>{})
            : MRayPixelTypeRT(MRayPixelType<MR_RGB_FLOAT>{}))
            : MRayPixelTypeRT(MRayPixelType<MR_RGB16_UNORM>{});
        auto inPixelType = (t == HDR) ? ((initInfo.isSpectralPack)
            ? MRayPixelTypeRT(MRayPixelType<MR_R_FLOAT>{})
            : MRayPixelTypeRT(MRayPixelType<MR_RGBA_FLOAT>{}))
            : MRayPixelTypeRT(MRayPixelType<MR_RGBA16_UNORM>{});
        auto imgFileType = (t == HDR)
            ? ImageType::EXR
            : ImageType::PNG;
        auto colorSpace = (t == HDR)
            ? initInfo.hdrColorSpace
            : initInfo.sdrColorSpace.first;
        Float gamma = (initInfo.isSpectralPack || t == HDR)
            ? Float(1.0)
            : initInfo.sdrColorSpace.second;
        size_t paddedImageSize = inPixelType.PixelSize() * initInfo.extent.Multiply();
        //
        WriteImageParams imgInfo =
        {
            .header =
            {
                .dimensions = Vector3ui(initInfo.extent, 1u),
                .mipCount = 1,
                .pixelType = outPixelType,
                .colorSpace = Pair(gamma, colorSpace)
            },
            .depth = initInfo.depth,
            .inputType = inPixelType,
            .pixels = Span<const Byte>(hStagePtr, paddedImageSize)
        };

        using namespace std::string_literals;
        std::string filePath = MRAY_FORMAT("{:s}_{:f}spp_{:f}sec",
                                           fileOutInfo.prefix,
                                           fileOutInfo.sample,
                                           fileOutInfo.time);
        // Vulkan images have top left starting coords,
        // so flip image while writing
        MRayError e = imgLoader->WriteImage(imgInfo, filePath,
                                            imgFileType,
                                            ImageFlagTypes::FLIP_Y_COORDINATE);

        // Signal ready for next command
        //imgSem.HostSignal(2);

        // Log the error
        if(e) MRAY_ERROR_LOG("{}", e.GetError());
    });
}

void RenderImagePool::IssueClear(const VulkanTimelineSemaphore& imgSem)
{
    // ============= //
    //   SUBMISSON   //
    // ============= //
    auto allStages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo waitSemaphore = imgSem.WaitInfo(allStages);
    VkSemaphoreSubmitInfo signalSemaphore = imgSem.SignalInfo(allStages, 1);
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
}

size_t RenderImagePool::UsedGPUMemBytes() const
{
    // Only show the device's memory here, skip staging memory
    return imgMemory.SizeBytes();
}