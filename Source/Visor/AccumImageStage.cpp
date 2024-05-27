#include "AccumImageStage.h"
#include "VulkanAllocators.h"

#ifdef MRAY_WINDOWS
    #define WINDOWS_LEAN_AND_MEAN
    #include <windows.h>
    #include <vulkan/vulkan_win32.h>

    // Changing this to translation unit global, this will be used only here
    PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32Handle = nullptr;
#endif

PFN_vkGetMemoryHostPointerPropertiesEXT AccumImageStage::vkGetMemoryHostPointerProperties = nullptr;

AccumImageStage::AccumImageStage(const VulkanSystemView& handles)
    : handlesVk(&handles)
    , pipeline(handles.deviceVk)
{}

AccumImageStage::AccumImageStage(AccumImageStage&& other)
    : uniformBuffer(other.uniformBuffer)
    , foreignMemory(std::exchange(other.foreignMemory, nullptr))
    , foreignBuffer(std::exchange(other.foreignBuffer, nullptr))
    , timelineSemaphoreVk(std::exchange(other.timelineSemaphoreVk, nullptr))
    , systemSemHandle(std::exchange(other.systemSemHandle, SystemSemaphoreHandle(0)))
    , handlesVk(other.handlesVk)
    , pipeline(std::move(other.pipeline))
    , hdrImage(other.hdrImage)
    , sampleImage(other.sampleImage)
    , accumulateCommand(std::exchange(other.accumulateCommand, nullptr))
{}

AccumImageStage& AccumImageStage::operator=(AccumImageStage&& other)
{
    assert(this != &other);
    Clear();
    uniformBuffer = other.uniformBuffer;
    foreignMemory = std::exchange(other.foreignMemory, nullptr);
    foreignBuffer = std::exchange(other.foreignBuffer, nullptr);
    systemSemHandle = std::exchange(other.systemSemHandle, SystemSemaphoreHandle(0));
    timelineSemaphoreVk = std::exchange(other.timelineSemaphoreVk, nullptr);
    handlesVk = other.handlesVk;
    pipeline = std::move(other.pipeline);
    hdrImage = other.hdrImage;
    sampleImage = other.sampleImage;
    accumulateCommand = std::exchange(other.accumulateCommand, nullptr);
    return *this;
}

AccumImageStage::~AccumImageStage()
{
    Clear();
}

MRayError AccumImageStage::Initialize(const std::string& execPath)
{
    if(vkGetMemoryHostPointerProperties == nullptr)
    {
        auto func = vkGetDeviceProcAddr(handlesVk->deviceVk,
                                        "vkGetMemoryHostPointerPropertiesEXT");
        vkGetMemoryHostPointerProperties = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(func);
    }

    #ifdef MRAY_WINDOWS
        if(vkGetSemaphoreWin32Handle == nullptr)
        {
            auto func = vkGetDeviceProcAddr(handlesVk->deviceVk,
                                            "vkGetSemaphoreWin32HandleKHR");
            vkGetSemaphoreWin32Handle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(func);
        }
    #endif

    static constexpr auto ExternalSemType = (MRAY_IS_ON_WINDOWS)
        ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
        : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkExportSemaphoreCreateInfoKHR exportInfo =
    {
        .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
        .pNext = nullptr,
        .handleTypes = ExternalSemType
    };
    VkSemaphoreTypeCreateInfo semTypeCInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = &exportInfo,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0
    };
    VkSemaphoreCreateInfo semCInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &semTypeCInfo,
        .flags = 0
    };
    vkCreateSemaphore(handlesVk->deviceVk, &semCInfo,
                      VulkanHostAllocator::Functions(),
                      &timelineSemaphoreVk);

    #ifdef MRAY_WINDOWS
    {
        if(systemSemHandle) CloseHandle(systemSemHandle);
        VkSemaphoreGetWin32HandleInfoKHR win32SemInfo =
        {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
            .pNext = nullptr,
            .semaphore = timelineSemaphoreVk,
            .handleType = ExternalSemType
        };
        vkGetSemaphoreWin32Handle(handlesVk->deviceVk, &win32SemInfo,
                                  &systemSemHandle);
    }
    #elif defined MRAY_LINUX
    {
        //#error "TODO: Implement!!"
    }
    #endif

    using namespace std::string_literals;
    MRayError e = pipeline.Initialize(
    {
        {
            {UNIFORM_BIND_INDEX,        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
            {HDR_IMAGE_BIND_INDEX,      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            {SAMPLE_IMAGE_BIND_INDEX,   VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            {IN_PIXEL_BUFF_BIND_INDEX,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {IN_SAMPLE_BUFF_BIND_INDEX, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
        }
    },
    "Shaders/AccumInput.spv"s, execPath, "KCAccumulateInputs"s);
    if(e) return e;

    descriptorSets = pipeline.GenerateDescriptorSets(handlesVk->mainDescPool);

    // Allocate Command buffer
    VkCommandBufferAllocateInfo cBuffAllocInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = handlesVk->mainCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(handlesVk->deviceVk, &cBuffAllocInfo,
                             &accumulateCommand);

    return MRayError::OK;
}

void AccumImageStage::Clear()
{
    if(foreignBuffer)
    {

        vkFreeMemory(handlesVk->deviceVk, foreignMemory,
                     VulkanHostAllocator::Functions());
        vkDestroyBuffer(handlesVk->deviceVk, foreignBuffer,
                        VulkanHostAllocator::Functions());
    }
    if(!timelineSemaphoreVk) return;

    #ifdef MRAY_WINDOWS
    {
        CloseHandle(systemSemHandle);
    }
    #elif defined MRAY_LINUX
    {
        //#error "TODO: Implement!!"
        close(systemSemHandle);
    }
    #endif
    vkDestroySemaphore(handlesVk->deviceVk, timelineSemaphoreVk,
                       VulkanHostAllocator::Functions());
    vkFreeCommandBuffers(handlesVk->deviceVk,
                         handlesVk->mainCommandPool, 1,
                         &accumulateCommand);
}

void AccumImageStage::ImportExternalHandles(const RenderBufferInfo& rbI)
{
    // Clear old
    if(foreignMemory)
    {
        vkFreeMemory(handlesVk->deviceVk, foreignMemory,
                     VulkanHostAllocator::Functions());
        vkDestroyBuffer(handlesVk->deviceVk, foreignBuffer,
                        VulkanHostAllocator::Functions());
    }

    static constexpr auto ForeignHostBit = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT;
    static const VkExternalMemoryBufferCreateInfo EXT_BUFF_INFO =
    {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = ForeignHostBit
    };
    VkBufferCreateInfo buffInfo =
    {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext = &EXT_BUFF_INFO,
        .flags = 0,
        .size = rbI.totalSize,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_CONCURRENT,
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices = &handlesVk->queueIndex,
    };
    vkCreateBuffer(handlesVk->deviceVk, &buffInfo,
                   VulkanHostAllocator::Functions(),
                   &foreignBuffer);

    VkMemoryHostPointerPropertiesEXT hostProps = {};
    vkGetMemoryHostPointerProperties(handlesVk->deviceVk,
                                     ForeignHostBit,
                                     rbI.data, &hostProps);

    // Allocation
    VkImportMemoryHostPointerInfoEXT hostImportInfo =
    {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .pNext = nullptr,
        .handleType = ForeignHostBit,
        .pHostPointer = rbI.data,
    };
    VkMemoryAllocateInfo memAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &hostImportInfo,
        .allocationSize = rbI.totalSize,
        .memoryTypeIndex = hostProps.memoryTypeBits
    };
    vkAllocateMemory(handlesVk->deviceVk, &memAllocInfo,
                     VulkanHostAllocator::Functions(),
                     &foreignMemory);
    // Binding
    vkBindBufferMemory(handlesVk->deviceVk, foreignBuffer, foreignMemory, 0);

}

void AccumImageStage::ChangeImage(const VulkanImage* hdrImageIn,
                                  const VulkanImage* sampleImageIn)
{
    hdrImage = hdrImageIn;
    sampleImage = sampleImageIn;

    std::array< VkWriteDescriptorSet, 2> writeInfo;
    VkDescriptorImageInfo hdrImgInfo =
    {
        .sampler = hdrImageIn->Sampler(),
        .imageView = hdrImageIn->View(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };
    writeInfo[0] = VkWriteDescriptorSet
    {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = descriptorSets[0],
        .dstBinding = HDR_IMAGE_BIND_INDEX,
        .dstArrayElement = 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &hdrImgInfo,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr
    };
    //
    VkDescriptorImageInfo sampleImgInfo =
    {
        .sampler = hdrImageIn->Sampler(),
        .imageView = hdrImageIn->View(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };
    writeInfo[1] = writeInfo[0];
    writeInfo[1].dstBinding = SAMPLE_IMAGE_BIND_INDEX;
    writeInfo[1].pImageInfo = &sampleImgInfo;

    vkUpdateDescriptorSets(handlesVk->deviceVk,
                           2, writeInfo.data(), 0, nullptr);

}

SemaphoreVariant AccumImageStage::IssueAccumulation(VkSemaphore prevCmdSignal,
                                                    const RenderImageSection& section)
{
    UniformBuffer buffer =
    {
        .imgResolution = hdrImage->Extent(),
        .pixStart = section.pixelMin,
        .pixEnd = section.pixelMax,
        .globalSampleWeight = section.globalWeight
    };
    std::memcpy(uniformBuffer.hostPtr, &buffer, sizeof(UniformBuffer));

    std::array<VkWriteDescriptorSet, 2> writeInfo;
    VkDescriptorBufferInfo pixBuffInfo =
    {
        .buffer = foreignBuffer,
        .offset = section.pixelStartOffset,
        // TODO: Check aliasing
        .range = VK_WHOLE_SIZE
    };
    writeInfo[0] = VkWriteDescriptorSet
    {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = descriptorSets[0],
        .dstBinding = IN_PIXEL_BUFF_BIND_INDEX,
        .dstArrayElement = 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = nullptr,
        .pBufferInfo = &pixBuffInfo,
        .pTexelBufferView = nullptr
    };
    //
    VkDescriptorBufferInfo sampleBuffInfo =
    {
        .buffer = foreignBuffer,
        .offset = section.sampleStartOffset,
        // TODO: Check aliasing
        .range = VK_WHOLE_SIZE
    };
    writeInfo[1] = writeInfo[0];
    writeInfo[1].dstBinding = IN_SAMPLE_BUFF_BIND_INDEX;
    writeInfo[1].pBufferInfo = &sampleBuffInfo;

    vkUpdateDescriptorSets(handlesVk->deviceVk,
                           2, writeInfo.data(), 0, nullptr);

    // ============= //
    //    DISPATCH   //
    // ============= //
    // Record command buffer
    VkCommandBufferBeginInfo bInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr
    };
    vkBeginCommandBuffer(accumulateCommand, &bInfo);
    Vector2ui totalPix = section.pixelMax - section.pixelMin;
    Vector2ui TPB = Vector2ui(VulkanComputePipeline::TPB_2D_X,
                              VulkanComputePipeline::TPB_2D_Y);
    Vector2ui groupSize = (totalPix + TPB - Vector2ui(1)) / TPB;
    vkCmdDispatch(accumulateCommand, groupSize[0], groupSize[1], 1);
    vkEndCommandBuffer(accumulateCommand);

    // ============= //
    //   SUBMISSON   //
    // ============= //
    uint64_t waitCounter = section.waitCounter;
    uint64_t signalCounter = section.waitCounter + 1;
    std::array<VkSemaphoreSubmitInfo, 2> waitSemaphores =
    {
        VkSemaphoreSubmitInfo
        {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = timelineSemaphoreVk,
            .value = waitCounter,
            // TODO change this to more fine-grained later maybe?
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0
        },
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
    };
    VkSemaphoreSubmitInfo signalSemaphores = waitSemaphores[0];
    signalSemaphores.value = signalCounter;
    VkCommandBufferSubmitInfo commandSubmitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = accumulateCommand,
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
    return SemaphoreVariant{signalCounter, timelineSemaphoreVk};
}

SystemSemaphoreHandle AccumImageStage::GetSemaphoreOSHandle() const
{
    return systemSemHandle;
}

size_t AccumImageStage::UniformBufferSize() const
{
    return sizeof(UniformBuffer);
}

void AccumImageStage::SetUniformBufferView(const UniformBufferMemView& ubo)
{
    uniformBuffer = ubo;
}