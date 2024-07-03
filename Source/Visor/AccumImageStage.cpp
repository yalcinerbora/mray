#include "AccumImageStage.h"
#include "VulkanAllocators.h"

#include "Core/Error.hpp"
#include "Core/TimelineSemaphore.h"
#include <BS/BS_thread_pool.hpp>

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
    , accumCompleteFence(std::exchange(other.accumCompleteFence, nullptr))
    , syncSemaphore(other.syncSemaphore)
    , threadPool(other.threadPool)
    , handlesVk(other.handlesVk)
    , pipeline(std::move(other.pipeline))
    , descriptorSets(std::move(other.descriptorSets))
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
    accumCompleteFence = std::exchange(other.accumCompleteFence, nullptr);
    syncSemaphore = other.syncSemaphore;
    threadPool = other.threadPool;
    timelineSemaphoreVk = std::exchange(other.timelineSemaphoreVk, nullptr);
    handlesVk = other.handlesVk;
    pipeline = std::move(other.pipeline);
    descriptorSets = std::move(other.descriptorSets);
    hdrImage = other.hdrImage;
    sampleImage = other.sampleImage;
    accumulateCommand = std::exchange(other.accumulateCommand, nullptr);
    return *this;
}

AccumImageStage::~AccumImageStage()
{
    Clear();
}

MRayError AccumImageStage::Initialize(TimelineSemaphore* ts,
                                      BS::thread_pool* tp,
                                      const std::string& execPath)
{
    syncSemaphore = ts;
    threadPool = tp;

    if(vkGetMemoryHostPointerProperties == nullptr)
    {
        auto func = vkGetDeviceProcAddr(handlesVk->deviceVk,
                                        "vkGetMemoryHostPointerPropertiesEXT");
        vkGetMemoryHostPointerProperties = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(func);
    }

    VkSemaphoreTypeCreateInfo semTypeCInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
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

    vkDestroySemaphore(handlesVk->deviceVk, timelineSemaphoreVk,
                       VulkanHostAllocator::Functions());
    vkFreeCommandBuffers(handlesVk->deviceVk,
                         handlesVk->mainCommandPool, 1,
                         &accumulateCommand);
    vkDestroyFence(handlesVk->deviceVk, accumCompleteFence,
                   VulkanHostAllocator::Functions());
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

    DescriptorBindList<ShaderBindingData> bindList =
    {
        {
            HDR_IMAGE_BIND_INDEX,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VkDescriptorImageInfo
            {
                .sampler = hdrImage->Sampler(),
                .imageView = hdrImage->View(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL
            },
        },
        {
            SAMPLE_IMAGE_BIND_INDEX,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VkDescriptorImageInfo
            {
                .sampler = sampleImage->Sampler(),
                .imageView = sampleImage->View(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL
            }
        }
    };
    pipeline.BindSetData(descriptorSets[0], bindList);
}

Optional<SemaphoreVariant> AccumImageStage::IssueAccumulation(SemaphoreVariant prevCmdSignal,
                                                              const RenderImageSection& section)
{
    // Pre-memcopy the buffer
    UniformBuffer buffer =
    {
        .imgResolution = hdrImage->Extent(),
        .pixStart = section.pixelMin,
        .pixEnd = section.pixelMax,
        .globalSampleWeight = section.globalWeight
    };
    std::memcpy(uniformBuffer.hostPtr, &buffer, sizeof(UniformBuffer));
    uniformBuffer.FlushRange(handlesVk->deviceVk);

    // TODO: Check if overlapping is allowed
    DescriptorBindList<ShaderBindingData> bindList =
    {
        {
            IN_PIXEL_BUFF_BIND_INDEX,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VkDescriptorBufferInfo
            {
                .buffer = foreignBuffer,
                .offset = section.pixelStartOffset,
                .range = VK_WHOLE_SIZE
            },
        },
        {
            IN_SAMPLE_BUFF_BIND_INDEX,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VkDescriptorBufferInfo
            {
                .buffer = foreignBuffer,
                .offset = section.sampleStartOffset,
                .range = VK_WHOLE_SIZE
            },
        }
    };
    pipeline.BindSetData(descriptorSets[0], bindList);

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
    Vector2ui groupSize = MathFunctions::DivideUp(totalPix, TPB);
    pipeline.BindPipeline(accumulateCommand);
    vkCmdDispatch(accumulateCommand, groupSize[0], groupSize[1], 1);
    vkEndCommandBuffer(accumulateCommand);

    // We need to wait the section to be ready.
    // Again we are waiting from host since not inter GPU
    // synch is available (Except on Linux I think, using the SYNC_FD
    // functionality)
    //
    // Tracer may abruptly terminated (crash probably),
    // so do not issue anything, return nullopt and
    // let the main render loop to terminate
    if(!syncSemaphore->Acquire(section.waitCounter))
        return std::nullopt;

    // ============= //
    //   SUBMISSON   //
    // ============= //
    uint64_t signalCounter = section.waitCounter + 1;
    std::array<VkSemaphoreSubmitInfo, 1> waitSemaphores =
    {
        VkSemaphoreSubmitInfo
        {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = prevCmdSignal.semHandle,
            .value = prevCmdSignal.value,
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
    vkQueueSubmit2(handlesVk->mainQueueVk, 1, &submitInfo, accumCompleteFence);

    // Launch a wait and reset task for this fence
    threadPool->detach_task([fence = this->accumCompleteFence,
                             sem = this->syncSemaphore,
                             device = this->handlesVk->deviceVk]()
    {
        // Wait over the fence
        // Thread pool is only used for saving the image and
        // doing this operation so it should be pretty fast (hopefully)
        vkWaitForFences(device, 1, &fence, VK_TRUE,
                        std::numeric_limits<uint64_t>::max());
        // Signal the MRay renderer that
        // this memory is available now.
        sem->Release();
        // Reset for the next issue
        vkResetFences(device, 1, &fence);
    });
    return SemaphoreVariant{signalCounter, timelineSemaphoreVk};
}

//SystemSemaphoreHandle AccumImageStage::GetSemaphoreOSHandle() const
//{
//    return systemSemHandle;
//}

size_t AccumImageStage::UniformBufferSize() const
{
    return sizeof(UniformBuffer);
}

void AccumImageStage::SetUniformBufferView(const UniformBufferMemView& ubo)
{
    uniformBuffer = ubo;
    DescriptorBindList<ShaderBindingData> bindList =
    {
        {
            UNIFORM_BIND_INDEX,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VkDescriptorBufferInfo
            {
                .buffer = ubo.bufferHandle,
                .offset = ubo.offset,
                .range = ubo.size
            },
        }
    };
    pipeline.BindSetData(descriptorSets[0], bindList);
}