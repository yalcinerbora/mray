#include "AccumImageStage.h"
#include "VulkanAllocators.h"

#include "Core/Error.hpp"
#include "Core/TimelineSemaphore.h"
#include <BS/BS_thread_pool.hpp>

PFN_vkGetMemoryHostPointerPropertiesEXT AccumImageStage::vkGetMemoryHostPointerProperties = nullptr;

MRayError AccumImageStage::Initialize(const VulkanSystemView& handles,
                                      TimelineSemaphore* ts,
                                      BS::thread_pool* tp,
                                      const std::string& execPath)
{
    handlesVk = &handles;
    syncSemaphore = ts;
    threadPool = tp;
    accumulateCommand = VulkanCommandBuffer(*handlesVk);
    accumCompleteFence = VulkanFence(*handlesVk, false);

    if(vkGetMemoryHostPointerProperties == nullptr)
    {
        auto func = vkGetDeviceProcAddr(handlesVk->deviceVk,
                                        "vkGetMemoryHostPointerPropertiesEXT");
        vkGetMemoryHostPointerProperties = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(func);
    }

    using namespace std::string_literals;
    MRayError e = pipeline.Initialize(handlesVk->deviceVk,
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
    return MRayError::OK;
}

void AccumImageStage::ImportExternalHandles(const RenderBufferInfo& rbI)
{
    foreignBuffer = VulkanBuffer(*handlesVk,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 rbI.totalSize, true);

    VkMemoryHostPointerPropertiesEXT hostProps = {};
    hostProps.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    vkGetMemoryHostPointerProperties(handlesVk->deviceVk,
                                     VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                     rbI.data, &hostProps);

    auto& dAllocator = VulkanDeviceAllocator::Instance();
    foreignMemory = dAllocator.AllocateForeignObject(foreignBuffer, rbI.data,
                                                     rbI.totalSize,
                                                     hostProps.memoryTypeBits);
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

bool AccumImageStage::IssueAccumulation(const RenderImageSection& section,
                                        const VulkanTimelineSemaphore& imgWriteSemaphore)
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
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VkDescriptorBufferInfo
            {
                .buffer = foreignBuffer.Buffer(),
                .offset = section.pixelStartOffset,
                .range = VK_WHOLE_SIZE
            },
        },
        {
            IN_SAMPLE_BUFF_BIND_INDEX,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VkDescriptorBufferInfo
            {
                .buffer = foreignBuffer.Buffer(),
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

    using MathFunctions::DivideUp;
    Vector2ui totalPix = section.pixelMax - section.pixelMin;
    Vector2ui groupSize = DivideUp(totalPix,
                                   VulkanComputePipeline::TPB_2D);
    pipeline.BindPipeline(accumulateCommand);
    pipeline.BindSet(accumulateCommand, 0, descriptorSets[0]);
    vkCmdDispatch(accumulateCommand, groupSize[0], groupSize[1], 1);
    vkEndCommandBuffer(accumulateCommand);

    // We need to wait the section to be ready.
    // Again we are waiting from host since inter GPU
    // synch is not available (Except on Linux I think, using the SYNC_FD
    // functionality)
    //
    // Tracer may abruptly terminated (crash probably),
    // so do not issue anything, return nullopt and
    // let the main render loop to terminate
    if(!syncSemaphore->Acquire(section.waitCounter))
        return false;

    MRAY_LOG("[Visor] AcquiredImg {}", section.waitCounter);

    // ============= //
    //   SUBMISSON   //
    // ============= //
    auto allStages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo waitSemaphore = imgWriteSemaphore.WaitInfo(allStages);
    VkSemaphoreSubmitInfo signalSemaphores = imgWriteSemaphore.SignalInfo(allStages, 1);
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
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &waitSemaphore,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandSubmitInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalSemaphores
    };

    // Finally submit!
    vkQueueSubmit2(handlesVk->mainQueueVk, 1, &submitInfo, accumCompleteFence);

    // Launch a wait and reset task for this fence
    threadPool->detach_task([fence = &this->accumCompleteFence,
                             sem = this->syncSemaphore,
                             device = this->handlesVk->deviceVk]()
    {
        VkFence fenceHandle = *fence;
        // Wait over the fence
        // Thread pool is only used for saving the image and
        // doing this operation so it should be pretty fast (hopefully)
        vkWaitForFences(device, 1, &fenceHandle, VK_TRUE,
                        std::numeric_limits<uint64_t>::max());
        MRAY_LOG("[Visor] Released Img\n"
                 "----------------------");
        // Signal the MRay renderer that
        // this memory is available now.
        sem->Release();
        // Reset for the next issue
        vkResetFences(device, 1, &fenceHandle);
    });
    return true;
}

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