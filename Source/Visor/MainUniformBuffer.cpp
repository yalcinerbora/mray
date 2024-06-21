#include "MainUniformBuffer.h"

void UniformBufferMemView::FlushRange(VkDevice deviceVk)
{
    VkMappedMemoryRange mRange =
    {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext = nullptr,
        .memory = memoryHandle,
        .offset = offset,
        .size = size
    };
    vkFlushMappedMemoryRanges(deviceVk, 1, &mRange);
}

MainUniformBuffer::MainUniformBuffer(const VulkanSystemView& handles)
    : mainUniformBuffer(handles)
    , handlesVk(&handles)
{}

MainUniformBuffer::MainUniformBuffer(MainUniformBuffer&& other)
    : mainUniformBuffer(std::move(other.mainUniformBuffer))
    , totalSize(other.totalSize)
    , alwaysMappedPtr(other.alwaysMappedPtr)
    , mainMemory(std::move(other.mainMemory))
    , handlesVk(other.handlesVk)
{}

MainUniformBuffer& MainUniformBuffer::operator=(MainUniformBuffer&& other)
{
    assert(this != &other);

    if(mainMemory.Memory())
        vkUnmapMemory(handlesVk->deviceVk,
                      mainMemory.Memory());

    mainUniformBuffer = std::move(other.mainUniformBuffer);
    totalSize = other.totalSize;
    alwaysMappedPtr = other.alwaysMappedPtr;
    mainMemory = std::move(other.mainMemory);
    handlesVk = other.handlesVk;

    return *this;
}

MainUniformBuffer::~MainUniformBuffer()
{
    if(mainMemory.Memory())
        vkUnmapMemory(handlesVk->deviceVk,
                      mainMemory.Memory());
}

template<size_t N>
void MainUniformBuffer::AllocateUniformBuffers(std::array<UniformMemoryRequesterI*, N>& bufferRequesters)
{
    std::array<size_t, N> offsets = {};
    for(size_t i = 0 ; i < N; i++)
    {
        const auto* requester = bufferRequesters[i];
        offsets[i] = totalSize;
        totalSize += requester->UniformBufferSize();
        totalSize = MathFunctions::NextMultiple(totalSize, VULKAN_META_ALIGNMENT);
    }
    //
    mainUniformBuffer = VulkanBuffer(*handlesVk,
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                     totalSize);
    mainMemory = VulkanDeviceAllocator::Instance().AllocateMultiObject(std::tie(mainUniformBuffer),
                                                                       VulkanDeviceAllocator::HOST_VISIBLE);
    void* hPtr;
    vkMapMemory(handlesVk->deviceVk, mainMemory.Memory(), 0,
                totalSize, 0, &hPtr);
    alwaysMappedPtr = reinterpret_cast<Byte*>(hPtr);

    for(size_t i = 0; i < N; i++)
    {
        UniformBufferMemView v =
        {
            .hostPtr = alwaysMappedPtr + offsets[i],
            .bufferHandle = mainUniformBuffer.Buffer(),
            .memoryHandle = mainMemory.Memory(),
            .offset = offsets[i],
            .size = totalSize
        };
        bufferRequesters[i].SetUniformBufferView(v);
    }
}

void MainUniformBuffer::FlushBuffer(VkCommandBuffer cmd)
{
    VkMemoryBarrier memBarrier
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         1, &memBarrier, 0, nullptr,
                         0, nullptr);

    VkMappedMemoryRange memRange =
    {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext = nullptr,
        .memory = mainMemory.Memory(),
        .offset = 0,
        .size = totalSize
    };
    vkFlushMappedMemoryRanges(handlesVk->deviceVk, 1, &memRange);
}