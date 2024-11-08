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
    : handlesVk(&handles)
{}

MainUniformBuffer::MainUniformBuffer(MainUniformBuffer&& other)
    : handlesVk(other.handlesVk)
    , mainUniformBuffer(std::move(other.mainUniformBuffer))
    , mainMemory(std::move(other.mainMemory))
    , totalSize(other.totalSize)
    , alwaysMappedPtr(other.alwaysMappedPtr)
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

void MainUniformBuffer::FlushBuffer(VkCommandBuffer cmd)
{
    VkMemoryBarrier memBarrier
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &memBarrier,
                         0, nullptr,
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