#include "MainUniformBuffer.h"


MainUniformBuffer::MainUniformBuffer(const VulkanSystemView& handles)
    : mainUniformBuffer(handles)
    , handlesVk(&handles)
{}

MainUniformBuffer::MainUniformBuffer(MainUniformBuffer&& other)
    : mainUniformBuffer(std::move(other.mainUniformBuffer))
    , totalSize(other.totalSize)
    , alwaysMappedPtr(other.alwaysMappedPtr)
    , mainMemory(std::exchange(other.mainMemory, nullptr))
    , handlesVk(other.handlesVk)
{}

MainUniformBuffer& MainUniformBuffer::operator=(MainUniformBuffer&& other)
{
    assert(this != &other);

    if(mainMemory)
        vkFreeMemory(handlesVk->deviceVk, mainMemory,
                     VulkanHostAllocator::Functions());

    mainUniformBuffer = std::move(other.mainUniformBuffer);
    totalSize = other.totalSize;
    alwaysMappedPtr = other.alwaysMappedPtr;
    mainMemory = std::exchange(other.mainMemory, nullptr);
    handlesVk = other.handlesVk;

    return *this;
}

MainUniformBuffer::~MainUniformBuffer()
{
    if(mainMemory)
    {
        vkUnmapMemory(handlesVk->deviceVk, mainMemory);
        vkFreeMemory(handlesVk->deviceVk, mainMemory,
                     VulkanHostAllocator::Functions());
    }
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
    vkMapMemory(handlesVk->deviceVk, mainMemory, 0,
                totalSize, 0, &hPtr);
    alwaysMappedPtr = reinterpret_cast<Byte*>(hPtr);

    for(size_t i = 0; i < N; i++)
    {
        UniformBufferMemView v =
        {
            .hostPtr = alwaysMappedPtr + offsets[i],
            .bufferHandle = mainUniformBuffer.Buffer(),
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
        .memory = mainMemory,
        .offset = 0,
        .size = totalSize
    };
    vkFlushMappedMemoryRanges(handlesVk->deviceVk, 1, &memRange);
}