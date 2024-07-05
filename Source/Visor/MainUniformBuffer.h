#pragma once

#include "VulkanTypes.h"
#include "VulkanAllocators.h"

struct UniformBufferMemView
{
    Byte*           hostPtr;
    VkBuffer        bufferHandle;
    VkDeviceMemory  memoryHandle;
    VkDeviceSize    offset;
    VkDeviceSize    size;

    void            FlushRange(VkDevice deviceVk);
};

class UniformMemoryRequesterI
{
    public:
    virtual         ~UniformMemoryRequesterI() = default;
    //
    virtual size_t  UniformBufferSize() const = 0;
    virtual void    SetUniformBufferView(const UniformBufferMemView& uniformBufferView) = 0;
};

class MainUniformBuffer
{
    static constexpr size_t VULKAN_META_ALIGNMENT = 256;

    private:
    const VulkanSystemView* handlesVk = nullptr;
    VulkanBuffer            mainUniformBuffer;
    VulkanDeviceMemory      mainMemory;
    size_t                  totalSize       = 0;
    Byte*                   alwaysMappedPtr = nullptr;

    public:
    // Constructors & Destructor
                        MainUniformBuffer(const VulkanSystemView&);
                        MainUniformBuffer(const MainUniformBuffer&) = delete;
                        MainUniformBuffer(MainUniformBuffer&&);
    MainUniformBuffer&  operator=(const MainUniformBuffer&) = delete;
    MainUniformBuffer&  operator=(MainUniformBuffer&&);
                        ~MainUniformBuffer();

    template<size_t N>
    void AllocateUniformBuffers(const std::array<UniformMemoryRequesterI*, N>& bufferRequesters);

    void FlushBuffer(VkCommandBuffer);
};


template<size_t N>
void MainUniformBuffer::AllocateUniformBuffers(const std::array<UniformMemoryRequesterI*, N>& bufferRequesters)
{
    std::array<size_t, N + 1> offsets = {};
    for(size_t i = 0; i < N; i++)
    {
        const auto* requester = bufferRequesters[i];
        offsets[i] = totalSize;
        totalSize += requester->UniformBufferSize();
        totalSize = MathFunctions::NextMultiple(totalSize, VULKAN_META_ALIGNMENT);
    }
    offsets[N] = totalSize;
    //
    mainUniformBuffer = VulkanBuffer(*handlesVk,
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
            .size = offsets[i + 1] - offsets[i]
        };
        bufferRequesters[i]->SetUniformBufferView(v);
    }
}
