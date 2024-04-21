#pragma once

#include "VulkanTypes.h"
#include "VulkanAllocators.h"

struct UniformBufferMemView
{
    Byte*           hostPtr;
    VkBuffer        bufferHandle;
    VkDeviceSize    offset;
    VkDeviceSize    size;
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
    VulkanBuffer    mainUniformBuffer;
    size_t          totalSize           = 0;
    Byte*           alwaysMappedPtr     = nullptr;
    VkDeviceMemory  mainMemory          = nullptr;
    const VulkanSystemView* handlesVk   = nullptr;

    public:
    // Constructors & Destructor
                        MainUniformBuffer(const VulkanSystemView&);
                        MainUniformBuffer(const MainUniformBuffer&) = delete;
                        MainUniformBuffer(MainUniformBuffer&&);
    MainUniformBuffer&  operator=(const MainUniformBuffer&) = delete;
    MainUniformBuffer&  operator=(MainUniformBuffer&&);
                        ~MainUniformBuffer();

    template<size_t N>
    void AllocateUniformBuffers(std::array<UniformMemoryRequesterI*, N>& bufferRequesters);

    void FlushBuffer(VkCommandBuffer);
};