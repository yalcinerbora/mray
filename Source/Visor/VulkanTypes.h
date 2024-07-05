#pragma once

#include <vulkan/vulkan.h>
#include <limits>

#include "Core/Vector.h"
#include "Core/DataStructures.h"
#include "Core/Types.h"

#include "VulkanAllocators.h"

namespace VkConversions
{
    Pair<MRayColorSpaceEnum, Float>
    VkToMRayColorSpace(VkColorSpaceKHR);
}

enum class VulkanSemaphoreType
{
    BINARY,
    TIMELINE
};

template<VulkanSemaphoreType T>
inline constexpr bool IS_TIMELINE_SEM = (T == VulkanSemaphoreType::TIMELINE);

struct VulkanSystemView
{
    VkInstance          instanceVk          = nullptr;
    VkPhysicalDevice    pDeviceVk           = nullptr;
    VkDevice            deviceVk            = nullptr;
    uint32_t            queueIndex          = std::numeric_limits<uint32_t>::max();
    VkQueue             mainQueueVk         = nullptr;
    VkCommandPool       mainCommandPool     = nullptr;
    VkDescriptorPool    mainDescPool        = nullptr;
    VkSampler           llnSampler          = nullptr;
    VkSampler           nnnSampler          = nullptr;
};

enum class VulkanSamplerMode
{
    NEAREST,
    LINEAR
};

class VulkanFence
{
    private:
    const VulkanSystemView* handlesVk   = nullptr;
    VkFence                 handle      = nullptr;

    public:
    // Constructors & Destructor
                    VulkanFence() = default;
                    VulkanFence(const VulkanSystemView&,
                                bool isSignalled = true);
                    VulkanFence(const VulkanFence&) = delete;
                    VulkanFence(VulkanFence&&);
    VulkanFence&    operator=(const VulkanFence&) = delete;
    VulkanFence&    operator=(VulkanFence&&);
                    ~VulkanFence();

    operator VkFence();
};

template<VulkanSemaphoreType T>
class VulkanSemaphore
{
    private:
    const VulkanSystemView* handlesVk   = nullptr;
    uint64_t                waitValue   = 0;
    VkSemaphore             handle      = nullptr;

    public:
    // Constructors & Destructor
                        VulkanSemaphore() = default;
                        VulkanSemaphore(const VulkanSystemView&);
                        VulkanSemaphore(const VulkanSemaphore&) = delete;
                        VulkanSemaphore(VulkanSemaphore&&);
    VulkanSemaphore&    operator=(const VulkanSemaphore&) = delete;
    VulkanSemaphore&    operator=(VulkanSemaphore&&);
                        ~VulkanSemaphore();


    VkSemaphoreSubmitInfo WaitInfo(VkPipelineStageFlags2) const;
    VkSemaphoreSubmitInfo SignalInfo(VkPipelineStageFlags2,
                                     uint64_t delta) const requires(IS_TIMELINE_SEM<T>);
    VkSemaphoreSubmitInfo SignalInfo(VkPipelineStageFlags2) const requires(!IS_TIMELINE_SEM<T>);

    void ChangeNextWait(uint64_t delta) requires(IS_TIMELINE_SEM<T>);
    void HostWait(uint64_t delta) const requires(IS_TIMELINE_SEM<T>);
    void HostSignal(uint64_t delta) const requires(IS_TIMELINE_SEM<T>);

    VkSemaphore Handle() const requires(!IS_TIMELINE_SEM<T>);
};

using VulkanBinarySemaphore = VulkanSemaphore<VulkanSemaphoreType::BINARY>;
using VulkanTimelineSemaphore = VulkanSemaphore<VulkanSemaphoreType::TIMELINE>;

class VulkanCommandBuffer
{
    private:
    const VulkanSystemView* handlesVk   = nullptr;
    VkCommandBuffer         commandBuff = nullptr;

    public:
    // Constructors & Destructor
                            VulkanCommandBuffer() = default;
                            VulkanCommandBuffer(const VulkanSystemView&);
                            VulkanCommandBuffer(const VulkanCommandBuffer&) = delete;
                            VulkanCommandBuffer(VulkanCommandBuffer&&);
    VulkanCommandBuffer&    operator=(const VulkanCommandBuffer&) = delete;
    VulkanCommandBuffer&    operator=(VulkanCommandBuffer&&);
                            ~VulkanCommandBuffer();

    operator VkCommandBuffer();
};

class VulkanImage
{
    private:
    const VulkanSystemView* handlesVk = nullptr;
    VkImage         imgVk       = nullptr;
    VkFormat        formatVk    = VK_FORMAT_UNDEFINED;
    VkImageView     viewVk      = nullptr;
    VkSampler       samplerVk   = nullptr;
    Vector2ui       extent      = Vector2ui::Zero();
    uint32_t        depth       = 0;

    public:
    // Constructors & Destructor
                    VulkanImage() = default;
                    VulkanImage(const VulkanSystemView&,
                                VulkanSamplerMode, VkFormat,
                                VkImageUsageFlags, Vector2ui pixRes,
                                uint32_t depth = 1);
                    VulkanImage(const VulkanImage&) = delete;
                    VulkanImage(VulkanImage&&);
    VulkanImage&    operator=(const VulkanImage&) = delete;
    VulkanImage&    operator=(VulkanImage&&);
                    ~VulkanImage();
    //
    SizeAlignPair   MemRequirements() const;
    void            AttachMemory(VkDeviceMemory, VkDeviceSize);
    void            IssueClear(VkCommandBuffer, VkClearColorValue);
    void            CreateView();

    VkImage         Image() const;
    VkSampler       Sampler() const;
    Vector2ui       Extent() const;
    VkImageView     View() const;

    //
    VkBufferImageCopy FullCopyParams() const;
};

class VulkanBuffer
{
    private:
    const VulkanSystemView* handlesVk   = nullptr;
    VkBuffer                bufferVk    = nullptr;

    public:
    // Constructors & Destructor
                    VulkanBuffer() = default;
                    VulkanBuffer(const VulkanSystemView&,
                                 VkBufferUsageFlags usageFlags,
                                 size_t size, bool isForeign = false);
                    VulkanBuffer(const VulkanBuffer&) = delete;
                    VulkanBuffer(VulkanBuffer&&);
    VulkanBuffer&   operator=(const VulkanBuffer&) = delete;
    VulkanBuffer&   operator=(VulkanBuffer&&);
                    ~VulkanBuffer();
    //
    SizeAlignPair   MemRequirements() const;
    void            AttachMemory(VkDeviceMemory, VkDeviceSize);
    VkBuffer        Buffer() const;
};

inline VkImage VulkanImage::Image() const
{
    return imgVk;
}
inline VkImageView VulkanImage::View() const
{
    assert(viewVk != nullptr);
    return viewVk;
}

inline VkSampler VulkanImage::Sampler() const
{
    return samplerVk;
}

inline Vector2ui VulkanImage::Extent() const
{
    return extent;
}

inline VkBuffer VulkanBuffer::Buffer() const
{
    return bufferVk;
}

static_assert(VulkanMemObjectC<VulkanImage>);
static_assert(VulkanMemObjectC<VulkanBuffer>);

template<VulkanSemaphoreType T>
VulkanSemaphore<T>::VulkanSemaphore(const VulkanSystemView& handles)
    : handlesVk(&handles)
{
    VkSemaphoreType semType = (IS_TIMELINE_SEM<T>)
        ? VK_SEMAPHORE_TYPE_TIMELINE
        : VK_SEMAPHORE_TYPE_BINARY;

    VkSemaphoreTypeCreateInfo tCInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = semType,
        .initialValue = waitValue
    };
    VkSemaphoreCreateInfo cInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &tCInfo,
        .flags = 0
    };
    vkCreateSemaphore(handlesVk->deviceVk, &cInfo,
                      VulkanHostAllocator::Functions(),
                      &handle);
}

template<VulkanSemaphoreType T>
VulkanSemaphore<T>::VulkanSemaphore(VulkanSemaphore&& other)
    : handlesVk(std::exchange(other.handlesVk, nullptr))
    , handle(std::exchange(other.handle, nullptr))
    , waitValue(other.waitValue)
{}

template<VulkanSemaphoreType T>
VulkanSemaphore<T>& VulkanSemaphore<T>::operator=(VulkanSemaphore&& other)
{
    assert(this != &other);
    if(handlesVk)
    {
        vkDestroySemaphore(handlesVk->deviceVk, handle,
                           VulkanHostAllocator::Functions());
    }
    handlesVk = std::exchange(other.handlesVk, nullptr);
    handle = std::exchange(other.handle, nullptr);
    waitValue = other.waitValue;
    return *this;
}

template<VulkanSemaphoreType T>
VulkanSemaphore<T>::~VulkanSemaphore()
{
    if(!handlesVk) return;

    vkDestroySemaphore(handlesVk->deviceVk, handle,
                       VulkanHostAllocator::Functions());
}

template<VulkanSemaphoreType T>
VkSemaphoreSubmitInfo VulkanSemaphore<T>::WaitInfo(VkPipelineStageFlags2 stages) const
{
    // This function is for verbosity, (getting "wait"/"signal info
    // is better than getting "submit" info)
    VkSemaphoreSubmitInfo result =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = handle,
        .value = waitValue,
        .stageMask = stages,
        .deviceIndex = 0
    };
    return result;
}

template<VulkanSemaphoreType T>
VkSemaphoreSubmitInfo VulkanSemaphore<T>::SignalInfo(VkPipelineStageFlags2 stages,
                                                  uint64_t delta) const
requires(IS_TIMELINE_SEM<T>)
{
    assert(delta != 0);
    VkSemaphoreSubmitInfo result =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = handle,
        .value = waitValue + delta,
        .stageMask = stages,
        .deviceIndex = 0
    };
    return result;
}

template<VulkanSemaphoreType T>
VkSemaphoreSubmitInfo VulkanSemaphore<T>::SignalInfo(VkPipelineStageFlags2 stages) const
requires(!IS_TIMELINE_SEM<T>)
{
    VkSemaphoreSubmitInfo result =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = handle,
        .value = 0,
        .stageMask = stages,
        .deviceIndex = 0
    };
    return result;
}

template<VulkanSemaphoreType T>
void VulkanSemaphore<T>::ChangeNextWait(uint64_t delta)
requires(IS_TIMELINE_SEM<T>)
{
    waitValue += delta;
}

template<VulkanSemaphoreType T>
void VulkanSemaphore<T>::HostWait(uint64_t delta) const
requires(IS_TIMELINE_SEM<T>)
{
    uint64_t curWaitValue = waitValue + delta;
    VkSemaphoreWaitInfo waitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext = nullptr,
        .flags = 0,
        .semaphoreCount = 1,
        .pSemaphores = &handle,
        .pValues = &curWaitValue
    };
    vkWaitSemaphores(handlesVk->deviceVk, &waitInfo,
                     std::numeric_limits<uint64_t>::max());
}

template<VulkanSemaphoreType T>
void VulkanSemaphore<T>::HostSignal(uint64_t delta) const
requires(IS_TIMELINE_SEM<T>)
{
    assert(delta != 0);
    VkSemaphoreSignalInfo signalInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .pNext = nullptr,
        .semaphore = handle,
        .value = waitValue + delta
    };
    // Signal the next save state, even if there is an error
    vkSignalSemaphore(handlesVk->deviceVk, &signalInfo);
}

template<VulkanSemaphoreType T>
VkSemaphore  VulkanSemaphore<T>::Handle() const
requires(!IS_TIMELINE_SEM<T>)
{
    return handle;
}