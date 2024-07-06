#pragma once

#include <vulkan/vulkan.h>

#include "Common/RenderImageStructs.h"
#include "Core/System.h"

#include "VulkanTypes.h"
#include "VulkanPipeline.h"
#include "MainUniformBuffer.h"

class TimelineSemaphore;
namespace BS { class thread_pool; }

class AccumImageStage : public UniformMemoryRequesterI
{
    private:
    struct UniformBuffer
    {
        Vector2ui   imgResolution;
        Vector2ui   pixStart;
        Vector2ui   pixEnd;
        float       globalSampleWeight;
    };

    static constexpr uint32_t UNIFORM_BIND_INDEX        = 0;
    static constexpr uint32_t HDR_IMAGE_BIND_INDEX      = 1;
    static constexpr uint32_t SAMPLE_IMAGE_BIND_INDEX   = 2;
    static constexpr uint32_t IN_PIXEL_BUFF_BIND_INDEX  = 3;
    static constexpr uint32_t IN_SAMPLE_BUFF_BIND_INDEX = 4;

    using DescriptorSets = typename VulkanComputePipeline::DescriptorSets;

    private:
    static PFN_vkGetMemoryHostPointerPropertiesEXT vkGetMemoryHostPointerProperties;
    //
    const VulkanSystemView* handlesVk = nullptr;
    // Device memory backed stuff
    UniformBufferMemView    uniformBuffer           = {};
    // Host memory backed stuff
    VulkanDeviceMemory      foreignMemory;
    VulkanBuffer            foreignBuffer;
    VulkanFence             accumCompleteFence;
    // Pre-recorded command
    VulkanCommandBuffer     accumulateCommand;
    // Main system related stuff
    TimelineSemaphore*      syncSemaphore   = nullptr;
    BS::thread_pool*        threadPool      = nullptr;
    //
    VulkanComputePipeline   pipeline;
    DescriptorSets          descriptorSets;
    //
    const VulkanImage*      hdrImage            = nullptr;
    const VulkanImage*      sampleImage         = nullptr;
    void                    Clear();

    public:
    // Constructor & Destructor
                AccumImageStage() = default;
    //
    MRayError   Initialize(const VulkanSystemView& handlesVk,
                           TimelineSemaphore* ts,
                           BS::thread_pool* threadPool,
                           const std::string& execPath);
    void        ImportExternalHandles(const RenderBufferInfo&);
    void        ChangeImage(const VulkanImage* hdrImageIn,
                            const VulkanImage* sampleImageIn);

    bool        IssueAccumulation(const RenderImageSection&,
                                  const VulkanTimelineSemaphore&);
    //
    size_t      UniformBufferSize() const override;
    void        SetUniformBufferView(const UniformBufferMemView& uniformBufferPtr) override;
    size_t      UsedGPUMemBytes() const;
};