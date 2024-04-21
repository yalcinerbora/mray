#pragma once

#include <vulkan/vulkan.h>

#include "MRay/TransferQueue.h"
#include "VulkanTypes.h"
#include "VulkanPipeline.h"
#include "MainUniformBuffer.h"

// TODO: Change this to a more generic way
#ifdef MRAY_WINDOWS
    #include <vulkan/vulkan_win32.h>
#endif

class AccumImageStage : UniformMemoryRequesterI
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
    // TODO: Change this to a more generic way
    static PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32Handle;

    // Device memory backed stuff
    UniformBufferMemView    uniformBuffer           = {};
    // Host memory backed stuff
    VkDeviceMemory          foreignMemory           = nullptr;
    VkBuffer                foreignBuffer           = nullptr;
    VkSemaphore             timelineSemaphoreVk     = nullptr;
    SystemSemaphoreHandle   systemSemHandle         = (MRAY_IS_ON_WINDOWS) ? nullptr : 0;
    // Main system related stuff
    const VulkanSystemView* handlesVk       = nullptr;
    //
    VulkanComputePipeline   pipeline;
    DescriptorSets          descriptorSets;
    //
    const VulkanImage*      hdrImage;
    const VulkanImage*      sampleImage;

    void                    Clear();

    public:
    // Constructor & Destructor
                        AccumImageStage(const VulkanSystemView&);
                        AccumImageStage(const AccumImageStage&) = delete;
                        AccumImageStage(AccumImageStage&&);
    AccumImageStage&    operator=(const AccumImageStage&) = delete;
    AccumImageStage&    operator=(AccumImageStage&&);
                        ~AccumImageStage();

    //
    MRayError               Initialize(const std::string& execPath);
    void                    ImportExternalHandles(const RenderBufferInfo&);
    void                    ChangeImage(const VulkanImage* hdrImageIn,
                                        const VulkanImage* sampleImageIn);
    void                    IssueAccumulation(VkCommandBuffer, const RenderImageSection&);
    SystemSemaphoreHandle   ExportSemaphore(const RenderBufferInfo&);

    SystemSemaphoreHandle   GetSemaphoreOSHandle() const;

    //
    size_t                  UniformBufferSize() const override;
    void                    SetUniformBufferView(const UniformBufferMemView& uniformBufferPtr) override;
};