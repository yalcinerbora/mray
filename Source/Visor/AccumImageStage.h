#pragma once

#include <vulkan/vulkan.h>

#include "MRay/TransferQueue.h"
#include "VulkanTypes.h"
#include "VulkanPipeline.h"

class AccumImageStage
{
    private:
    struct UniformBuffer
    {
        Vector2i    imgResolution;
        Vector2i    pixStart;
        Vector2i    pixEnd;
        float       globalSampleWeight;
    };

    private:
    static PFN_vkGetMemoryHostPointerPropertiesEXT vkGetMemoryHostPointerProperties;
    // Device memory backed stuff
    VulkanImage     pixelImage;
    VulkanImage     sampleImage;
    VulkanBuffer    uniformBuffer;
    VkDeviceMemory  imgAndBufferMemory      = nullptr;
    // Host memory backed stuff
    VkDeviceMemory  foreignMemory           = nullptr;
    VkBuffer        foreignBuffer           = nullptr;
    VkSemaphore     readyForReadSignalVk    = nullptr;
    VkSemaphore     readFinishedSignalVk    = nullptr;
    // Main system related stuff
    const VulkanSystemView* handlesVk       = nullptr;
    //
    VulkanComputePipeline pipeline;

    void    Clear();
    void    ImportMemory(const RenderBufferInfo&);

    public:
    // Constructor & Destructor
                        AccumImageStage(const VulkanSystemView&);
                        AccumImageStage(const AccumImageStage&) = delete;
                        AccumImageStage(AccumImageStage&&);
    AccumImageStage&    operator=(const AccumImageStage&) = delete;
    AccumImageStage&    operator=(AccumImageStage&&);
                        ~AccumImageStage();

    //
    MRayError           Initialize(const std::string& execPath, VkDescriptorPool pool);
    void                ResetBuffers(const RenderBufferInfo&);
    void                IssueAccumulation(VkCommandBuffer, const RenderImageSection&);
    void                IssueClear(VkCommandBuffer);
    const VulkanImage&  GetImage();
};

inline const VulkanImage& AccumImageStage::GetImage()
{
    return pixelImage;
}