#pragma once

#include "Core/DataStructures.h"
#include <vulkan/vulkan.h>
#include "Core/Expected.h"

static constexpr size_t VISOR_MAX_SHADER_BINDINGS = 8;
static constexpr size_t VISOR_MAX_SHADER_SETS = 4;

template <class T>
using DescriptorSetList = StaticVector<T, VISOR_MAX_SHADER_SETS>;
template <class T>
using DescriptorBindList = StaticVector<T, VISOR_MAX_SHADER_BINDINGS>;

template <class T>
using Descriptor2DList = DescriptorSetList<DescriptorBindList<T>>;

struct ShaderBindingData
{
    using DescriptorInfo = Variant<VkDescriptorBufferInfo, VkDescriptorImageInfo>;
    uint32_t            index;
    VkDescriptorType    type;
    DescriptorInfo      dataInfo;
};

struct ShaderBindingInfo
{
    uint32_t bindingPoint;
    VkDescriptorType type;
    uint32_t elementCount = 1;
};

// Only compute pipelines are needed, since rendering
// is done via other means.
class VulkanComputePipeline
{
    public:
    using SetLayouts        = DescriptorSetList<VkDescriptorSetLayout>;
    using DescriptorSets    = DescriptorSetList<VkDescriptorSet>;

    static constexpr uint32_t   TPB_1D = 256;
    static constexpr auto       TPB_2D = Vector2ui(16, 16);

    private:
    VkDevice            deviceVk        = nullptr;
    VkPipeline          computePipeline = nullptr;
    VkPipelineLayout    pipelineLayout  = nullptr;
    SetLayouts          setLayouts;

    static Expected<std::vector<Byte>> DevourFile(const std::string& shaderName,
                                                  const std::string& executablePath);

    void                    Clear();
    public:
                            VulkanComputePipeline() = default;
                            VulkanComputePipeline(const VulkanComputePipeline&) = delete;
                            VulkanComputePipeline(VulkanComputePipeline&&);
    VulkanComputePipeline&  operator=(const VulkanComputePipeline&) = delete;
    VulkanComputePipeline&  operator=(VulkanComputePipeline&&);
                            ~VulkanComputePipeline();

    MRayError               Initialize(VkDevice deviceVk,
                                       const Descriptor2DList<ShaderBindingInfo>& bindingInfo,
                                       const std::string& shaderName,
                                       const std::string& executablePath,
                                       const std::string& entryPointName,
                                       uint32_t pushConstantSize = 0);
    DescriptorSets          GenerateDescriptorSets(VkDescriptorPool pool);
    void                    BindSetData(VkDescriptorSet descriptorSet,
                                        const DescriptorBindList<ShaderBindingData>& bindingDataList);
    void                    BindSet(VkCommandBuffer cmd, uint32_t setIndex,
                                    VkDescriptorSet set);
    void                    BindPipeline(VkCommandBuffer cmd);

    template<class T>
    void                    PushConstant(VkCommandBuffer cmd, const T&);
};


template<class T>
inline
void VulkanComputePipeline::PushConstant(VkCommandBuffer cmd, const T& data)
{
    using Math::NextMultiple;
    uint32_t size = NextMultiple(uint32_t(sizeof(T)), 4u);
    vkCmdPushConstants(cmd, pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0u, size, &data);
}