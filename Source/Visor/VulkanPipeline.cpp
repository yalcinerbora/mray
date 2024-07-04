#include "VulkanPipeline.h"

#include "Core/Error.h"
#include "Core/Filesystem.h"
#include "Core/Expected.h"
#include "Core/Error.h"
#include "Core/Error.hpp"

#include "VulkanAllocators.h"
#include "VulkanTypes.h"

#include <fstream>

Expected<std::vector<Byte>>
VulkanComputePipeline::DevourFile(const std::string& shaderName,
                                  const std::string& executablePath)
{
    std::string fullPath = Filesystem::RelativePathToAbsolute(shaderName,
                                                              executablePath);
    std::streamoff size = std::ifstream(fullPath,
                                        std::ifstream::ate |
                                        std::ifstream::binary).tellg();
    assert(size == MathFunctions::NextMultiple(size, std::streamoff(4)));
    std::vector<Byte> source(static_cast<size_t>(size), Byte(0));
    std::ifstream shaderFile = std::ifstream(fullPath, std::ios::binary);

    if(!shaderFile.is_open())
        return MRayError("Unable to open shader file \"{}\"",
                         fullPath);
    shaderFile.read(reinterpret_cast<char*>(source.data()),
                    static_cast<std::streamsize>(source.size()));

    return source;
}

void VulkanComputePipeline::Clear()
{
    if(!computePipeline) return;

    for(auto& setLayout : setLayouts)
    {
        vkDestroyDescriptorSetLayout(deviceVk, setLayout,
                                     VulkanHostAllocator::Functions());
    }
    vkDestroyPipelineLayout(deviceVk, pipelineLayout,
                            VulkanHostAllocator::Functions());

    vkDestroyPipeline(deviceVk, computePipeline,
                      VulkanHostAllocator::Functions());
}

VulkanComputePipeline::VulkanComputePipeline(VkDevice deviceVk)
    : deviceVk(deviceVk)
{}

VulkanComputePipeline::VulkanComputePipeline(VulkanComputePipeline&& other)
    : deviceVk(other.deviceVk)
    , computePipeline(std::exchange(other.computePipeline, nullptr))
    , pipelineLayout(std::exchange(other.pipelineLayout, nullptr))
    , setLayouts(std::move(other.setLayouts))
{}

VulkanComputePipeline& VulkanComputePipeline::operator=(VulkanComputePipeline&& other)
{
    assert(this != &other);
    Clear();
    deviceVk = other.deviceVk;
    computePipeline = std::exchange(other.computePipeline, nullptr);
    pipelineLayout = std::exchange(other.pipelineLayout, nullptr);
    setLayouts = std::move(other.setLayouts);
    return *this;
}

VulkanComputePipeline::~VulkanComputePipeline()
{
    Clear();
}

MRayError VulkanComputePipeline::Initialize(const Descriptor2DList<ShaderBindingInfo>& bindingInfoList,
                                            const std::string& shaderName,
                                            const std::string& execName,
                                            const std::string& entryPointName)
{
    auto sourceE = DevourFile(shaderName, execName);
    if(sourceE.has_error()) return sourceE.error();
    const auto& source = sourceE.value();
    // ================= //
    //   Shader Module   //
    // ================= //
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo smInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = source.size(),
        .pCode = reinterpret_cast<const uint32_t*>(source.data())
    };
    vkCreateShaderModule(deviceVk, &smInfo,
                         VulkanHostAllocator::Functions(),
                         &shaderModule);

    // ====================== //
    //  Discriptor Set Layout //
    // ====================== //
    Descriptor2DList<VkDescriptorSetLayoutBinding> bindings;
    for(const auto& bindingSet : bindingInfoList)
    {
        //
        bindings.push_back({});
        for(const ShaderBindingInfo& bindingInfo : bindingSet)
        {
            VkDescriptorSetLayoutBinding bInfo =
            {
                .binding = bindingInfo.bindingPoint,
                .descriptorType = bindingInfo.type,
                .descriptorCount = bindingInfo.elementCount,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr
            };
            bindings.back().push_back(bInfo);
        }
    }
    for(const auto& bindingSet : bindings)
    {
        VkDescriptorSetLayout setLayout;
        VkDescriptorSetLayoutCreateInfo dsInfo =
        {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = static_cast<uint32_t>(bindingSet.size()),
            .pBindings = bindingSet.data()
        };
        vkCreateDescriptorSetLayout(deviceVk, &dsInfo,
                                    VulkanHostAllocator::Functions(),
                                    &setLayout);
        setLayouts.push_back(setLayout);
    }

    // ================= //
    //  Pipeline Layout  //
    // ================= //
    VkPipelineLayoutCreateInfo pInfo =
    {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
        .pSetLayouts = setLayouts.data(),
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr
    };
    vkCreatePipelineLayout(deviceVk, &pInfo,
                           VulkanHostAllocator::Functions(),
                           &pipelineLayout);

    // ================= //
    //      Pipeline     //
    // ================= //
    VkComputePipelineCreateInfo cInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shaderModule,
            .pName = entryPointName.c_str()
        },
        .layout = pipelineLayout,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = 0
    };
    vkCreateComputePipelines(deviceVk, nullptr,
                             1, &cInfo,
                             VulkanHostAllocator::Functions(),
                             &computePipeline);

    // Afaik, we do not need to keep the module it is embedded to
    // pipeline
    vkDestroyShaderModule(deviceVk, shaderModule,
                          VulkanHostAllocator::Functions());

    return MRayError::OK;
}

typename VulkanComputePipeline::DescriptorSets
VulkanComputePipeline::GenerateDescriptorSets(VkDescriptorPool pool)
{
    DescriptorSets sets;
    for(const auto& setLayout : setLayouts)
    {
        VkDescriptorSet set;
        VkDescriptorSetAllocateInfo allocInfo =
        {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &setLayout,
        };
        vkAllocateDescriptorSets(deviceVk, &allocInfo, &set);
        sets.push_back(set);
    }
    return sets;
}

void VulkanComputePipeline::BindSetData(VkDescriptorSet descriptorSet,
                                        const DescriptorBindList<ShaderBindingData>& bindingDataList)
{
    // TODO: This function does not refer to any members of "VulkanComputePipeline"
    // change to free function.
    DescriptorBindList<VkWriteDescriptorSet> writeSets;
    for(const auto& bindingData : bindingDataList)
    {
        bool isBuffer = std::holds_alternative<VkDescriptorBufferInfo>(bindingData.dataInfo);
        bool isImage = std::holds_alternative<VkDescriptorImageInfo>(bindingData.dataInfo);

        VkWriteDescriptorSet writeInfo =
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = descriptorSet,
            .dstBinding = bindingData.index,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = bindingData.type,
            .pImageInfo = isImage ? &std::get<VkDescriptorImageInfo>(bindingData.dataInfo) : nullptr,
            .pBufferInfo = isBuffer ? &std::get<VkDescriptorBufferInfo>(bindingData.dataInfo) : nullptr,
            .pTexelBufferView = nullptr
        };
        writeSets.push_back(writeInfo);
    }

    vkUpdateDescriptorSets(deviceVk, static_cast<uint32_t>(writeSets.size()),
                            writeSets.data(), 0, nullptr);
}

void VulkanComputePipeline::BindSet(VkCommandBuffer cmd, uint32_t setIndex,
                                    VkDescriptorSet set)
{
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, setIndex, 1,
                            &set, 0, nullptr);
}

void VulkanComputePipeline::BindPipeline(VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
}