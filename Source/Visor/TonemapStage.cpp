#include "TonemapStage.h"

#include <cassert>
#include <type_traits>
#include <utility>
#include <map>
#include <filesystem>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_to_string.hpp>

#include "VulkanTypes.h"
#include "VulkanPipeline.h"

#include "Core/Error.hpp"

// Common descriptor set indices of tonemappers
static constexpr uint32_t TONEMAP_UNIFORM_BUFF_INDEX = 0;
static constexpr uint32_t GAMMA_UNIFORM_BUFF_INDEX = 1;
static constexpr uint32_t HDR_IMAGE_INDEX = 2;
static constexpr uint32_t SDR_IMAGE_INDEX = 3;
static constexpr uint32_t STAGING_BUFFER_INDEX = 4;

struct MaxAvgStageBuffer
{
    static_assert(sizeof(uint32_t) == sizeof(float));
    uint32_t maxLum;
    uint32_t avgLum;
};

struct GammaBuffer
{
    float gamma;
};

class TonemapperBase : public TonemapperI
{
    using DescriptorSets = typename VulkanComputePipeline::DescriptorSets;
    private:
    std::string_view        moduleName;
    VkColorSpaceKHR         outColorSpace;
    MRayColorSpaceEnum      inColorSpace;
    //
    VulkanComputePipeline   tonemapPipeline;
    VulkanComputePipeline   reducePipeline;
    StagingBufferMemView    stagingBuffer = {};
    DescriptorSets          tonemapDescriptorSets;
    DescriptorSets          reduceDescriptorSets;
    float                   outGammaValue;
    size_t                  uniformBufferSize;

    protected:
    UniformBufferMemView    uniformBuffer = {};
    const VulkanSystemView* handlesVk = nullptr;
    virtual void            UpdateUniformData() = 0;

    public:
    // Constructors & Destructor
                 TonemapperBase(const VulkanSystemView& sys,
                                std::string_view moduleName,
                                MRayColorSpaceEnum inColorSpace,
                                VkColorSpaceKHR outColorSpace,
                                size_t uniformBufferSize);

    MRayError   Initialize(const std::string& execPath) override;
    void        TonemapImage(VkCommandBuffer cmd,
                             const VulkanImage& hdrImg,
                             const VulkanImage& sdrImg) override;
    void        BindImages(const VulkanImage& hdrImg,
                           const VulkanImage& sdrImg) override;
    //
    size_t      UniformBufferSize() const override;
    void        SetUniformBufferView(const UniformBufferMemView& uniformBufferView) override;
    //
    size_t      StagingBufferSize() const override;
    void        SetStagingBufferView(const StagingBufferMemView& stagingBufferView) override;
    //
    MRayColorSpaceEnum  InputColorspace() const override;
    VkColorSpaceKHR     OutputColorspace() const override;
};

//============================//
//   TONEMAP ACESCG -> SRGB   //
//============================//
class Tonemapper_Reinhard_AcesCG_To_SRGB : public TonemapperBase
{
    static constexpr std::string_view MODULE_NAME = "Shaders/Tonemap-MRAY_TONEMAP_REINHARD_ACES_CG_TO_SRGB.spv";
    static constexpr VkColorSpaceKHR OUT_CS = VkColorSpaceKHR::VK_COLOR_SPACE_HDR10_ST2084_EXT;
    static constexpr MRayColorSpaceEnum IN_CS = MRayColorSpaceEnum::MR_ACES_CG;

    struct Uniforms
    {
        uint32_t doKeyAdjust = false;
        float    burnRatio = 1.0f;
        float    key = 0.18f;
    };

    class GUI : public GUITonemapperI
    {
        private:
        Uniforms&   tmParameters;
        bool        paramsChanged;

        public:
                    GUI(Uniforms&);
        void        Render() override;
        bool        IsParamsChanged() const;
    };

    private:
    Uniforms        tmParameters;
    GUI             gui;

    protected:
    void            UpdateUniformData() override;

    public:
    // Constructors & Destructor
                    Tonemapper_Reinhard_AcesCG_To_SRGB(const VulkanSystemView& sys);
    //
    GUITonemapperI* AcquireGUI() override;
};

//============================//
//    EMPTY ACESCG -> HDR10   //
//============================//
class Tonemapper_Empty_AcesCG_To_HDR10 : public TonemapperBase
{
    static constexpr std::string_view MODULE_NAME = "Shaders/Tonemap-MRAY_TONEMAP_EMPTY_ACES_CG_TO_HDR10.spv";
    static constexpr VkColorSpaceKHR OUT_CS = VkColorSpaceKHR::VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    static constexpr MRayColorSpaceEnum IN_CS = MRayColorSpaceEnum::MR_ACES_CG;

    private:
    protected:
    void            UpdateUniformData() override {};

    public:
    // Constructors & Destructor
                    Tonemapper_Empty_AcesCG_To_HDR10(const VulkanSystemView& sys);
    //
    GUITonemapperI* AcquireGUI() override;
};

TonemapperBase::TonemapperBase(const VulkanSystemView& sys,
                               std::string_view mName,
                               MRayColorSpaceEnum iColor,
                               VkColorSpaceKHR oColor,
                               size_t uboSize)
    : moduleName(mName)
    , inColorSpace(iColor)
    , outColorSpace(oColor)
    , handlesVk(&sys)
    , tonemapPipeline(sys.deviceVk)
    , reducePipeline(sys.deviceVk)
    , outGammaValue(2.2f)
    , uniformBufferSize(uboSize)
{}

MRayError TonemapperBase::Initialize(const std::string& execPath)
{
    using namespace std::string_literals;
    static const std::string tmEntryName = "KCTonemapImage"s;
    static const std::string reduceEntryName = "KCFindAvgMaxLum"s;
    static const std::string moduleNameString = std::string(moduleName);
    MRayError e = tonemapPipeline.Initialize(
    {
        {
            {TONEMAP_UNIFORM_BUFF_INDEX,    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {GAMMA_UNIFORM_BUFF_INDEX,      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {HDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
            {SDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            {STAGING_BUFFER_INDEX,          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
        }
    }, moduleNameString, execPath, tmEntryName);

    if(e) return e;
    tonemapDescriptorSets = tonemapPipeline.GenerateDescriptorSets(handlesVk->mainDescPool);
    using namespace std::string_literals;
    e = reducePipeline.Initialize(
    {
        {
            {TONEMAP_UNIFORM_BUFF_INDEX,    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {GAMMA_UNIFORM_BUFF_INDEX,      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {HDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
            {SDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            {STAGING_BUFFER_INDEX,          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
        }
    }, moduleNameString, execPath, reduceEntryName);
    if(e) return e;
    reduceDescriptorSets = reducePipeline.GenerateDescriptorSets(handlesVk->mainDescPool);
    return MRayError::OK;
};

void TonemapperBase::TonemapImage(VkCommandBuffer cmd,
                                  const VulkanImage& hdrImg,
                                  const VulkanImage& sdrImg)
{
    UpdateUniformData();

    assert(hdrImg.Extent() == sdrImg.Extent());
    Vector2ui totalPix = sdrImg.Extent();
    Vector2ui TPB = Vector2ui(VulkanComputePipeline::TPB_2D_X,
                              VulkanComputePipeline::TPB_2D_Y);
    Vector2ui groupSize = MathFunctions::DivideUp(totalPix, TPB);

    // ============= //
    //    DISPATCH   //
    // ============= //
    reducePipeline.BindSet(cmd, 0, reduceDescriptorSets[0]);
    reducePipeline.BindPipeline(cmd);
    vkCmdDispatch(cmd, groupSize[0], groupSize[1], 1);

    VkBufferMemoryBarrier buffMemBarrierInfo =
    {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = handlesVk->queueIndex,
        .dstQueueFamilyIndex = handlesVk->queueIndex,
        .buffer = stagingBuffer.bufferHandle,
        .offset = stagingBuffer.offset,
        .size = stagingBuffer.size
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr,
                         1, &buffMemBarrierInfo,
                         0, nullptr);

    // ============= //
    //    DISPATCH   //
    // ============= //
    tonemapPipeline.BindPipeline(cmd);
    tonemapPipeline.BindSet(cmd, 0, tonemapDescriptorSets[0]);
    vkCmdDispatch(cmd, groupSize[0], groupSize[1], 1);

    VkImageMemoryBarrier imgMemBarrierInfo =
    {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = handlesVk->queueIndex,
        .dstQueueFamilyIndex = handlesVk->queueIndex,
        .image = sdrImg.Image(),
        .subresourceRange = VkImageSubresourceRange
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr,
                         0, nullptr,
                         1, &imgMemBarrierInfo);
}

void TonemapperBase::BindImages(const VulkanImage& hdrImg,
                                const VulkanImage& sdrImg)
{
    DescriptorBindList<ShaderBindingData> bindList =
    {
        {
            HDR_IMAGE_INDEX,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VkDescriptorImageInfo
            {
                .sampler = hdrImg.Sampler(),
                .imageView = hdrImg.View(),
                .imageLayout = VkImageLayout::VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL
            }
        },
        {
            SDR_IMAGE_INDEX,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VkDescriptorImageInfo
            {
                .sampler = sdrImg.Sampler(),
                .imageView = sdrImg.View(),
                .imageLayout = VkImageLayout::VK_IMAGE_LAYOUT_GENERAL
            }
        }
    };
    reducePipeline.BindSetData(reduceDescriptorSets[0], bindList);
    tonemapPipeline.BindSetData(tonemapDescriptorSets[0], bindList);
}

size_t TonemapperBase::UniformBufferSize() const
{
    // TODO: Alignment
    return sizeof(uniformBufferSize) + sizeof(GammaBuffer);
}

void TonemapperBase::SetUniformBufferView(const UniformBufferMemView& ubo)
{
    uniformBuffer = ubo;
    // TODO: ALIGNMENT!!!!
    DescriptorBindList<ShaderBindingData> bindList =
    {
        {
            TONEMAP_UNIFORM_BUFF_INDEX,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VkDescriptorBufferInfo
            {
                .buffer = uniformBuffer.bufferHandle,
                .offset = uniformBuffer.offset,
                .range = sizeof(uniformBufferSize)
            },
        },
        {
            GAMMA_UNIFORM_BUFF_INDEX,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VkDescriptorBufferInfo
            {
                .buffer = uniformBuffer.bufferHandle,
                .offset = uniformBuffer.offset + sizeof(uniformBufferSize),
                .range = sizeof(GammaBuffer)
            },
        }
    };
    tonemapPipeline.BindSetData(tonemapDescriptorSets[0], bindList);
    reducePipeline.BindSetData(tonemapDescriptorSets[0], bindList);
}

size_t TonemapperBase::StagingBufferSize() const
{
    return sizeof(MaxAvgStageBuffer);
}

void TonemapperBase::SetStagingBufferView(const StagingBufferMemView& ssbo)
{
    stagingBuffer = ssbo;
    DescriptorBindList<ShaderBindingData> bindList =
    {
        {
            STAGING_BUFFER_INDEX,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VkDescriptorBufferInfo
            {
                .buffer = stagingBuffer.bufferHandle,
                .offset = stagingBuffer.offset,
                .range = stagingBuffer.size
            },
        },
    };
    tonemapPipeline.BindSetData(tonemapDescriptorSets[0], bindList);
    reducePipeline.BindSetData(tonemapDescriptorSets[0], bindList);
}

MRayColorSpaceEnum TonemapperBase::InputColorspace() const
{
    return inColorSpace;
}

VkColorSpaceKHR TonemapperBase::OutputColorspace() const
{
    return outColorSpace;
}

Tonemapper_Reinhard_AcesCG_To_SRGB::GUI::GUI(Uniforms& ubo)
    : tmParameters(ubo)
{}

void Tonemapper_Reinhard_AcesCG_To_SRGB::GUI::Render()
{
    paramsChanged = false;
    // GO Full imgui here
}

bool Tonemapper_Reinhard_AcesCG_To_SRGB::GUI::IsParamsChanged() const
{
    return paramsChanged;
}

void Tonemapper_Reinhard_AcesCG_To_SRGB::UpdateUniformData()
{
    if(gui.IsParamsChanged())
    {
        std::memcpy(uniformBuffer.hostPtr, &tmParameters, sizeof(Uniforms));
        uniformBuffer.FlushRange(handlesVk->deviceVk);
    }
}

Tonemapper_Reinhard_AcesCG_To_SRGB::Tonemapper_Reinhard_AcesCG_To_SRGB(const VulkanSystemView& sys)
    : TonemapperBase(sys, MODULE_NAME, IN_CS, OUT_CS, sizeof(Uniforms))
    , gui(tmParameters)
{}

GUITonemapperI* Tonemapper_Reinhard_AcesCG_To_SRGB::AcquireGUI()
{
    return &gui;
}

Tonemapper_Empty_AcesCG_To_HDR10::Tonemapper_Empty_AcesCG_To_HDR10(const VulkanSystemView& sys)
    : TonemapperBase(sys, MODULE_NAME, IN_CS, OUT_CS, sizeof(EmptyType))
{}

GUITonemapperI* Tonemapper_Empty_AcesCG_To_HDR10::AcquireGUI()
{
    return nullptr;
}

//======================//
//    TONEMAP STAGE     //
//======================//
TonemapStage::TonemapStage(const VulkanSystemView& view)
    : handlesVk(&view)
    , stagingBuffer(view)
    , memory(view.deviceVk)
{}

MRayError TonemapStage::Initialize(const std::string& execPath)
{
    using namespace std::string_view_literals;
    namespace fs = std::filesystem;

    std::unique_ptr<TonemapperI> tm0 = std::make_unique<Tonemapper_Empty_AcesCG_To_HDR10>(*handlesVk);
    std::unique_ptr<TonemapperI> tm1 = std::make_unique<Tonemapper_Reinhard_AcesCG_To_SRGB>(*handlesVk);

    MRayError e = MRayError::OK;
    e = tm0->Initialize(execPath);
    if(e) return e;
    e = tm1->Initialize(execPath);
    if(e) return e;

    size_t stagingBufferSize = std::max(tm0->StagingBufferSize(),
                                        tm1->StagingBufferSize());
    stagingBuffer = VulkanBuffer(*handlesVk, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 stagingBufferSize);

    auto deviceAllocator = VulkanDeviceAllocator::Instance();
    memory = deviceAllocator.AllocateMultiObject(std::tie(stagingBuffer),
                                                 VulkanDeviceAllocator::DEVICE);
    tm0->SetStagingBufferView(StagingBufferMemView
                              {
                                  .bufferHandle = stagingBuffer.Buffer(),
                                  .offset = 0,
                                  .size = tm0->StagingBufferSize()
                              });
    tm1->SetStagingBufferView(StagingBufferMemView
                              {
                                  .bufferHandle = stagingBuffer.Buffer(),
                                  .offset = 0,
                                  .size = tm1->StagingBufferSize()
                              });


    tonemappers.try_emplace({tm0->InputColorspace(), tm0->OutputColorspace()},
                            std::move(tm0));
    tonemappers.try_emplace({tm1->InputColorspace(), tm1->OutputColorspace()},
                            std::move(tm1));
    return MRayError::OK;
}

void TonemapStage::ChangeImage(const VulkanImage* hdrImageIn,
                               const VulkanImage* sdrImageIn)
{
    hdrImage = hdrImageIn;
    sdrImage = sdrImageIn;
}

Expected<GUITonemapperI*>
TonemapStage::ChangeTonemapper(MRayColorSpaceEnum renderColorSpace,
                               VkColorSpaceKHR swapchainColorSpace)
{
    auto loc = tonemappers.find({renderColorSpace, swapchainColorSpace});
    if(loc == tonemappers.end())
        return MRayError("[Visor]: Unable to find appropriate tonemapper \"{}->{}\"",
                         MRayColorSpaceStringifier::ToString(renderColorSpace),
                         vk::to_string(vk::ColorSpaceKHR(swapchainColorSpace)));

    currentTonemapper = loc->second.get();
    return currentTonemapper->AcquireGUI();
}

void TonemapStage::IssueTonemap(VkCommandBuffer cmd)
{
    currentTonemapper->TonemapImage(cmd, *hdrImage, *sdrImage);
}

size_t TonemapStage::UniformBufferSize() const
{
    size_t maxSize = std::transform_reduce(tonemappers.cbegin(), tonemappers.cend(),
                                           size_t(0),
    [](size_t l, size_t r) -> size_t
    {
        return std::max(l, r);
    },
    [](const auto& kv) ->size_t
    {
        return kv.second->UniformBufferSize();
    });
    return maxSize;
}

void TonemapStage::SetUniformBufferView(const UniformBufferMemView& uniformBufferPtr)
{
    std::for_each(tonemappers.begin(), tonemappers.end(),
                  [&uniformBufferPtr](auto& kv)
    {
        kv.second->SetUniformBufferView(uniformBufferPtr);
    });
}