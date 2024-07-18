#include "TonemapStage.h"

#include <cassert>
#include <type_traits>
#include <utility>
#include <map>
#include <filesystem>

#include <Imgui/imgui.h>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_to_string.hpp>

#include "VulkanTypes.h"
#include "VulkanPipeline.h"

#include "Core/Error.hpp"
#include "Core/MemAlloc.h"

// Common descriptor set indices of tonemappers
static constexpr uint32_t TONEMAP_UNIFORM_BUFF_INDEX = 0;
static constexpr uint32_t GAMMA_UNIFORM_BUFF_INDEX = 1;
static constexpr uint32_t HDR_IMAGE_INDEX = 2;
static constexpr uint32_t SDR_IMAGE_INDEX = 3;
static constexpr uint32_t STAGING_BUFFER_INDEX = 4;

struct MinMaxAvgStageBuffer
{
    static_assert(sizeof(uint32_t) == sizeof(float));
    float minLum;
    float maxLum;
    float avgLum;
    float logAvgLum;
};

class TonemapperBase : public TonemapperI
{
    using DescriptorSets = typename VulkanComputePipeline::DescriptorSets;
    private:
    std::string_view        moduleName;
    MRayColorSpaceEnum      inColorSpace;
    VkColorSpaceKHR         outColorSpace;
    //
    VulkanComputePipeline   tonemapPipeline;
    VulkanComputePipeline   reduceTexPipeline;
    VulkanComputePipeline   reduceBuffPipeline;
    StagingBufferMemView    stagingBuffer = {};
    DescriptorSets          tonemapDescriptorSets;
    DescriptorSets          reduceTexDescriptorSets;
    DescriptorSets          reduceBuffDescriptorSets;
    size_t                  uniformBufferSize;
    size_t                  eotfBufferSize;

    protected:
    const VulkanSystemView* handlesVk = nullptr;
    UniformBufferMemView    uniformBuffer = {};
    UniformBufferMemView    eotfBuffer = {};

    public:
    // Constructors & Destructor
                TonemapperBase(std::string_view moduleName,
                               MRayColorSpaceEnum inColorSpace,
                               VkColorSpaceKHR outColorSpace,
                               size_t uniformBufferSize,
                               size_t eotfBufferSize);

    MRayError   Initialize(const VulkanSystemView& sys,
                           const std::string& execPath) override;
    void        RecordTonemap(VkCommandBuffer cmd,
                              const VulkanImage& hdrImg,
                              const VulkanImage& sdrImg) override;
    void        BindImages(const VulkanImage& hdrImg,
                           const VulkanImage& sdrImg) override;
    //
    size_t      UniformBufferSize() const override;
    void        SetUniformBufferView(const UniformBufferMemView& uniformBufferView) override;
    //
    size_t      StagingBufferSize(const Vector2ui& inputImgSize) const override;
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
    static constexpr VkColorSpaceKHR OUT_CS = VkColorSpaceKHR::VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    static constexpr MRayColorSpaceEnum IN_CS = MRayColorSpaceEnum::MR_ACES_CG;

    struct Uniforms
    {
        uint32_t enable = true;
        uint32_t doKeyAdjust = false;
        float    burnRatio = 1.0f;
        float    key = 0.18f;
    };

    class GUI : public GUITonemapperI
    {
        private:
        Uniforms&   opts;
        float&      gamma;
        bool        paramsChanged = true;

        public:
                    GUI(Uniforms&, float& gamma);
        bool        Render(bool& onOff) override;
        bool        IsParamsChanged() const;
    };

    private:
    Uniforms        tmParameters;
    float           gamma = 2.2f;
    GUI             gui;

    public:
    Tonemapper_Reinhard_AcesCG_To_SRGB();
    //
    GUITonemapperI* AcquireGUI() override;
    void            UpdateUniforms() override;
};

//============================//
//    EMPTY ACESCG -> HDR10   //
//============================//
class Tonemapper_Empty_AcesCG_To_HDR10 : public TonemapperBase
{
    static constexpr std::string_view MODULE_NAME = "Shaders/Tonemap-MRAY_TONEMAP_EMPTY_ACES_CG_TO_HDR10.spv";
    static constexpr VkColorSpaceKHR OUT_CS = VkColorSpaceKHR::VK_COLOR_SPACE_HDR10_ST2084_EXT;
    static constexpr MRayColorSpaceEnum IN_CS = MRayColorSpaceEnum::MR_ACES_CG;

    struct Uniforms
    {
        uint32_t enable = true;
    };

    class GUI : public GUITonemapperI
    {
        private:
        Uniforms&   opts;
        float&      displayNits;
        bool        paramsChanged = true;

        public:
                    GUI(Uniforms&, float& displayNits);
        bool        Render(bool& onOff) override;
        bool        IsParamsChanged() const;
    };

    private:
    Uniforms        tmParameters;
    float           peakBrightness = 400.0f;
    GUI             gui;

    public:
    Tonemapper_Empty_AcesCG_To_HDR10();
    //
    GUITonemapperI* AcquireGUI() override;
    void            UpdateUniforms() override;
};

TonemapperBase::TonemapperBase(std::string_view mName,
                               MRayColorSpaceEnum iColor,
                               VkColorSpaceKHR oColor,
                               size_t uboSize,
                               size_t eotfSize)
    : moduleName(mName)
    , inColorSpace(iColor)
    , outColorSpace(oColor)
    , uniformBufferSize(uboSize)
    , eotfBufferSize(eotfSize)
{}

MRayError TonemapperBase::Initialize(const VulkanSystemView& sys,
                                     const std::string& execPath)
{
    handlesVk = &sys;
    using namespace std::string_literals;
    const std::string tmEntryName = "KCTonemapImage"s;
    const std::string reduceTexEntryName = "KCFindAvgMaxLumTex"s;
    const std::string reduceBuffEntryName = "KCFindAvgMaxLumBuff"s;
    const std::string moduleNameString = std::string(moduleName);
    // ============= //
    //    Tonemap    //
    // ============= //
    MRayError e = tonemapPipeline.Initialize(handlesVk->deviceVk,
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

    // ============= //
    //   ReduceTex   //
    // ============= //
    e = reduceTexPipeline.Initialize(handlesVk->deviceVk,
    {
        {
            {TONEMAP_UNIFORM_BUFF_INDEX,    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {GAMMA_UNIFORM_BUFF_INDEX,      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {HDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
            {SDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            {STAGING_BUFFER_INDEX,          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
        }
    }, moduleNameString, execPath, reduceTexEntryName);
    if(e) return e;
    reduceTexDescriptorSets = reduceTexPipeline.GenerateDescriptorSets(handlesVk->mainDescPool);

    // ============= //
    //   ReduceBuff  //
    // ============= //
    e = reduceBuffPipeline.Initialize(handlesVk->deviceVk,
    {
        {
            {TONEMAP_UNIFORM_BUFF_INDEX,    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {GAMMA_UNIFORM_BUFF_INDEX,      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {HDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
            {SDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            {STAGING_BUFFER_INDEX,          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}
        }
    }, moduleNameString, execPath, reduceBuffEntryName, sizeof(uint32_t));
    if(e) return e;
    reduceBuffDescriptorSets = reduceBuffPipeline.GenerateDescriptorSets(handlesVk->mainDescPool);

    return MRayError::OK;
};

void TonemapperBase::RecordTonemap(VkCommandBuffer cmd,
                                  const VulkanImage& hdrImg,
                                  const VulkanImage& sdrImg)
{
    using MathFunctions::DivideUp;

    VkCommandBufferBeginInfo beginInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
        .pInheritanceInfo = nullptr,
    };
    vkBeginCommandBuffer(cmd, &beginInfo);

    std::array<VkImageMemoryBarrier, 2> imgBarrierInfo = {};
    // Change the HDR image state to read-only optimal
    imgBarrierInfo[0] =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = handlesVk->queueIndex,
        .dstQueueFamilyIndex = handlesVk->queueIndex,
        .image = hdrImg.Image(),
        .subresourceRange = VkImageSubresourceRange
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
    };
    // Change the SDR image writable
    imgBarrierInfo[1] = imgBarrierInfo[0];
    std::swap(imgBarrierInfo[1].srcAccessMask,
              imgBarrierInfo[1].dstAccessMask);
    std::swap(imgBarrierInfo[1].oldLayout,
              imgBarrierInfo[1].newLayout);
    imgBarrierInfo[1].image = sdrImg.Image();
    vkCmdPipelineBarrier(cmd,
                         (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                          VK_PIPELINE_STAGE_TRANSFER_BIT),
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr,
                         0, nullptr,
                         2, imgBarrierInfo.data());

    // ===================== //
    //  REDUCTION DISPATCHES //
    // ===================== //
    assert(hdrImg.Extent() == sdrImg.Extent());
    using MathFunctions::DivideUp;
    static constexpr uint32_t TPB_1D = VulkanComputePipeline::TPB_1D;
    uint32_t pixelCount = hdrImg.Extent().Multiply();
    for(uint32_t totalPix = pixelCount; totalPix != 1;
        totalPix = DivideUp(totalPix, TPB_1D))
    {
        assert(hdrImg.Extent() == sdrImg.Extent());
        uint32_t reduceGroupSize = DivideUp(totalPix, TPB_1D);
        if(totalPix == pixelCount)
        {
            reduceTexPipeline.BindSet(cmd, 0, reduceTexDescriptorSets[0]);
            reduceTexPipeline.BindPipeline(cmd);
        }
        else
        {
            reduceBuffPipeline.BindSet(cmd, 0, reduceBuffDescriptorSets[0]);
            reduceBuffPipeline.BindPipeline(cmd);
            reduceBuffPipeline.PushConstant(cmd, pixelCount);
        }
        vkCmdDispatch(cmd, reduceGroupSize, 1, 1);

        // Barrier between successive dispatches
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
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr,
                             1, &buffMemBarrierInfo,
                             0, nullptr);
    }

    // ============= //
    //    DISPATCH   //
    // ============= //
    Vector2ui tmGroupSize = DivideUp(sdrImg.Extent(),
                                     VulkanComputePipeline::TPB_2D);
    tonemapPipeline.BindPipeline(cmd);
    tonemapPipeline.BindSet(cmd, 0, tonemapDescriptorSets[0]);
    vkCmdDispatch(cmd, tmGroupSize[0], tmGroupSize[1], 1);

    // Change the SDR image to fragment-shader readable
    imgBarrierInfo[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imgBarrierInfo[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imgBarrierInfo[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarrierInfo[0].newLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
    imgBarrierInfo[0].image = sdrImg.Image();

    // Change the HDR image state to writable for next accum event
    imgBarrierInfo[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imgBarrierInfo[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imgBarrierInfo[1].oldLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
    imgBarrierInfo[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarrierInfo[1].image = hdrImg.Image();

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT), 0,
                         0, nullptr,
                         0, nullptr,
                         2, imgBarrierInfo.data());

    vkEndCommandBuffer(cmd);
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
    reduceTexPipeline.BindSetData(reduceTexDescriptorSets[0], bindList);
    reduceBuffPipeline.BindSetData(reduceBuffDescriptorSets[0], bindList);
    tonemapPipeline.BindSetData(tonemapDescriptorSets[0], bindList);
}

size_t TonemapperBase::UniformBufferSize() const
{
    // Design fails here we do not have two different uniform buffer
    // requesters. User our giga alignment to be sure
    // We need to change this later
    using MemAlloc::DefaultSystemAlignment;
    using MathFunctions::NextMultiple;
    size_t uniformSize = NextMultiple(uniformBufferSize,
                                      DefaultSystemAlignment());
    size_t gammaSize = NextMultiple(eotfBufferSize,
                                    DefaultSystemAlignment());
    return uniformSize + gammaSize;
}

void TonemapperBase::SetUniformBufferView(const UniformBufferMemView& ubo)
{
    using MemAlloc::DefaultSystemAlignment;
    using MathFunctions::NextMultiple;
    size_t uSize = NextMultiple(uniformBufferSize,
                                DefaultSystemAlignment());
    size_t gSize = NextMultiple(eotfBufferSize,
                                DefaultSystemAlignment());
    uniformBuffer = ubo;
    uniformBuffer.size = uSize;
    eotfBuffer = ubo;
    eotfBuffer.size = gSize;
    eotfBuffer.offset += uSize;

    DescriptorBindList<ShaderBindingData> bindList =
    {
        {
            TONEMAP_UNIFORM_BUFF_INDEX,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VkDescriptorBufferInfo
            {
                .buffer = uniformBuffer.bufferHandle,
                .offset = uniformBuffer.offset,
                .range = uniformBuffer.size
            },
        },
        {
            GAMMA_UNIFORM_BUFF_INDEX,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VkDescriptorBufferInfo
            {
                .buffer = eotfBuffer.bufferHandle,
                .offset = eotfBuffer.offset,
                .range = eotfBuffer.size
            },
        }
    };
    tonemapPipeline.BindSetData(tonemapDescriptorSets[0], bindList);
    reduceTexPipeline.BindSetData(reduceTexDescriptorSets[0], bindList);
    reduceBuffPipeline.BindSetData(reduceBuffDescriptorSets[0], bindList);
}

size_t TonemapperBase::StagingBufferSize(const Vector2ui& inputImgSize) const
{
    using MathFunctions::DivideUp;

    uint32_t linearSize = inputImgSize.Multiply();
    uint32_t transientElementCount = DivideUp(linearSize,
                                              VulkanComputePipeline::TPB_1D);
    return transientElementCount * sizeof(MinMaxAvgStageBuffer);
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
    reduceTexPipeline.BindSetData(reduceTexDescriptorSets[0], bindList);
    reduceBuffPipeline.BindSetData(reduceBuffDescriptorSets[0], bindList);
}

MRayColorSpaceEnum TonemapperBase::InputColorspace() const
{
    return inColorSpace;
}

VkColorSpaceKHR TonemapperBase::OutputColorspace() const
{
    return outColorSpace;
}

Tonemapper_Reinhard_AcesCG_To_SRGB::GUI::GUI(Uniforms& ubo, float& gammaIn)
    : opts(ubo)
    , gamma(gammaIn)
{}

bool Tonemapper_Reinhard_AcesCG_To_SRGB::GUI::Render(bool& onOff)
{
    paramsChanged = false;
    if(ImGui::Begin("AcesCG -> sRGB", &onOff,
                    ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoScrollbar |
                    ImGuiWindowFlags_AlwaysAutoResize))
    {
        // Enable box
        bool bEnable = static_cast<uint32_t>(opts.enable);
        paramsChanged |= ImGui::Checkbox("Enable", &bEnable);
        opts.enable = static_cast<uint32_t>(bEnable);

        // Key Adjust enable
        bool bKeyAdj = static_cast<uint32_t>(opts.doKeyAdjust);
        paramsChanged |= ImGui::Checkbox("Key Adjust", &bKeyAdj);
        opts.doKeyAdjust = static_cast<uint32_t>(bKeyAdj);
        // Key
        ImGui::Text("Key       ");
        ImGui::SameLine();
        //paramsChanged |= ImGui::InputFloat("##Key", &opts.key, 0.01f, 0.30f);
        paramsChanged |= ImGui::DragFloat("##Key", &opts.key,
                                          0.005f, 0.005f, 50.0f,
                                          "%.3f");

        // Burn Ratio
        ImGui::Text("Burn Ratio");
        ImGui::SameLine();
        paramsChanged |= ImGui::SliderFloat("##Burn", &opts.burnRatio, 0.0f, 2.0f);
        // Gamma
        ImGui::Text("Gamma     ");
        ImGui::SameLine();
        paramsChanged |= ImGui::SliderFloat("##GammaSlider", &gamma, 0.1f, 4.0f, "%0.3f");
    }
    ImGui::End();
    return paramsChanged;
}

bool Tonemapper_Reinhard_AcesCG_To_SRGB::GUI::IsParamsChanged() const
{
    return paramsChanged;
}

Tonemapper_Reinhard_AcesCG_To_SRGB::Tonemapper_Reinhard_AcesCG_To_SRGB()
    : TonemapperBase(MODULE_NAME, IN_CS, OUT_CS, sizeof(Uniforms), sizeof(float))
    , gui(tmParameters, gamma)
{}

GUITonemapperI* Tonemapper_Reinhard_AcesCG_To_SRGB::AcquireGUI()
{
    return &gui;
}

void Tonemapper_Reinhard_AcesCG_To_SRGB::UpdateUniforms()
{
    if(gui.IsParamsChanged())
    {
        std::memcpy(uniformBuffer.hostPtr + uniformBuffer.offset,
                    &tmParameters, sizeof(Uniforms));
        std::memcpy(eotfBuffer.hostPtr + eotfBuffer.offset,
                    &gamma, sizeof(float));
        uniformBuffer.FlushRange(handlesVk->deviceVk);
    }
}

Tonemapper_Empty_AcesCG_To_HDR10::GUI::GUI(Uniforms& ubo, float& nitsIn)
    : opts(ubo)
    , displayNits(nitsIn)
{}

bool Tonemapper_Empty_AcesCG_To_HDR10::GUI::Render(bool& onOff)
{
    paramsChanged = false;
    if(ImGui::Begin("AcesCG -> HDR10", &onOff,
                    ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoScrollbar |
                    ImGuiWindowFlags_AlwaysAutoResize))
    {
        // Enable box
        bool bEnable = static_cast<uint32_t>(opts.enable);
        paramsChanged |= ImGui::Checkbox("Enable", &bEnable);
        opts.enable = static_cast<uint32_t>(bEnable);

        ImGui::Text("Display Brightness");
        ImGui::SameLine();
        paramsChanged |= ImGui::DragFloat("##Brightness", &displayNits,
                                          4.0f, 100.0f, 2000.0f,
                                          "%.1f nits");
    }
    ImGui::End();
    return paramsChanged;
}

bool Tonemapper_Empty_AcesCG_To_HDR10::GUI::IsParamsChanged() const
{
    return paramsChanged;
}

Tonemapper_Empty_AcesCG_To_HDR10::Tonemapper_Empty_AcesCG_To_HDR10()
    : TonemapperBase(MODULE_NAME, IN_CS, OUT_CS, sizeof(Uniforms), sizeof(float))
    , gui(tmParameters, peakBrightness)
{}

GUITonemapperI* Tonemapper_Empty_AcesCG_To_HDR10::AcquireGUI()
{
    return &gui;
}

void Tonemapper_Empty_AcesCG_To_HDR10::UpdateUniforms()
{
    if(gui.IsParamsChanged())
    {
        std::memcpy(uniformBuffer.hostPtr + uniformBuffer.offset,
                    &tmParameters, sizeof(Uniforms));
        std::memcpy(eotfBuffer.hostPtr + eotfBuffer.offset,
                    &peakBrightness, sizeof(float));
        uniformBuffer.FlushRange(handlesVk->deviceVk);
    }
}

//======================//
//    TONEMAP STAGE     //
//======================//
MRayError TonemapStage::Initialize(const VulkanSystemView& view,
                                   const std::string& execPath)
{
    handlesVk = &view;
    tmCommand = VulkanCommandBuffer(*handlesVk);

    using namespace std::string_view_literals;
    namespace fs = std::filesystem;

    std::unique_ptr<TonemapperI> tm0 = std::make_unique<Tonemapper_Empty_AcesCG_To_HDR10>();
    std::unique_ptr<TonemapperI> tm1 = std::make_unique<Tonemapper_Reinhard_AcesCG_To_SRGB>();

    MRayError e = MRayError::OK;
    e = tm0->Initialize(*handlesVk, execPath);
    if(e) return e;
    e = tm1->Initialize(*handlesVk, execPath);
    if(e) return e;

    tonemappers.try_emplace({tm0->InputColorspace(), tm0->OutputColorspace()},
                            std::move(tm0));
    tonemappers.try_emplace({tm1->InputColorspace(), tm1->OutputColorspace()},
                            std::move(tm1));
    return MRayError::OK;
}

void TonemapStage::ChangeImage(const VulkanImage* hdrImageIn,
                               const VulkanImage* sdrImageIn)
{
    size_t stagingBufferSize = 0;
    Vector2ui extents = hdrImageIn->Extent();
    for(auto& [_, tm] : tonemappers)
    {
        stagingBufferSize = std::max(stagingBufferSize,
                                     tm->StagingBufferSize(extents));
    }

    // TODO: Should we shrink to fit or use large memory?
    // Currently we do not reallocate when buffer is already large
    if(stagingBuffer.Size() < stagingBufferSize)
    {
        auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        stagingBuffer = VulkanBuffer(*handlesVk, usage, stagingBufferSize);

        auto deviceAllocator = VulkanDeviceAllocator::Instance();
        stagingMemory = deviceAllocator.AllocateMultiObject(std::tie(stagingBuffer),
                                                            VulkanDeviceAllocator::DEVICE);
        for(auto& [_, tm] : tonemappers)
        {
            StagingBufferMemView buffView =
            {
                .bufferHandle = stagingBuffer.Buffer(),
                .offset = 0,
                .size = tm->StagingBufferSize(extents)
            };
            tm->SetStagingBufferView(buffView);
        }
    }

    assert(hdrImageIn != nullptr);
    assert(sdrImageIn != nullptr);
    hdrImage = hdrImageIn;
    sdrImage = sdrImageIn;
    for(auto& [_, tm] : tonemappers)
    {
        tm->BindImages(*hdrImageIn, *sdrImageIn);
    }
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
    currentTonemapper->RecordTonemap(tmCommand, *hdrImage, *sdrImage);
    return currentTonemapper->AcquireGUI();
}

void TonemapStage::IssueTonemap(const VulkanTimelineSemaphore& imgSem)
{
    currentTonemapper->UpdateUniforms();

    auto allStages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo waitSemaphore = imgSem.WaitInfo(allStages);
    VkSemaphoreSubmitInfo signalSemaphores = imgSem.SignalInfo(allStages, 1);
    VkCommandBufferSubmitInfo commandSubmitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = tmCommand,
        .deviceMask = 0
    };
    VkSubmitInfo2 submitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &waitSemaphore,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandSubmitInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalSemaphores
    };
    vkQueueSubmit2(handlesVk->mainQueueVk, 1, &submitInfo, nullptr);
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

size_t TonemapStage::UsedGPUMemBytes() const
{
    return stagingMemory.SizeBytes();
}