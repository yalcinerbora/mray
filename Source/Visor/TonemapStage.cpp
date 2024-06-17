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
static constexpr uint32_t HDR_IMAGE_INDEX = 0;
static constexpr uint32_t SDR_IMAGE_INDEX = 1;
static constexpr uint32_t TONEMAP_UNIFORM_INDEX = 2;
static constexpr uint32_t STAGING_BUFFER_INDEX = 3;
static constexpr uint32_t GAMMA_UNIFORM_BUFFER_INDEX = 4;

struct MaxAvgStageBuffer
{
    static_assert(sizeof(uint32_t) == sizeof(float));
    uint32_t maxLum;
    uint32_t avgLum;
};

MRayError GenericCreatePipeline(VulkanComputePipeline& tonemapPipeline,
                                VulkanComputePipeline& reducePipeline,
                                typename VulkanComputePipeline::DescriptorSets& tonemapDescriptorSets,
                                typename VulkanComputePipeline::DescriptorSets& reduceDescriptorSets,
                                const std::string& moduleName,
                                const std::string& execPath,
                                const std::string& tmEntryName,
                                const std::string& reduceEntryName,
                                const VulkanSystemView& handlesVk)
{
    MRayError e = tonemapPipeline.Initialize(
    {
        {
            {HDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            {SDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            {TONEMAP_UNIFORM_INDEX,         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
            {STAGING_BUFFER_INDEX,          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {GAMMA_UNIFORM_BUFFER_INDEX,    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}
        }
    },
    moduleName, execPath, tmEntryName);
    if(e) return e;
    tonemapDescriptorSets = tonemapPipeline.GenerateDescriptorSets(handlesVk.mainDescPool);

    using namespace std::string_literals;
    e = reducePipeline.Initialize(
    {
        {
            {HDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            {SDR_IMAGE_INDEX,               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            {TONEMAP_UNIFORM_INDEX,         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
            {STAGING_BUFFER_INDEX,          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {GAMMA_UNIFORM_BUFFER_INDEX,    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}
        }
    },
    moduleName, execPath, reduceEntryName);
    if(e) return e;
    reduceDescriptorSets = reducePipeline.GenerateDescriptorSets(handlesVk.mainDescPool);
    return MRayError::OK;
}

class Tonemapper_Reinhard_AcesCG_To_SRGB : public TonemapperI
{
    using DescriptorSets = typename VulkanComputePipeline::DescriptorSets;
    static constexpr std::string_view MODULE_NAME = "Shaders/Tonemap-MRAY_TONEMAP_REINHARD_ACES_CG_TO_SRGB.spv";
    static constexpr VkColorSpaceKHR OUT_CS = VkColorSpaceKHR::VK_COLOR_SPACE_HDR10_ST2084_EXT;
    static constexpr MRayColorSpaceEnum IN_CS = MRayColorSpaceEnum::MR_ACES_CG;

    public:
    struct UniformBuffer
    {
        uint32_t doKeyAdjust;
        float    burnRatio;
        float    key;
    };

    private:
    const VulkanSystemView* handlesVk = nullptr;
    StagingBufferMemView    stagingBuffer = {};
    UniformBufferMemView    uniformBuffer = {};
    VulkanComputePipeline   tonemapPipeline;
    VulkanComputePipeline   reducePipeline;
    UniformBuffer           tmParameters;
    float                   outGammaVal;
    DescriptorSets          tonemapDescriptorSets;
    DescriptorSets          reduceDescriptorSets;

    public:
    // Constructors & Destructor
    Tonemapper_Reinhard_AcesCG_To_SRGB(const VulkanSystemView& sys);

    //
    MRayError       Initialize(const std::string& execPath) override;
    GUITonemapperI* AcquireGUI() override;
    void            TonemapImage(VkCommandBuffer cmd,
                                 const VulkanImage& hdrImg,
                                 const VulkanImage& sdrImg) override;
    //
    MRayColorSpaceEnum  InputColorspace() const override;
    VkColorSpaceKHR     OutputColorspace() const override;
    //
    size_t  UniformBufferSize() const override;
    void    SetUniformBufferView(const UniformBufferMemView& uniformBufferView) override;
    //
    size_t  StagingBufferSize() const override;
    void    SetStagingBufferView(const StagingBufferMemView& stagingBufferView) override;
};

class Tonemapper_Empty_AcesCG_To_HDR10 : public TonemapperI
{
    using DescriptorSets = typename VulkanComputePipeline::DescriptorSets;
    static constexpr std::string_view MODULE_NAME = "Shaders/Tonemap-MRAY_TONEMAP_EMPTY_ACES_CG_TO_HDR10.spv";
    static constexpr VkColorSpaceKHR OUT_CS = VkColorSpaceKHR::VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    static constexpr MRayColorSpaceEnum IN_CS = MRayColorSpaceEnum::MR_ACES_CG;

    private:
    const VulkanSystemView* handlesVk = nullptr;
    StagingBufferMemView    stagingBuffer = {};
    VulkanComputePipeline   tonemapPipeline;
    VulkanComputePipeline   reducePipeline;
    float                   outGammaVal;
    DescriptorSets          tonemapDescriptorSets;
    DescriptorSets          reduceDescriptorSets;

    public:
    // Constructors & Destructor
    Tonemapper_Empty_AcesCG_To_HDR10(const VulkanSystemView& sys);

    //
    MRayError       Initialize(const std::string& execPath) override;
    GUITonemapperI* AcquireGUI() override;
    void            TonemapImage(VkCommandBuffer cmd,
                                 const VulkanImage& hdrImg,
                                 const VulkanImage& sdrImg) override;
    //
    MRayColorSpaceEnum  InputColorspace() const override;
    VkColorSpaceKHR     OutputColorspace() const override;
    //
    size_t  UniformBufferSize() const override;
    void    SetUniformBufferView(const UniformBufferMemView& uniformBufferView) override;
    //
    size_t  StagingBufferSize() const override;
    void    SetStagingBufferView(const StagingBufferMemView& stagingBufferView) override;
};

//============================//
//   TONEMAP ACESCG -> SRGB   //
//============================//
Tonemapper_Reinhard_AcesCG_To_SRGB::Tonemapper_Reinhard_AcesCG_To_SRGB(const VulkanSystemView& sys)
    : handlesVk(&sys)
    , tonemapPipeline(sys.deviceVk)
    , reducePipeline(sys.deviceVk)
{}

MRayError Tonemapper_Reinhard_AcesCG_To_SRGB::Initialize(const std::string& execPath)
{
    using namespace std::string_literals;
    MRayError e = GenericCreatePipeline(tonemapPipeline, reducePipeline,
                                        tonemapDescriptorSets, reduceDescriptorSets,
                                        std::string(MODULE_NAME), execPath,
                                        "KCTonemapImage"s, "KCFindAvgMaxLum"s,
                                        *handlesVk);
    return e;
}

GUITonemapperI* Tonemapper_Reinhard_AcesCG_To_SRGB::AcquireGUI()
{
    return nullptr;
}

void Tonemapper_Reinhard_AcesCG_To_SRGB::TonemapImage(VkCommandBuffer cmd,
                                                      const VulkanImage& hdrImg,
                                                      const VulkanImage& sdrImg)
{

}

MRayColorSpaceEnum Tonemapper_Reinhard_AcesCG_To_SRGB::InputColorspace() const
{
    return IN_CS;
}

VkColorSpaceKHR Tonemapper_Reinhard_AcesCG_To_SRGB::OutputColorspace() const
{
    return OUT_CS;
}

size_t Tonemapper_Reinhard_AcesCG_To_SRGB::UniformBufferSize() const
{
    return sizeof(UniformBuffer);
}

void Tonemapper_Reinhard_AcesCG_To_SRGB::SetUniformBufferView(const UniformBufferMemView& uniformBufferView)
{
    uniformBuffer = uniformBufferView;
}

size_t Tonemapper_Reinhard_AcesCG_To_SRGB::StagingBufferSize() const
{
    return sizeof(MaxAvgStageBuffer);
}

void Tonemapper_Reinhard_AcesCG_To_SRGB::SetStagingBufferView(const StagingBufferMemView& view)
{
    stagingBuffer = view;
}

//============================//
//  TONEMAP ACESCG -> HDR10   //
//============================//
Tonemapper_Empty_AcesCG_To_HDR10::Tonemapper_Empty_AcesCG_To_HDR10(const VulkanSystemView& sys)
    : handlesVk(&sys)
    , tonemapPipeline(sys.deviceVk)
    , reducePipeline(sys.deviceVk)
{}

MRayError Tonemapper_Empty_AcesCG_To_HDR10::Initialize(const std::string& execPath)
{
    using namespace std::string_literals;
    MRayError e = GenericCreatePipeline(tonemapPipeline, reducePipeline,
                                        tonemapDescriptorSets, reduceDescriptorSets,
                                        std::string(MODULE_NAME), execPath,
                                        "KCTonemapImage"s, "KCFindAvgMaxLum"s,
                                        *handlesVk);
    return e;
}

GUITonemapperI* Tonemapper_Empty_AcesCG_To_HDR10::AcquireGUI()
{
    return nullptr;
}

void Tonemapper_Empty_AcesCG_To_HDR10::TonemapImage(VkCommandBuffer cmd,
                                                    const VulkanImage& hdrImg,
                                                    const VulkanImage& sdrImg)
{

}

MRayColorSpaceEnum Tonemapper_Empty_AcesCG_To_HDR10::InputColorspace() const
{
    return IN_CS;
}

VkColorSpaceKHR Tonemapper_Empty_AcesCG_To_HDR10::OutputColorspace() const
{
    return OUT_CS;
}

size_t Tonemapper_Empty_AcesCG_To_HDR10::UniformBufferSize() const
{
    return sizeof(EmptyType);
}

void Tonemapper_Empty_AcesCG_To_HDR10::SetUniformBufferView(const UniformBufferMemView&)
{}

size_t Tonemapper_Empty_AcesCG_To_HDR10::StagingBufferSize() const
{
    return sizeof(MaxAvgStageBuffer);
}

void Tonemapper_Empty_AcesCG_To_HDR10::SetStagingBufferView(const StagingBufferMemView& view)
{
    stagingBuffer = view;

}

//======================//
//    TONEMAP STAGE     //
//======================//
TonemapStage::TonemapStage(const VulkanSystemView& view)
    : handlesVk(&view)
    , stagingBuffer(view)
    , memory(view.deviceVk)
{}

//
//TonemapStage::TonemapStage(TonemapStage&& other)
//    : uniformBuffer(std::move(other.uniformBuffer))
//    , sdrImage(std::move(other.sdrImage))
//    , tonemappers(std::move(other.tonemappers))
//    , currentTonemapper(other.currentTonemapper)
//    , handlesVk(other.handlesVk)
//{}
//
//TonemapStage& TonemapStage::operator=(TonemapStage&& other)
//{
//    assert(this != &other);
//    uniformBuffer = std::move(other.uniformBuffer);
//    tonemappers = std::move(other.tonemappers);
//    currentTonemapper = other.currentTonemapper;
//    handlesVk = other.handlesVk;
//    return *this;
//}
//
//TonemapStage::~TonemapStage()
//{}

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

    tonemappers.try_emplace({tm0->InputColorspace(), tm0->OutputColorspace()},
                            std::move(tm0));
    tonemappers.try_emplace({tm1->InputColorspace(), tm1->OutputColorspace()},
                            std::move(tm1));

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
    return MRayError::OK;
}

void TonemapStage::ChangeImage(const VulkanImage* hdrImageIn,
                               const VulkanImage* sdrImageIn)
{
    hdrImage = hdrImageIn;
    sdrImage = sdrImageIn;

    // TODO: Invalidate descriptors
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

void TonemapStage::IssueTonemap(VkCommandBuffer)
{
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