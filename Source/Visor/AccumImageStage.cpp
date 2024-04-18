#include "AccumImageStage.h"
#include "VulkanAllocators.h"

PFN_vkGetMemoryHostPointerPropertiesEXT AccumImageStage::vkGetMemoryHostPointerProperties = nullptr;

AccumImageStage::AccumImageStage(const VulkanSystemView& handles)
    : handlesVk(&handles)
    , pixelImage(handles)
    , sampleImage(handles)
    , uniformBuffer(handles)
{
    if(vkGetMemoryHostPointerProperties == nullptr)
    {
        auto func = vkGetDeviceProcAddr(handlesVk->deviceVk,
                                        "vkGetMemoryHostPointerPropertiesEXT");
        vkGetMemoryHostPointerProperties = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(func);
    }
}

AccumImageStage::AccumImageStage(AccumImageStage&& other)
    : pixelImage(std::move(other.pixelImage))
    , sampleImage(std::move(other.sampleImage))
    , uniformBuffer(std::move(other.uniformBuffer))
    , imgAndBufferMemory(std::exchange(other.imgAndBufferMemory, nullptr))
    , foreignMemory(std::exchange(other.foreignMemory, nullptr))
    , foreignBuffer(std::exchange(other.foreignBuffer, nullptr))
    , readyForReadSignalVk(std::exchange(other.readyForReadSignalVk, nullptr))
    , readFinishedSignalVk(std::exchange(other.readFinishedSignalVk, nullptr))
    , handlesVk(other.handlesVk)
    , pipeline(std::move(other.pipeline))
{}

AccumImageStage& AccumImageStage::operator=(AccumImageStage&& other)
{
    assert(this != &other);
    Clear();
    pixelImage = std::move(other.pixelImage);
    sampleImage = std::move(other.sampleImage);
    uniformBuffer = std::move(other.uniformBuffer);
    imgAndBufferMemory = std::exchange(other.imgAndBufferMemory, nullptr);
    foreignMemory = std::exchange(other.foreignMemory, nullptr);
    foreignBuffer = std::exchange(other.foreignBuffer, nullptr);
    readyForReadSignalVk = std::exchange(other.readyForReadSignalVk, nullptr);
    readFinishedSignalVk = std::exchange(other.readFinishedSignalVk, nullptr);
    handlesVk = other.handlesVk;
    pipeline = std::move(other.pipeline);
    return *this;
}

AccumImageStage::~AccumImageStage()
{
    Clear();
}

MRayError AccumImageStage::Initialize(const std::string& execPath,
                                      VkDescriptorPool pool)
{
    using namespace std::string_literals;

    MRayError e = pipeline.Initialize(
    {
        {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}
        }
    },
    "AccumInput.spv"s, execPath, "KCAccumulateInputs"s);

    if(e) return e;


    return MRayError::OK;
}

void AccumImageStage::Clear()
{
    if(!foreignBuffer) return;

    pixelImage = VulkanImage(*handlesVk);
    sampleImage = VulkanImage(*handlesVk);
    uniformBuffer = VulkanBuffer(*handlesVk);
    vkFreeMemory(handlesVk->deviceVk, imgAndBufferMemory,
                 VulkanHostAllocator::Functions());
    //
    vkDestroyBuffer(handlesVk->deviceVk, foreignBuffer,
                    VulkanHostAllocator::Functions());
    vkDestroySemaphore(handlesVk->deviceVk, readyForReadSignalVk,
                       VulkanHostAllocator::Functions());
    vkDestroySemaphore(handlesVk->deviceVk, readFinishedSignalVk,
                       VulkanHostAllocator::Functions());
}

void AccumImageStage::ImportMemory(const RenderBufferInfo& rbI)
{
    static constexpr auto ForeignHostBit = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT;
    static const VkExternalMemoryBufferCreateInfo EXT_BUFF_INFO =
    {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = ForeignHostBit
    };
    VkBufferCreateInfo buffInfo =
    {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext = &EXT_BUFF_INFO,
        .flags = 0,
        .size = rbI.totalSize,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_CONCURRENT,
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices = &handlesVk->queueIndex,
    };
    vkCreateBuffer(handlesVk->deviceVk, &buffInfo,
                   VulkanHostAllocator::Functions(),
                   &foreignBuffer);


    VkMemoryHostPointerPropertiesEXT hostProps = {};
    vkGetMemoryHostPointerProperties(handlesVk->deviceVk,
                                     ForeignHostBit,
                                     rbI.data, &hostProps);

    // Allocation
    VkImportMemoryHostPointerInfoEXT hostImportInfo =
    {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .pNext = nullptr,
        .handleType = ForeignHostBit,
        .pHostPointer = rbI.data,
    };
    VkMemoryAllocateInfo memAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &hostImportInfo,
        .allocationSize = rbI.totalSize,
        .memoryTypeIndex = hostProps.memoryTypeBits
    };
    vkAllocateMemory(handlesVk->deviceVk, &memAllocInfo,
                     VulkanHostAllocator::Functions(),
                     &foreignMemory);
    // Binding
    vkBindBufferMemory(handlesVk->deviceVk, foreignBuffer, foreignMemory, 0);
}

void AccumImageStage::ResetBuffers(const RenderBufferInfo& rbI)
{
    Clear();
    ImportMemory(rbI);



}

void AccumImageStage::IssueAccumulation(VkCommandBuffer, const RenderImageSection& newSection)
{

}

void AccumImageStage::IssueClear(VkCommandBuffer cmd)
{
    VkClearColorValue value = {};
    value.float32[0] = 0.0f; value.float32[1] = 0.0f;
    value.float32[2] = 0.0f; value.float32[3] = 0.0f;


    pixelImage.IssueClear(cmd, value);
    sampleImage.IssueClear(cmd, value);
}