#include "Visor.h"

#include "Core/MRayDescriptions.h"
#include "Core/DataStructures.h"
#include "Core/Log.h"
#include "Core/Error.h"

#include <Imgui/imgui.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_vulkan.h>

#include "VisorWindow.h"
#include "VulkanAllocators.h"
#include "VulkanCapabilityFinder.h"
#include "FontAtlas.h"

VKAPI_ATTR VkBool32 VKAPI_CALL
VisorDebugSystem::Callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                           VkDebugUtilsMessageTypeFlagsEXT messageType,
                           const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                           void*)
{
    using namespace std::literals;
    std::string_view severity;
    switch(messageSeverity)
    {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            severity = "Verbose"sv;
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            severity = "Info"sv;
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            severity = "Warning"sv;
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            severity = "Error"sv;
            break;
        default: severity = "UNKNOWN!"sv; break;
    }
    std::string type;
    if(messageType >> 0 & 0b1)
        type += "General"s;
    else if(messageType >> 1 & 0b1)
        type += ((type.empty()) ? "" : "-") + "Validation"s;
    else if(messageType >> 2 & 0b1)
        type += ((type.empty()) ? "" : "-") + "Performance"s;
    else
        type = "UNKNOWN"sv;

    MRAY_LOG("[Vulkan]:[{}]:[{}]\n{}",
             severity, type, pCallbackData->pMessage);
    return VK_FALSE;
}

MRayError VisorDebugSystem::Initialize(VkInstance inst)
{
    instance = inst;

    vkCreateDbgMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>
        (vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));

    vkDestroyDbgMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>
        (vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));

    VkResult result = vkCreateDbgMessenger(instance,
                                           CreateInfo(),
                                           VulkanHostAllocator::Functions(),
                                           &messenger);
    if(result != VK_SUCCESS)
        return MRayError("Unable to create Vulkan Debug Messenger!");
    return MRayError::OK;
}

VisorDebugSystem& VisorDebugSystem::operator=(VisorDebugSystem&& other)
{
    assert(this != &other);

    if(instance && messenger)
        vkDestroyDbgMessenger(instance, messenger,
                              VulkanHostAllocator::Functions());

    instance = other.instance;
    vkCreateDbgMessenger = other.vkCreateDbgMessenger;
    vkDestroyDbgMessenger = other.vkDestroyDbgMessenger;
    messenger = other.messenger;
    return *this;
}

VisorDebugSystem::~VisorDebugSystem()
{
    if(instance && messenger)
        vkDestroyDbgMessenger(instance, messenger,
                              VulkanHostAllocator::Functions());
}

const VkDebugUtilsMessengerCreateInfoEXT* VisorDebugSystem::CreateInfo()
{
    static constexpr VkDebugUtilsMessengerCreateInfoEXT createInfo =
    {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity = (VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                            //VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT  |
                            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT),
        .messageType = (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT),
        .pfnUserCallback = &Callback,
        .pUserData = nullptr
    };
    return &createInfo;
}

// This is not initialization order fiasco, intra-translation unit
// init order is well defined (it is declaration order)
using namespace std::string_literals;
const std::string VisorVulkan::Name = std::string(MRay::Name) + "-Visor"s;
const std::string VisorVulkan::WindowTitle = Name + " "s + std::string(MRay::VersionString);

bool VisorVulkan::EnableValidation(VkInstanceCreateInfo& vInfo)
{
    using namespace std::literals;
    static constexpr std::array<const char*, 1> RequestedLayers =
    {
        "VK_LAYER_KHRONOS_validation"
    };

    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);

    bool allLayersOK = true;
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    for(const auto& layer : RequestedLayers)
    {
        auto it = std::find_if(availableLayers.cbegin(),
                               availableLayers.cend(),
                               [&layer](const VkLayerProperties& props)
        {
            return layer == std::string_view(props.layerName);
        });

        if(it == availableLayers.cend())
        {
            MRAY_WARNING_LOG("Visor: Unable to find layer \"{}\"", layer);
            allLayersOK &= false;
        }
    }

    if(!allLayersOK)
    {
        MRAY_WARNING_LOG("Visor: Unable to find validation related layers, "
                         "skipping....");
        return false;
    }

    vInfo.enabledLayerCount = static_cast<uint32_t>(RequestedLayers.size());
    vInfo.ppEnabledLayerNames = RequestedLayers.data();

    instanceExtList.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    vInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtList.size());
    vInfo.ppEnabledExtensionNames = instanceExtList.data();

    // TODO: normally this is not required either we make or break it
    // Move the layers to runtime
    layerList.resize(RequestedLayers.size());
    std::copy(RequestedLayers.cbegin(), RequestedLayers.cend(), layerList.begin());
    return true;
}

// Callbacks (Window Related)
void VisorVulkan::WindowPosGLFW(GLFWwindow* wind, int posX, int posY)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->WndPosChanged(posX, posY);
}

void VisorVulkan::WindowFBGLFW(GLFWwindow* wind, int newX, int newY)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->WndFBChanged(newX, newY);
}

void VisorVulkan::WindowSizeGLFW(GLFWwindow* wind, int newX, int newY)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->WndResized(newX, newY);
}

void VisorVulkan::WindowCloseGLFW(GLFWwindow* wind)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->WndClosed();
}

void VisorVulkan::WindowRefreshGLFW(GLFWwindow* wind)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->WndRefreshed();
}

void VisorVulkan::WindowFocusedGLFW(GLFWwindow* wind, int isFocused)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->WndFocused(isFocused);
}

void VisorVulkan::WindowMinimizedGLFW(GLFWwindow* wind, int isMinimized)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->WndMinimized(static_cast<bool>(isMinimized));
}

void VisorVulkan::KeyboardUsedGLFW(GLFWwindow* wind,
                                   int key, int scanCode, int action, int modifier)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->KeyboardUsed(key, scanCode, action, modifier);
}

void VisorVulkan::MouseMovedGLFW(GLFWwindow* wind, double px, double py)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->MouseMoved(px, py);
}

void VisorVulkan::MousePressedGLFW(GLFWwindow* wind, int key, int action, int modifier)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->MousePressed(key, action, modifier);
}

void VisorVulkan::MouseScrolledGLFW(GLFWwindow* wind, double dx, double dy)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->MouseScrolled(dx, dy);
}

void VisorVulkan::PathDroppedGLFW(GLFWwindow* wind, int count, const char** paths)
{
    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->PathDropped(count, paths);
}

void VisorVulkan::RegisterCallbacks(GLFWwindow* w)
{
    glfwSetWindowPosCallback(w, &VisorVulkan::WindowPosGLFW);
    glfwSetFramebufferSizeCallback(w, &VisorVulkan::WindowFBGLFW);
    glfwSetWindowSizeCallback(w, &VisorVulkan::WindowSizeGLFW);
    glfwSetWindowCloseCallback(w, &VisorVulkan::WindowCloseGLFW);
    glfwSetWindowRefreshCallback(w, &VisorVulkan::WindowRefreshGLFW);
    glfwSetWindowFocusCallback(w, &VisorVulkan::WindowFocusedGLFW);
    glfwSetWindowIconifyCallback(w, &VisorVulkan::WindowMinimizedGLFW);

    glfwSetKeyCallback(w, &VisorVulkan::KeyboardUsedGLFW);
    glfwSetCursorPosCallback(w, &VisorVulkan::MouseMovedGLFW);
    glfwSetMouseButtonCallback(w, &VisorVulkan::MousePressedGLFW);
    glfwSetScrollCallback(w, &VisorVulkan::MouseScrolledGLFW);
    glfwSetDropCallback(w, &VisorVulkan::PathDroppedGLFW);
}

// System related
void VisorVulkan::ErrorCallbackGLFW(int errorCode, const char* err)
{
    MRAY_ERROR_LOG("[GLFW]:[{}]: \"{}\"", errorCode, err);
}

void VisorVulkan::MonitorCallback(GLFWmonitor* monitor, int action)
{
    MRAY_LOG("Monitor!!!!!");
    if(action == GLFW_CONNECTED)
    {
        MRAY_LOG("[GLFW]: New Monitor: {}",
                 glfwGetMonitorName(monitor));
        FontAtlas::Instance().AddMonitorFont(monitor);
    }
    else if(action == GLFW_DISCONNECTED)
    {
        MRAY_LOG("[GLFW]: Monitor Removed: {}",
                 glfwGetMonitorName(monitor));
        FontAtlas::Instance().RemoveMonitorFont(monitor);
    }
}

MRayError VisorVulkan::QueryAndPickPhysicalDevice(const VisorConfig& visorConfig)
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instanceVk, &deviceCount, nullptr);

    if(deviceCount == 0)
        return MRayError("Unable to find Vulkan capable devices!");

    auto deviceList = StaticVector<VkPhysicalDevice, 32>(StaticVecSize(deviceCount));
    vkEnumeratePhysicalDevices(instanceVk, &deviceCount, deviceList.data());


    static StaticVector<const char*, 16> RequiredExtensions =
    {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME,
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        //VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
        VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME
    };

    static constexpr size_t MAX_GPU = 32;
    struct DeviceParams
    {
        VkPhysicalDeviceType    type;
        VkPhysicalDevice        pDevice;
        uint32_t                queueFamilyIndex;
    };
    StaticVector<DeviceParams, MAX_GPU>  goodDevices;

    for(const auto& pDevice : deviceList)
    {
        VkPhysicalDeviceProperties deviceProps;
        vkGetPhysicalDeviceProperties(pDevice, &deviceProps);

        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(pDevice, &deviceFeatures);

        uint32_t queuePropCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pDevice, &queuePropCount, nullptr);
        auto queuePropList = StaticVector<VkQueueFamilyProperties, 16>(StaticVecSize(queuePropCount));
        vkGetPhysicalDeviceQueueFamilyProperties(pDevice, &queuePropCount, queuePropList.data());

        // Check if this device has at least one proper queue
        uint32_t selectedQueueFamilyIndex = 0;
        bool isUsableDevice = false;
        for(const auto& queueProp : queuePropList)
        {
            isUsableDevice = ((queueProp.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                              (queueProp.queueFlags & VK_QUEUE_COMPUTE_BIT));
            if(isUsableDevice) break;
        }

        // If device is not capable continue
        if(!isUsableDevice) continue;

        // All good, check extensions
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(pDevice, nullptr,
                                             &extensionCount, nullptr);

        // Many extension better to use heap here
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(pDevice, nullptr, &extensionCount,
                                             availableExtensions.data());
        //
        bool hasAllExtensions =
        CheckAllInList(availableExtensions.cbegin(), availableExtensions.cend(),
                       RequiredExtensions.cbegin(), RequiredExtensions.cend(),
                       [](const VkExtensionProperties& p, const char* const name)
        {
            return std::strncmp(name, p.extensionName, VK_MAX_EXTENSION_NAME_SIZE) == 0;
        });

        // Required extensions are not available on this device skip
        if(!hasAllExtensions) continue;

        //
        goodDevices.emplace_back(deviceProps.deviceType, pDevice,
                                 selectedQueueFamilyIndex);
    }

    // Bad luck...
    if(goodDevices.isEmpty())
        return MRayError("Unable to find Vulkan capable devices!");

    auto loc = std::find_if(goodDevices.cbegin(),
                            goodDevices.cend(),
    [forceIGPU = visorConfig.enforceIGPU](const auto& gpu)
    {
        VkPhysicalDeviceType checkType = (forceIGPU)
                    ? VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
                    : VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        return gpu.type == checkType;
    });
    if(loc == goodDevices.cend())
    {
        using namespace std::string_view_literals;
        MRAY_WARNING_LOG("[Visor]: \"EnforceIGPU\" is {} but couldn't find {} GPU! "
                         "Continuing with first capable gpu",
                         (visorConfig.enforceIGPU) ? "on"sv : "off"sv,
                         (visorConfig.enforceIGPU) ? "integrated"sv : "discrete"sv);
        loc = goodDevices.cbegin();
    };

    // All should be fine, add extensions
    // Continue creating logical device
    for(const auto& extName : RequiredExtensions)
        deviceExtList.push_back(extName);

    DeviceParams selectedDevice = *loc;

    // Re-acquire props
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT extMemProps =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
        .pNext = nullptr,
        .minImportedHostPointerAlignment = 0
    };

    VkPhysicalDeviceProperties2 selectedDeviceProps
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &extMemProps,
        .properties = {}
    };
    vkGetPhysicalDeviceProperties2(selectedDevice.pDevice, &selectedDeviceProps);

    // Get the host import alignment
    hostImportAlignment = static_cast<uint32_t>(extMemProps.minImportedHostPointerAlignment);

    // The first device that matches the conditions
    // is deemed enough.
    //
    // TODO: Change this when MRay is capable enough
    // to support SYCL.
    //
    // For like 5 ppl in the world that has 2 different vendor
    // devices on the same PC, we may need this functionality
    //
    // Now actually create the device and queue
    // TODO: Get the dedicated DMA queue maybe?
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueCI =
    {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = selectedDevice.queueFamilyIndex,
        .queueCount = 1,
        .pQueuePriorities = &priority
    };

    //  Re-acquire device features here
    VkPhysicalDeviceSynchronization2Features sync2Features =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        .pNext = nullptr,
        .synchronization2 = 0
    };
    VkPhysicalDeviceHostQueryResetFeatures resetFeatures =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES,
        .pNext = &sync2Features,
        .hostQueryReset = 0
    };
    VkPhysicalDeviceTimelineSemaphoreFeatures semFeatures =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        .pNext = &resetFeatures,
        .timelineSemaphore = 0
    };
    VkPhysicalDeviceFeatures2 deviceFeatures2 =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &semFeatures,
        .features = {}
    };
    // We explicitly set all these to "false"
    // We first get the features then immediately feed this to "VkCreateDevice"
    // If it fails, terminate
    vkGetPhysicalDeviceFeatures2(selectedDevice.pDevice, &deviceFeatures2);

    // Actual device creation
    VkDeviceCreateInfo deviceCI =
    {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &deviceFeatures2,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueCI,
        .enabledLayerCount = static_cast<uint32_t>(layerList.size()),
        .ppEnabledLayerNames = (layerList.size() != 0) ? layerList.data() : nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtList.size()),
        .ppEnabledExtensionNames = deviceExtList.data(),
        // Get all the features?
        // TODO: Does this has a drawback?
        .pEnabledFeatures = nullptr
    };
    if(vkCreateDevice(selectedDevice.pDevice, &deviceCI,
                      VulkanHostAllocator::Functions(),
                      &deviceVk))
    {
        return MRayError("Unable to create logical device on \"{}\"!",
                         selectedDeviceProps.properties.deviceName);
    };
    // Store the selected physical device
    pDeviceVk = selectedDevice.pDevice;

    // All is nice!
    // Report the device and exit
    queueFamilyIndex = selectedDevice.queueFamilyIndex;

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(selectedDevice.pDevice,
                                        &memProps);
    // Get the queue
    vkGetDeviceQueue(deviceVk, queueFamilyIndex, 0,
                     &mainQueueVk);

    // Gen Command Pool
    VkCommandPoolCreateInfo cpCreateInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queueFamilyIndex
    };
    vkCreateCommandPool(deviceVk, &cpCreateInfo,
                        VulkanHostAllocator::Functions(),
                        &mainCommandPool);

    // Gen Descriptor Pool
        // Finally Create a descriptor pool
    // TODO: Check if large pool has performance penalty
    static const StaticVector<VkDescriptorPoolSize, 4> imguiPoolSizes =
    {
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32 }
    };
    VkDescriptorPoolCreateInfo descPoolInfo =
    {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 32,
        .poolSizeCount = static_cast<uint32_t>(imguiPoolSizes.size()),
        .pPoolSizes = imguiPoolSizes.data()
    };
    vkCreateDescriptorPool(deviceVk, &descPoolInfo,
                           VulkanHostAllocator::Functions(),
                           &mainDescPool);

    // Initialize the memory allocator
    // Find the common alignment
    const auto& limits = selectedDeviceProps.properties.limits;
    deviceAlignment = uint32_t(std::max(limits.minStorageBufferOffsetAlignment,
                                        limits.minUniformBufferOffsetAlignment));
    Span<const VkMemoryType> memTypeSpan(memProps.memoryTypes,
                                         memProps.memoryTypeCount);
    VulkanDeviceAllocator::Instance(deviceVk,
                                    deviceAlignment,
                                    memTypeSpan,
                                    selectedDevice.type);

    // Create the common samplers
    //  Min  / Mag  /  Mip
    // Linear/Linear/Nearest sampler
    VkSamplerCreateInfo llnSamplerCreateInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 1.0f,
        .anisotropyEnable = false,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0,
        .maxLod = 0,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = false
    };
    vkCreateSampler(deviceVk, &llnSamplerCreateInfo,
                    VulkanHostAllocator::Functions(),
                    &llnSampler);
    //   Min  /  Mag  /  Mip
    // Nearest/Nearest/Nearest sampler
    VkSamplerCreateInfo nnnSamplerCreateInfo = llnSamplerCreateInfo;
    nnnSamplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    nnnSamplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    vkCreateSampler(deviceVk, &nnnSamplerCreateInfo,
                    VulkanHostAllocator::Functions(),
                    &nnnSampler);
    return MRayError::OK;
}

Expected<VisorWindow> VisorVulkan::GenerateWindow(TransferQueue::VisorView& transferQueue,
                                                  TimelineSemaphore* syncSem,
                                                  ThreadPool* tp,
                                                  const VisorConfig& vConfig)
{
    VisorWindow w;
    VulkanSystemView handlesVk =
    {
        .instanceVk         = instanceVk,
        .pDeviceVk          = pDeviceVk,
        .deviceVk           = deviceVk,
        .queueIndex         = queueFamilyIndex,
        .mainQueueVk        = mainQueueVk,
        .mainCommandPool    = mainCommandPool,
        .mainDescPool       = mainDescPool,
        .llnSampler         = llnSampler,
        .nnnSampler         = nnnSampler
    };
    MRayError e = w.Initialize(transferQueue, handlesVk,
                               syncSem, hostImportAlignment,
                               tp, WindowTitle,
                               vConfig, processPath);
    if(e) return e;
    return w;
}

MRayError VisorVulkan::InitImGui()
{
    // Init Imgui stuff
    if(!IMGUI_CHECKVERSION())
        return MRayError("ImGui: Version mismatch!");

    if(ImGui::CreateContext() == nullptr)
        return MRayError("ImGui: Unable to create context!");

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Pre-generate fonts
    FontAtlas::Instance(processPath);

    int monitorCount;
    GLFWmonitor** monitorList = glfwGetMonitors(&monitorCount);
    for(int i = 0; i < monitorCount; i++)
        FontAtlas::Instance().AddMonitorFont(monitorList[i]);

    ImGui::StyleColorsDark();
    auto& style = ImGui::GetStyle();
    style.Colors[ImGuiCol_Button] = style.Colors[ImGuiCol_MenuBarBg];
    style.Colors[ImGuiCol_Button].w = 0.1f;
    return MRayError::OK;
}

MRayError VisorVulkan::MTInitialize(TransferQueue& transferQueue,
                                    TimelineSemaphore* syncSem,
                                    ThreadPool* tp,
                                    const VisorConfig& visorConfig,
                                    const std::string& pPath)
{
    MRayError e = MRayError::OK;
    processPath = pPath;
    config = visorConfig;

    // From here
    // https://github.com/ocornut/imgui/blob/master/examples/example_glfw_vulkan/main.cpp
    // TBH, did not check what portability one is but
    // its here
    instanceExtList.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    instanceExtList.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);

    glfwInitVulkanLoader(vkGetInstanceProcAddr);
    int err = glfwInit();
    if(err != GLFW_TRUE)
    {
        const char* errString; glfwGetError(&errString);
        return MRayError("GLFW: {}", errString);
    }
    glfwSetErrorCallback(&VisorVulkan::ErrorCallbackGLFW);
    glfwSetMonitorCallback(&VisorVulkan::MonitorCallback);

    uint32_t glfwExtCount;
    auto glfwExtNames = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    for(uint32_t i = 0; i < glfwExtCount; i++)
        instanceExtList.push_back(glfwExtNames[i]);


    const VkApplicationInfo VisorAppInfo =
    {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = MRay::Name.data(),
        .applicationVersion = VK_MAKE_API_VERSION(0,
                                                  MRay::VersionMajor,
                                                  MRay::VersionMinor,
                                                  MRay::VersionPatch),
        .pEngineName = VisorVulkan::Name.c_str(),
        .engineVersion = VK_MAKE_API_VERSION(0,
                                             MRay::VersionMajor,
                                             MRay::VersionMinor,
                                             MRay::VersionPatch),
        .apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 280)
    };
    VkInstanceCreateInfo instanceCreateInfo =
    {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = &VisorAppInfo,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(instanceExtList.size()),
        .ppEnabledExtensionNames = instanceExtList.data(),
    };

    // Add debug layer if "DEBUG"
    bool validationLayerFound = false;
    if constexpr(MRAY_IS_DEBUG)
    {
        validationLayerFound = EnableValidation(instanceCreateInfo);
        if(validationLayerFound)
            instanceCreateInfo.pNext = static_cast<const void*>(VisorDebugSystem::CreateInfo());
    }

    // Finally create instance
    VkResult r = vkCreateInstance(&instanceCreateInfo,
                                  VulkanHostAllocator::Functions(),
                                  &instanceVk);
    if(r != VK_SUCCESS)
        return MRayError("Unable to create Vulkan Instance");

    if constexpr(MRAY_IS_DEBUG)
    if(validationLayerFound)
    {
        e = debugSystem.Initialize(instanceVk);
        if(e) return e;
    }

    // Now query devices
    e = QueryAndPickPhysicalDevice(visorConfig);
    if(e) return e;

    // Imgui
    e = InitImGui();
    if(e) return e;

    // Main window
    auto windowE = GenerateWindow(transferQueue.GetVisorView(),
                                  syncSem, tp, visorConfig);
    if(windowE.has_error())
        return windowE.error();
    window = std::move(windowE.value());
    return MRayError::OK;
}

void VisorVulkan::TriggerEvent()
{
    glfwPostEmptyEvent();
}

bool VisorVulkan::MTIsTerminated()
{
    return window.ShouldClose();
}

void VisorVulkan::MTWaitForInputs()
{
    if(config.realTime || window.ShouldElevatePollingToRealTime())
        glfwPollEvents();
    else
        glfwWaitEvents();
}

bool VisorVulkan::MTRender()
{
    return window.Render();
}

void VisorVulkan::MTDestroy()
{
    // Finalize everything
    vkDeviceWaitIdle(deviceVk);
    // Destroy swapchain window etc..
    window = VisorWindow();

    // Destroy Imgui
    FontAtlas::Instance().ClearFonts();
    ImGui::DestroyContext();

    // Samplers
    vkDestroySampler(deviceVk, llnSampler,
                    VulkanHostAllocator::Functions());
    vkDestroySampler(deviceVk, nnnSampler,
                     VulkanHostAllocator::Functions());

    // Destroy vulkan etc..
    vkDestroyCommandPool(deviceVk, mainCommandPool,
                         VulkanHostAllocator::Functions());
    vkDestroyDescriptorPool(deviceVk, mainDescPool,
                         VulkanHostAllocator::Functions());
    vkDestroyDevice(deviceVk, VulkanHostAllocator::Functions());

    if constexpr(MRAY_IS_DEBUG)
    {
        debugSystem = VisorDebugSystem();
    }

    vkDestroyInstance(instanceVk, VulkanHostAllocator::Functions());
    // Terminal glfw system
    glfwTerminate();
}

void VisorVulkan::MTInitiallyStartRender(const Optional<std::string_view>& renderConfigPath,
                                         const Optional<std::string_view>& sceneFile)
{
    window.SetKickstartParameters(renderConfigPath, sceneFile);
}