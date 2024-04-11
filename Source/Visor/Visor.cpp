#include "Visor.h"

#include "Core/MRayDescriptions.h"
#include "Core/DataStructures.h"
#include "Core/MemAlloc.h"

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

    MRAY_LOG("[Vulkan]:[{}]:[{}]: {}",
             severity, type, pCallbackData->pMessage);
    return VK_FALSE;
}

MRayError VisorDebugSystem::Initialize(VkInstance inst)
{
    instance = inst;

    vkCreateDbgMessenger = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    vkDestroyDbgMessenger = (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

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

VkInstanceCreateInfo VisorVulkan::EnableValidation(VkInstanceCreateInfo vInfo)
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
            MRAY_LOG("Visor: Unable to find layer \"{}\"", layer);
            allLayersOK &= false;
        }
    }

    if(!allLayersOK)
    {
        MRAY_LOG("Visor: Unable to find some layers, "
                 "disabling all of the layers!");
        return vInfo;
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
    return vInfo;
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
    MRAY_LOG("Refreshed!");

    auto wPtr = static_cast<VisorWindow*>(glfwGetWindowUserPointer(wind));
    wPtr->WndRefreshed();


    wPtr->Render();
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

    static constexpr std::array<const char*, 3> RequiredExtensions =
    {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME,
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
    };

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

        // If device is *not* IGPU but we want IGPU continue
        if(deviceProps.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
           visorConfig.enforceIGPU)
            continue;
        // Other way around
        if(deviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
           !visorConfig.enforceIGPU)
            continue;

        // All good, check extensions
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(pDevice, nullptr,
                                             &extensionCount, nullptr);

        // Many extension better to use heap here
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(pDevice, nullptr, &extensionCount,
                                             availableExtensions.data());
        // TODO: Change linear search later
        bool hasAllExtensions =
        CheckAllInList(availableExtensions.cbegin(), availableExtensions.cend(),
                       RequiredExtensions.cbegin(), RequiredExtensions.cend(),
                       [](const VkExtensionProperties& p, const char* const name)
        {
            return std::strncmp(name, p.extensionName, VK_MAX_EXTENSION_NAME_SIZE) == 0;
        });

        // Required extensions are not available on this device skip
        if(!hasAllExtensions) continue;

        for(const auto& extName : RequiredExtensions)
            deviceExtList.push_back(extName);

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
            .queueFamilyIndex = selectedQueueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &priority
        };
        VkDeviceCreateInfo deviceCI =
        {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCI,
            .enabledLayerCount = static_cast<uint32_t>(layerList.size()),
            .ppEnabledLayerNames = layerList.data(),
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtList.size()),
            .ppEnabledExtensionNames = deviceExtList.data(),
            // Get all the features?
            // TODO: Does this has a drawback?
            .pEnabledFeatures = &deviceFeatures
        };

        // Actual device creation
        if(vkCreateDevice(pDevice, &deviceCI,
                          VulkanHostAllocator::Functions(),
                          &deviceVk))
            return MRayError("Unable to create logical device on \"{}\"!",
                             deviceProps.deviceName);

        // Store the selected physical device
        pDeviceVk = pDevice;

        // All is nice!
        // Report the device and exit
        queueFamilyIndex = selectedQueueFamilyIndex;

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(pDevice, &memProps);

        size_t memSize = 0;
        for(size_t i = 0; i < memProps.memoryHeapCount; i++)
        {
            const VkMemoryHeap& heap = memProps.memoryHeaps[i];
            if(heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            {
                memSize = heap.size;
                break;
            }
        }

        MRAY_LOG("----Visor-GPU----\n"
                 "Name      : {}\n"
                 "Max Tex2D : [{}, {}]\n"
                 "Memory    : {:.3f}GiB\n"
                 "-----------------\n",
                 deviceProps.deviceName,
                 deviceProps.limits.maxImageDimension2D,
                 deviceProps.limits.maxImageDimension2D,
                 (static_cast<double>(memSize) / 1024.0 / 1024.0 / 1024.0));

        // Get the queue
        vkGetDeviceQueue(deviceVk, queueFamilyIndex, 0,
                         &mainQueueVk);

        return MRayError::OK;
    }

    return MRayError("Unable to find Vulkan capable devices!");
}

Expected<VisorWindow> VisorVulkan::GenerateWindow(const VisorConfig& config)
{
    VisorWindow w;
    VulkanSystemView handlesVk =
    {
        .instanceVk     = instanceVk,
        .pDeviceVk      = pDeviceVk,
        .deviceVk       = deviceVk,
        .queueIndex     = queueFamilyIndex,
        .mainQueueVk    = mainQueueVk
    };
    MRayError e = w.Initialize(handlesVk, WindowTitle, config);
    if(e) return e;
    return w;
}

MRayError VisorVulkan::InitImGui()
{
    // Init Imgui stuff
    if(!IMGUI_CHECKVERSION())
        return MRayError("ImGui: Version mistmatch!");

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
    return MRayError::OK;
}

MRayError VisorVulkan::MTInitialize(VisorConfig visorConfig,
                                    const std::string& pPath)
{
    MRayError e = MRayError::OK;
    processPath = pPath;

    // From here
    // https://github.com/ocornut/imgui/blob/master/examples/example_glfw_vulkan/main.cpp
    // TBH, did not check what portability one is but
    // its here
    instanceExtList.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

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
    if constexpr(MRAY_IS_DEBUG)
    {
        instanceCreateInfo = EnableValidation(instanceCreateInfo);
        instanceCreateInfo.pNext = (const void*)VisorDebugSystem::CreateInfo();
    }

    // Finally create instance
    VkResult r = vkCreateInstance(&instanceCreateInfo,
                                  VulkanHostAllocator::Functions(),
                                  &instanceVk);
    if(r != VK_SUCCESS)
        return MRayError("Unable to create Vulkan Instance");

    if constexpr(MRAY_IS_DEBUG)
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
    auto windowE = GenerateWindow(visorConfig);
    if(windowE.has_error())
        return windowE.error();
    window = std::move(windowE.value());
    return MRayError::OK;
}

bool VisorVulkan::MTIsTerminated()
{
    return window.ShouldClose();
}

void VisorVulkan::MTWaitForInputs()
{
    glfwWaitEvents();
}

void VisorVulkan::MTRender()
{
    window.Render();
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
    // Destroy vulkan etc..
    vkDestroyDevice(deviceVk, VulkanHostAllocator::Functions());

    if constexpr(MRAY_IS_DEBUG)
    {
        debugSystem = VisorDebugSystem();
    }

    vkDestroyInstance(instanceVk, VulkanHostAllocator::Functions());
    // Terminal glfw system
    glfwTerminate();
}