#pragma once

#include "VisorI.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "Core/DataStructures.h"
#include "VisorWindow.h"

struct MRayError;

// We gonna use this a lot I think
template<size_t N>
using CStringList = StaticVector<const char*, N>;

class VisorDebugSystem
{
    private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL
    Callback(VkDebugUtilsMessageSeverityFlagBitsEXT,
             VkDebugUtilsMessageTypeFlagsEXT,
             const VkDebugUtilsMessengerCallbackDataEXT*,
             void*);

    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDbgMessenger = nullptr;
    PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDbgMessenger = nullptr;
    VkInstance instance = nullptr;
    VkDebugUtilsMessengerEXT messenger = nullptr;
    public:
    // Constructors & Destructor
                        VisorDebugSystem() = default;
                        VisorDebugSystem(const VisorDebugSystem&) = delete;
    VisorDebugSystem&   operator=(const VisorDebugSystem&) = delete;
    VisorDebugSystem&   operator=(VisorDebugSystem&&);
                        ~VisorDebugSystem();

    static const VkDebugUtilsMessengerCreateInfoEXT*
                        CreateInfo();

    MRayError           Initialize(VkInstance);
};

class VisorVulkan : public VisorI
{
    private:
    // Statics
    static const std::string Name;
    static const std::string WindowTitle;

    // GLFW Callbacks
    public:
    static void     WindowPosGLFW(GLFWwindow*, int, int);
    static void     WindowFBGLFW(GLFWwindow*, int, int);
    static void     WindowSizeGLFW(GLFWwindow*, int, int);
    static void     WindowCloseGLFW(GLFWwindow*);
    static void     WindowRefreshGLFW(GLFWwindow*);
    static void     WindowFocusedGLFW(GLFWwindow*, int);
    static void     WindowMinimizedGLFW(GLFWwindow*, int);
    static void     KeyboardUsedGLFW(GLFWwindow*, int, int, int, int);
    static void     MouseMovedGLFW(GLFWwindow*, double, double);
    static void     MousePressedGLFW(GLFWwindow*, int, int, int);
    static void     MouseScrolledGLFW(GLFWwindow*, double, double);
    static void     PathDroppedGLFW(GLFWwindow*, int, const char**);
    static void     RegisterCallbacks(GLFWwindow*);

    static void     ErrorCallbackGLFW(int, const char*);
    static void     MonitorCallback(GLFWmonitor*, int action);

    private:
    CStringList<32>     instanceExtList;
    CStringList<32>     deviceExtList;
    CStringList<16>     layerList;

    VkInstance          instanceVk          = nullptr;
    VkDevice            deviceVk            = nullptr;
    VkPhysicalDevice    pDeviceVk           = nullptr;
    VkQueue             mainQueueVk         = nullptr;
    VkCommandPool       mainCommandPool     = nullptr;
    VkDescriptorPool    mainDescPool        = nullptr;
    uint32_t            queueFamilyIndex    = std::numeric_limits<uint32_t>::max();
    uint32_t            deviceAlignment     = std::numeric_limits<uint32_t>::max();
    uint32_t            hostImportAlignment = std::numeric_limits<uint32_t>::max();
    VkSampler           llnSampler          = nullptr;
    VkSampler           nnnSampler          = nullptr;

    VisorDebugSystem    debugSystem;
    std::string         processPath;
    VisorConfig         config;
    // TODO: Move this away when multi-window is required
    // but imgui looks like single-windowed?
    VisorWindow         window;

    bool                    EnableValidation(VkInstanceCreateInfo&);
    MRayError               QueryAndPickPhysicalDevice(const VisorConfig&);
    Expected<VisorWindow>   GenerateWindow(TransferQueue::VisorView&,
                                           TimelineSemaphore* syncSem, ThreadPool*,
                                           const VisorConfig&);
    MRayError               InitImGui();

    public:
    // Constructors & Destructor
                        VisorVulkan() = default;
                        VisorVulkan(const VisorVulkan&) = delete;
                        VisorVulkan(VisorVulkan&&) = delete;
    VisorVulkan&        operator=(const VisorVulkan&) = delete;
    VisorVulkan&        operator=(VisorVulkan&&) = delete;
                        ~VisorVulkan() = default;
    //
    MRayError           MTInitialize(TransferQueue& transferQueue,
                                     TimelineSemaphore* syncSem,
                                     ThreadPool*,
                                     const VisorConfig&,
                                     const std::string& processPath) override;
    bool                MTIsTerminated() override;
    void                MTWaitForInputs() override;
    bool                MTRender() override;
    void                MTDestroy() override;
    void                TriggerEvent() override;
    void                MTInitiallyStartRender(const Optional<std::string_view>& renderConfigPath,
                                               const Optional<std::string_view>& sceneFile) override;
};