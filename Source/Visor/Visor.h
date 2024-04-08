#pragma once

#include "VisorI.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "Core/Error.h"
#include "Core/DataStructures.h"
#include "VisorWindow.h"

// We gonna use this alot I think
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
    static void     ErrorCallbackGLFW(int, const char*);
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


    private:
    CStringList<32>     instanceExtList;
    CStringList<32>     deviceExtList;
    CStringList<16>     layerList;

    VkInstance          instanceVk = nullptr;
    VkDevice            deviceVk = nullptr;
    VkPhysicalDevice    pDeviceVk = nullptr;
    uint32_t            queueFamilyIndex = std::numeric_limits<uint32_t>::max();
    VisorDebugSystem    debugSystem;

    VkInstanceCreateInfo    EnableValidation(VkInstanceCreateInfo);
    MRayError               QueryAndPickPhysicalDevice(const VisorConfig&);
    Expected<VisorWindow>   GenerateWindow(const VisorConfig&);

    // TODO: Move this away when multi-window is required
    VisorWindow         window;

    //void                THRDProcessCommands();
    public:
                        VisorVulkan() = default;

    MRayError           MTInitialize(VisorConfig) override;
    void                MTWaitForInputs() override;
    void                MTRender() override;
    void                MTDestroy() override;
};