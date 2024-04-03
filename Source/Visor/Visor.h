#pragma once

#include "VisorI.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "Core/Error.h"
#include "Core/DataStructures.h"

// We gonna use this alot I think
template<size_t N>
using CStringList = StaticVector<const char*, N>;

class VulkanHostAllocator
{
    private:
    static void* VKAPI_CALL Allocate(void*, size_t, size_t,
                                     VkSystemAllocationScope);
    static void* VKAPI_CALL Realloc(void*, void*, size_t, size_t,
                                    VkSystemAllocationScope);
    static void VKAPI_CALL  Free(void*, void*);
    static void VKAPI_CALL  InternalAllocNotify(void*, size_t,
                                                VkInternalAllocationType,
                                                VkSystemAllocationScope);
    static void VKAPI_CALL  InternalFreeNotify(void*, size_t,
                                               VkInternalAllocationType,
                                               VkSystemAllocationScope);
    public:
    static const VkAllocationCallbacks* Functions();
};

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

    // Callbacks
    // GLFW
    static void             ErrorCallbackGLFW(int errorCode, const char* err);
    //


    private:
    // Vulkan related
    CStringList<64>     extensionList;
    CStringList<16>     layerList;
    VkInstance          instanceVk = nullptr;
    VkDevice            deviceVk = nullptr;
    uint32_t            queueFamilyIndex = std::numeric_limits<uint32_t>::max();
    VisorDebugSystem    debugSystem;
    // GLFW related
    GLFWwindow*         window = nullptr;

    private:
    // Vulkan related
    VkInstanceCreateInfo    EnableValidation(VkInstanceCreateInfo);
    MRayError               QueryAndPickPhysicalDevice(const VisorConfig&);


    //void                THRDProcessCommands();
    public:
                        VisorVulkan() = default;

    MRayError           MTInitialize(VisorConfig) override;
    void                MTWaitForInputs() override;
    void                MTRender() override;
    void                MTDestroy() override;
};