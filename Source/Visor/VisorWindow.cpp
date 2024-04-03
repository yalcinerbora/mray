#include "VisorWindow.h"

#include <GLFW/glfw3.h>
#include <cassert>

#include "Core/Error.h"
#include "VisorI.h"
#include "VulkanAllocators.h"

void VisorWindow::WndPosChanged(int posX, int posY)
{
}

void VisorWindow::WndFBChanged(int newX, int newY)
{
}

void VisorWindow::WndResized(int newX, int newY)
{
}

void VisorWindow::WndClosed()
{
}

void VisorWindow::WndRefreshed()
{
}

void VisorWindow::WndFocused(bool isFocused)
{
}

void VisorWindow::WndMinimized(bool isMinimized)
{
}

void VisorWindow::KeyboardUsed(int key, int scancode, int action, int modifier)
{
}

void VisorWindow::MouseMoved(double px, double py)
{
}

void VisorWindow::MousePressed(int button, int action, int modifier)
{
}

void VisorWindow::MouseScrolled(double dx, double dy)
{
}

MRayError VisorWindow::Initialize(VkDevice deviceVk,
                                  VkPhysicalDevice pDeviceVk,
                                  VkInstance instanceVk,
                                  uint32_t queueFamilyIndex,
                                  const std::string& windowTitle,
                                  const VisorConfig& config)
{
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(config.wSize[0], config.wSize[1],
                              windowTitle.c_str(), nullptr, nullptr);

    // Acquire a surface
    VkSurfaceKHR surface;
    if(glfwCreateWindowSurface(instanceVk, window,
                               VulkanHostAllocator::Functions(),
                               &surface))
    {
        // Window surface creation failed1
        return MRayError("Unable to create VkSurface for a GLFW window!");
    }

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(pDeviceVk, queueFamilyIndex,
                                         surface, &presentSupport);

    VkQueue presentQueue;
    vkGetDeviceQueue(deviceVk, queueFamilyIndex, 0, &presentQueue);

    // Translate the surface

    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pDeviceVk, surface,
                                              &capabilities);
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pDeviceVk, surface,
                                         &formatCount, nullptr);
    formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pDeviceVk, surface,
                                         &formatCount, formats.data());
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pDeviceVk, surface,
                                              &presentModeCount, nullptr);
    presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(pDeviceVk, surface,
                                              &presentModeCount,
                                              presentModes.data());


    MRAY_LOG(".....");
    return MRayError::OK;
}

VisorWindow::VisorWindow(VisorWindow&& other)
    : window(other.window)
    , queue(other.queue)
{
    other.window = nullptr;
    other.queue = nullptr;
}

VisorWindow& VisorWindow::operator=(VisorWindow&& other)
{
    assert(this != &other);
    if(window) glfwDestroyWindow(window);
    //if(queue) vkDestroyDeviceQueue
    window = other.window;
    queue = other.queue;

    other.window = nullptr;
    other.queue = nullptr;

    return *this;
}

VisorWindow::~VisorWindow()
{


    glfwDestroyWindow(window);
}