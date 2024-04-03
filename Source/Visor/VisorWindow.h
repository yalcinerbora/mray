#pragma once

#include <string>
#include <vulkan/vulkan.h>
#include "Core/Error.h"

struct GLFWwindow;
struct VisorConfig;

class VisorWindow
{
    GLFWwindow* window = nullptr;
    VkQueue     queue = nullptr;

    private:
    friend class VisorVulkan;
    void        WndPosChanged(int posX, int posY);
    void        WndFBChanged(int newX, int newY);
    void        WndResized(int newX, int newY);
    void        WndClosed();
    void        WndRefreshed();
    void        WndFocused(bool isFocused);
    void        WndMinimized(bool isMinimized);
    void        KeyboardUsed(int key, int scancode, int action, int modifier);
    void        MouseMoved(double px, double py);
    void        MousePressed(int button, int action, int modifier);
    void        MouseScrolled(double dx, double dy);

                VisorWindow() = default;

    MRayError   Initialize(VkDevice deviceVk,
                           VkPhysicalDevice pDeviceVk,
                           VkInstance instanceVk,
                           uint32_t queueFamilyIndex,
                           const std::string& windowTitle,
                           const VisorConfig& config);

    public:
    // Constructors & Destructor
                    VisorWindow(const VisorWindow&) = delete;
                    VisorWindow(VisorWindow&&);
    VisorWindow&    operator=(VisorWindow&&);
    VisorWindow&    operator=(VisorWindow&) = delete;
                    ~VisorWindow();
};