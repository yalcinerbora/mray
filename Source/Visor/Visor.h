#pragma once

#include "VisorI.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "Core/Error.h"

class VisorVulkan : public VisorI
{
    private:
    static const std::string WINDOW_TITLE;

    GLFWwindow* window = nullptr;

    // OGL Debug Context Callback
    //static void     VulkanCallbackRender(const char* message,
    //                                     const void* userParam);

    //void                THRDProcessCommands();

    public:
                        VisorVulkan() = default;

    MRayError           MTInitialize(VisorConfig) override;
    void                MTWaitForInputs() override;
    void                MTRender() override;
    void                MTDestroy() override;
};