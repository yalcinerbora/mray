#include "Visor.h"
#include "Core/MRayDescriptions.h"

#include <Imgui/imgui.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_vulkan.h>

using namespace std::string_literals;

const std::string VisorVulkan::WINDOW_TITLE = (std::string(MRay::Name) + "Visor"s +
                                               std::string(MRay::VersionString));

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage, void* pUserData)
{
    (void)flags; (void)object; (void)location; (void)messageCode; (void)pUserData; (void)pLayerPrefix; // Unused arguments
    fprintf(stderr, "[vulkan] Debug report from ObjectType: %i\nMessage: %s\n\n", objectType, pMessage);
    return VK_FALSE;
};

VkAllocationCallbacks a;

MRayError VisorVulkan::MTInitialize(VisorConfig config)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();


    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(config.wSize[0], config.wSize[1],
                                          WINDOW_TITLE.c_str(),
                                          nullptr, nullptr);

    VkRenderPass renderPass;
    VkInstance instance;
    VkSurfaceKHR surface;
    VkResult err = glfwCreateWindowSurface(instance, window, nullptr, &surface);
    if(err)
    {
        // Window surface creation failed
        return MRayError("Unable to create vksurf");
    }

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo init_info = {};

    ImGui_ImplVulkan_Init(&init_info, renderPass);
}

void VisorVulkan::MTWaitForInputs()
{
    glfwWaitEvents();
}

void VisorVulkan::MTRender()
{
    VkCommandBuffer commandBuffer;


    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // GUI Setup
    //---------
    ImGui::ShowDemoWindow();
    //---------

    // IssueCommand
    // -->

    // Rendering
    // ---------

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

void VisorVulkan::MTDestroy()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}