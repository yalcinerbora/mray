#pragma once

#include <string>
#include <vulkan/vulkan.h>

#include "Core/Error.h"
#include "Core/DataStructures.h"

#include "VulkanTypes.h"
#include "FramePool.h"
#include "VisorGUI.h"

struct ImFont;
struct GLFWwindow;
struct VisorConfig;

class Swapchain
{
    static const std::array<VkColorSpaceKHR, 4> FormatListHDR;
    static const std::array<VkColorSpaceKHR, 6> FormatListSDR;
    static const std::array<VkPresentModeKHR, 3> PresentModes;

    private:
    static constexpr size_t MAX_WINDOW_FBO_COUNT = 3;
    template<class T>
    using SwapchainVec = StaticVector<T, MAX_WINDOW_FBO_COUNT>;

    VkSurfaceCapabilitiesKHR                capabilities;
    StaticVector<VkSurfaceFormatKHR, 32>    surfaceTypeList;
    StaticVector<VkPresentModeKHR, 8>       presentModeTypeList;

    // From other classes
    VulkanSystemView            handlesVk       = {};
    VkSurfaceKHR                surface         = nullptr;
    // Cleanup responsible
    VkSwapchainKHR              swapChainVk     = nullptr;
    VkRenderPass                renderPass      = nullptr;
    SwapchainVec<VkImage>       images          = {};
    SwapchainVec<VkImageView>   imageViews      = {};
    SwapchainVec<VkFramebuffer> framebuffers    = {};

    bool                        tryHDR          = false;
    Vector2ui                   fboSize         = Vector2ui::Zero();
    bool                        fboSizeChanged  = false;
    uint32_t                    currentImgIndex = 0;

    VkPresentModeKHR            presentMode     = VK_PRESENT_MODE_FIFO_KHR;
    VkColorSpaceKHR             colorSpace      = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkFormat                    format          = VK_FORMAT_UNDEFINED;
    VkExtent2D                  extent          = VkExtent2D{0, 0};

    // Imgui Related
    VkDescriptorPool            imguiDescPool   = nullptr;

    //
    MRayError                   FixSwapchain(bool isFirstFix = false);
    VkSwapchainKHR              Cleanup(bool deleteSwapchain,
                                        bool issueImguiShutdown = true);

    public:
                        Swapchain() = default;
                        Swapchain(const Swapchain&) = delete;
                        Swapchain(Swapchain&&);
    Swapchain&          operator=(const Swapchain&) = delete;
    Swapchain&          operator=(Swapchain&&);
                        ~Swapchain();

    MRayError           Initialize(VulkanSystemView handlesVk,
                                   VkSurfaceKHR surface,
                                   bool tryHDR);

    FramebufferPack     NextFrame(VkSemaphore imgAvailSignal);
    void                PresentFrame(VkSemaphore waitSignal);
    void                FBOSizeChanged(Vector2ui newSize);

};

class VisorWindow
{
    private:
    Swapchain           swapchain       = {};
    FramePool           framePool       = {};
    VkSurfaceKHR        surfaceVk       = nullptr;
    GLFWwindow*         window          = nullptr;
    bool                hdrRequested    = false;
    VulkanSystemView    handlesVk       = {};
    bool                stopPresenting  = false;
    VisorGUI            gui;


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
    void        PathDropped(int count, const char** paths);
    // Only Visor can create windows
                VisorWindow() = default;
    // Thus, only visor can initialize windows
    MRayError   Initialize(VulkanSystemView handlesVk,
                           const std::string& windowTitle,
                           const VisorConfig& config);

    public:
    // Constructors & Destructor
    // TODO: Imgui has global state but relates to glfwWindow.
    // Dont know if move is valid...
                    VisorWindow(const VisorWindow&) = delete;
                    VisorWindow(VisorWindow&&);
    VisorWindow&    operator=(VisorWindow&) = delete;
    VisorWindow&    operator=(VisorWindow&&);
                    ~VisorWindow();

    bool                ShouldClose();
    FramePack           NextFrame();
    void                PresentFrame();
    ImFont*             CurrentFont();
    void                Render();

};