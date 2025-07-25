#pragma once

#include <string>
#include <vulkan/vulkan.h>

#include "Core/DataStructures.h"

#include "Common/TransferQueue.h"

#include "VulkanTypes.h"
#include "FramePool.h"
#include "VisorGUI.h"
#include "VisorState.h"
#include "AccumImageStage.h"
#include "TonemapStage.h"
#include "RenderImagePool.h"

class ThreadPool;
struct ImFont;
struct GLFWwindow;
struct VisorConfig;
struct MRayError;

struct FrameCounter
{
    public:
    static constexpr size_t AVG_FRAME_COUNT = 8;
    using QueryData = std::array<uint64_t, 4>;
    using FrameTimeAvg = Math::MovingAverage<AVG_FRAME_COUNT>;

    private:
    const VulkanSystemView* handlesVk = nullptr;
    VkQueryPool             queryPool = nullptr;
    VulkanCommandBuffer     startCommand;
    //
    QueryData       queryData;
    float           timestampPeriod = 0;
    FrameTimeAvg    avg;

    public:
                    FrameCounter() = default;
                    FrameCounter(const VulkanSystemView& handlesVk);
                    FrameCounter(const FrameCounter&) = delete;
                    FrameCounter(FrameCounter&&);
    FrameCounter&   operator=(const FrameCounter&) = delete;
    FrameCounter&   operator=(FrameCounter&&);
                    ~FrameCounter();

    bool        StartRecord(const VulkanTimelineSemaphore& sem);
    void        EndRecord(VkCommandBuffer cmd);
    float       AvgFrame();

};

class Swapchain
{
    static const std::array<VkColorSpaceKHR, 4> FormatListHDR;
    static const std::array<VkColorSpaceKHR, 6> FormatListSDR;
    static const std::array<VkPresentModeKHR, 3> PresentModes;

    private:
    // MESA intel iGPU returns minImage as 3 so increasing this to 4
    // TODO: Make it dynamic later maybe?
    static constexpr size_t MAX_WINDOW_FBO_COUNT = 8;
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

    SwapchainInfo               swapchainInfo   = {};

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

    FramebufferPack                 NextFrame(const VulkanBinarySemaphore& imgAvailSignal);
    void                            PresentFrame(const VulkanBinarySemaphore& waitSignal);
    void                            FBOSizeChanged(Vector2ui newSize);
    Pair<MRayColorSpaceEnum, Float> ColorSpace() const;
    VkColorSpaceKHR                 ColorSpaceVk() const;
    SwapchainInfo                   GetSwapchainInfo() const;

};

class VisorWindow
{
    private:
    static VulkanSystemView handlesVk;

    private:
    Swapchain           swapchain       = {};
    FramePool           framePool       = {};
    VkSurfaceKHR        surfaceVk       = nullptr;
    GLFWwindow*         window          = nullptr;
    bool                hdrRequested    = false;
    bool                stopPresenting  = false;
    ThreadPool*         threadPool      = nullptr;
    //
    VisorGUI                    gui;
    VisorState                  visorState      = {};
    FrameCounter                frameCounter;
    TransferQueue::VisorView*   transferQueue   = nullptr;
    // Rendering Stages
    AccumImageStage         accumulateStage;
    TonemapStage            tonemapStage;
    RenderImagePool         renderImagePool;
    MainUniformBuffer       uniformBuffer = MainUniformBuffer(handlesVk);
    VulkanTimelineSemaphore imgWriteSem;
    // Initial rendering
    Optional<std::string_view>      initialSceneFile;
    Optional<std::string_view>      initialTracerRenderConfigPath;
    //
    bool shouldPollRealTime = false;

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
    MRayError   Initialize(TransferQueue::VisorView& transferQueue,
                           const VulkanSystemView& handlesVk,
                           TimelineSemaphore* syncSem,
                           uint32_t hostImportAlignment,
                           ThreadPool* threadPool,
                           const std::string& windowTitle,
                           const VisorConfig& config,
                           const std::string& processPath);

    void        StartRenderpass(const FramePack& frameHandle);
    void        StartCommandBuffer(const FramePack& frameHandle);
    void        HandleGUIChanges(const GUIChanges&);
    void        DoInitialActions();
    size_t      QueryTotalGPUMemory() const;

    public:
    // Constructors & Destructor
    // TODO: Imgui has global state but relates to glfwWindow.
    // Dont know if move is valid...
                    VisorWindow(const VisorWindow&) = delete;
                    VisorWindow(VisorWindow&&);
    VisorWindow&    operator=(VisorWindow&) = delete;
    VisorWindow&    operator=(VisorWindow&&);
                    ~VisorWindow();

    //
    bool            ShouldClose();
    FramePack       NextFrame();
    void            PresentFrame(const VulkanTimelineSemaphore* extraWaitSemaphore);
    ImFont*         CurrentFont();
    bool            Render();
    void            SetKickstartParameters(const Optional<std::string_view>& renderConfigPath,
                                           const Optional<std::string_view>& sceneFile);
    bool            ShouldElevatePollingToRealTime() const;

};