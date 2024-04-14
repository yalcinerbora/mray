#pragma once

#include "Core/MathForward.h"
#include "Core/Vector.h"
#include "Core/Types.h"
#include "vulkan/vulkan.h"

struct VisorState;
struct ImFont;

enum class RunState
{
    RUNNING,
    STOPPED,
    PAUSED,

    END
};

using namespace std::string_view_literals;

class MainStatusBar
{
    private:

    static constexpr auto RENDERING_NAME = "Rendering"sv;
    static constexpr auto PAUSED_NAME    = "PAUSED"sv;
    static constexpr auto STOPPED_NAME   = "STOPPED"sv;

    bool    paused;
    bool    running;
    bool    stopped;

    protected:
    public:
    // Constructors & Destructor
                MainStatusBar();
                ~MainStatusBar() = default;

    Optional<RunState>    Render(const VisorState&);
};

class VisorGUI
{
    private:
    MainStatusBar   statusBar;
    bool            fpsInfoOn      = false;
    bool            topBarOn       = true;
    bool            bottomBarOn    = true;

    void                ShowFrameOverlay(bool&, const VisorState&);
    void                ShowTopMenu(bool&, const VisorState&);
    Optional<RunState>  ShowStatusBar(bool&, const VisorState&);
    void                ShowMainImage();

    public:
    void            Render(ImFont* windowScaledFont,
                           VkDescriptorSet displayImage,
                           const VisorState& globalState);

    VkDescriptorSet AddTexForRender(VkImageView, VkSampler);
};