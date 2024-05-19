#pragma once

#include "Core/MathForward.h"
#include "Core/Vector.h"
#include "Core/Types.h"
#include <vulkan/vulkan.h>

class GUITonemapperI;
class VulkanImage;
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
    bool            fpsInfoOn       = false;
    bool            topBarOn        = true;
    bool            bottomBarOn     = true;
    //
    GUITonemapperI* tonemapperGUI   = nullptr;
    //
    VkDescriptorSet mainImage       = nullptr;
    Vector2i        imgSize         = Vector2i::Zero();
    //
    void                ShowFrameOverlay(bool&, const VisorState&);
    void                ShowTopMenu(const VisorState&);
    Optional<RunState>  ShowStatusBar(const VisorState&);
    void                ShowMainImage();

    public:
    void            Render(ImFont* windowScaledFont,
                           const VisorState& globalState);
     void           ChangeDisplayImage(const VulkanImage&);
     void           ChangeTonemapperGUI(GUITonemapperI*);
};