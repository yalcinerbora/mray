#pragma once

#include "Common/AnalyticStructs.h"
#include "Core/MathForward.h"
#include "Core/Vector.h"
#include "Core/Types.h"
#include "VisorI.h"

#include <vulkan/vulkan.h>
#include <map>

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

struct GUIChanges
{
    using StatusBarChanges = Pair<Optional<RunState>, Optional<int32_t>>;
    //
    StatusBarChanges statusBarState;
    Optional<CameraTransform> transform;
};

class MainStatusBar
{
    private:
    static constexpr auto RENDERING_NAME = "Rendering"sv;
    static constexpr auto PAUSED_NAME    = "PAUSED"sv;
    static constexpr auto STOPPED_NAME   = "STOPPED"sv;

    const std::map<VisorUserAction, ImGuiKey>& keyMap;

    bool    paused;
    bool    running;
    bool    stopped;

    protected:
    public:
    // Constructors & Destructor
                MainStatusBar(const std::map<VisorUserAction, ImGuiKey>& km);
                ~MainStatusBar() = default;

    [[nodiscard]]
    typename GUIChanges::StatusBarChanges
                Render(const VisorState&);
};

class VisorGUI
{
    public:
    static const std::map<VisorUserAction, ImGuiKey> DefaultKeyMap;

    private:
    const std::map<VisorUserAction, ImGuiKey>& keyMap;

    MainStatusBar   statusBar;
    bool            fpsInfoOn       = false;
    bool            topBarOn        = true;
    bool            bottomBarOn     = true;
    bool            camLocked       = true;
    //
    GUITonemapperI* tonemapperGUI   = nullptr;
    //
    VkDescriptorSet mainImage       = nullptr;
    Vector2i        imgSize         = Vector2i::Zero();
    //
    void                        ShowFrameOverlay(bool&, const VisorState&);
    void                        ShowTopMenu(const VisorState&);
    Optional<CameraTransform>   ShowMainImage();

    [[nodiscard]]
    typename GUIChanges::StatusBarChanges
                    ShowStatusBar(const VisorState&);

    public:
    VisorGUI(const std::map<VisorUserAction, ImGuiKey>* = nullptr);

    [[nodiscard]]
    GUIChanges      Render(ImFont* windowScaledFont,
                           const VisorState& globalState);
     void           ChangeDisplayImage(const VulkanImage&);
     void           ChangeTonemapperGUI(GUITonemapperI*);
};