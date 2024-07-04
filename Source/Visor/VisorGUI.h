#pragma once

#include "Common/AnalyticStructs.h"
#include "Core/MathForward.h"
#include "Core/Vector.h"
#include "Core/Types.h"

#include "VisorI.h"
#include "MovementSchemes.h"
#include "InputChecker.h"

#include <vulkan/vulkan.h>
#include <map>
#include <memory>

class GUITonemapperI;
class VulkanImage;
struct VisorState;
struct ImFont;



using namespace std::string_view_literals;


struct TopBarChanges
{
    Optional<int32_t> rendererIndex;
    Optional<int32_t> customLogicIndex0;
    Optional<int32_t> customLogicIndex1;
};

struct StatusBarChanges
{
    Optional<TracerRunState> runState;
    Optional<int32_t> cameraIndex;
};

struct GUIChanges
{
    TopBarChanges               topBarChanges;
    StatusBarChanges            statusBarState;
    Optional<CameraTransform>   transform;
    bool                        visorIsClosed = false;
    bool                        hdrSaveTrigger = false;
    bool                        sdrSaveTrigger = false;
};

class MainStatusBar
{
    private:
    static constexpr auto RENDERING_NAME = "Rendering"sv;
    static constexpr auto PAUSED_NAME    = "PAUSED"sv;
    static constexpr auto STOPPED_NAME   = "STOPPED"sv;

    const InputChecker* inputChecker;

    bool    paused;
    bool    running;
    bool    stopped;

    protected:
    public:
    // Constructors & Destructor
                        MainStatusBar(const InputChecker&);
                        ~MainStatusBar() = default;

    [[nodiscard]]
    StatusBarChanges    Render(const VisorState&, bool camLocked);
};

class VisorGUI
{
    using MovementSchemeList = std::vector<std::unique_ptr<MovementSchemeI>>;
    public:
    static const VisorKeyMap DefaultKeyMap;

    private:
    InputChecker    inputChecker;

    MainStatusBar   statusBar;
    bool            fpsInfoOn       = false;
    bool            topBarOn        = true;
    bool            bottomBarOn     = true;
    bool            camLocked       = true;
    bool            tmWindowOn      = false;
    //
    GUITonemapperI* tonemapperGUI   = nullptr;
    //
    VkDescriptorSet mainImage       = nullptr;
    Vector2i        imgSize         = Vector2i::Zero();
    //
    MovementSchemeList  movementSchemes;
    int32_t             movementIndex;
    //
    void                ShowFrameOverlay(bool&, const VisorState&);
    MovementSchemeI&    CurrentMovement();

    [[nodiscard]]
    TopBarChanges   ShowTopMenu(const VisorState&);

    [[nodiscard]]
    Optional<CameraTransform>
                    ShowMainImage(const VisorState&);

    [[nodiscard]]
    Optional<int32_t>
                    ShowRendererComboBox(const VisorState&);

    [[nodiscard]]
    StatusBarChanges
                    ShowStatusBar(const VisorState&);

    public:
                    VisorGUI(const VisorKeyMap* = nullptr);
                    VisorGUI(VisorGUI&&);
    VisorGUI&       operator=(VisorGUI&&);

    [[nodiscard]]
    GUIChanges      Render(ImFont* windowScaledFont,
                           const VisorState& globalState);
     void           ChangeDisplayImage(const VulkanImage&);
     void           ChangeTonemapperGUI(GUITonemapperI*);
};