#pragma once

#include "Common/AnalyticStructs.h"
#include "Common/TransferQueue.h"
#include "Core/MathForward.h"
#include "Core/Vector.h"
#include "Core/Types.h"
#include "Core/Optional.h"

#include "VisorI.h"
#include "MovementSchemes.h"
#include "InputChecker.h"

#include <vulkan/vulkan.h>
#include <memory>
#include <mutex>

class GUITonemapperI;
class VulkanImage;
struct VisorState;
struct ImFont;

struct TopBarChanges
{
    using RendererOptionList = StaticVector<RendererOptionData, 8>;
    //
    Optional<int32_t>   rendererIndex;
    RendererOptionList  changedRendererOptions;
    bool                newTMParams = false;
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
    static constexpr auto RENDERING_NAME = "Rendering";
    static constexpr auto PAUSED_NAME    = "PAUSED";
    static constexpr auto STOPPED_NAME   = "STOPPED";

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

class ImageSaveProgress
{
    private:
    float       counter;
    std::string filename;
    public:
    // Constructors & Destructor
            ImageSaveProgress(std::string&& filename);
    //
    float*  Counter();
    void    Render();
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
    bool            rendererOptsOn  = false;
    //
    GUITonemapperI* tonemapperGUI   = nullptr;
    //
    VkDescriptorSet mainImage       = nullptr;
    Vector2         imgSize         = Vector2::Zero();
    //
    MovementSchemeList  movementSchemes;
    int32_t             movementIndex;
    //
    void                ShowFrameOverlay(bool&, const VisorState&);
    MovementSchemeI&    CurrentMovement();
    //
    std::mutex                  imgSaveMutex;
    Optional<ImageSaveProgress> imgSaveProgress;

    //
    HeapRendererOptionPack      rendererOptionPack;

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

    [[nodiscard]]
    typename TopBarChanges::RendererOptionList
                    RenderRendererOptions(const VisorState&);

    public:
                    VisorGUI(const VisorKeyMap* = nullptr);
                    VisorGUI(VisorGUI&&);
    VisorGUI&       operator=(VisorGUI&&);

    [[nodiscard]]
    GUIChanges      Render(ImFont* windowScaledFont,
                           const VisorState& globalState);
    void            ChangeDisplayImage(const VulkanImage&);
    void            ChangeTonemapperGUI(GUITonemapperI*);
    void            OverrideRendererOptions(HeapRendererOptionPack&&);

    // Save progress related
    ImageSaveProgress& CreateSaveProgressWindow(std::string&& fileName);
    void               RemoveSaveProgressWindow();
};