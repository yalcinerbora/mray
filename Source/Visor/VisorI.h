#pragma once

#include "Core/Vector.h"
#include <map>

using namespace std::string_literals;

class TransferQueue;
namespace BS { class thread_pool; }

// This is basically ImGui stuff
using VisorInputType = int;

enum class VisorUserAction : int
{
    TOGGLE_TOP_BAR,
    TOGGLE_BOTTOM_BAR,

    NEXT_CAM,
    PREV_CAM,
    TOGGLE_MOVEMENT_LOCK,
    PRINT_CUSTOM_CAMERA,

    NEXT_MOVEMENT,
    PREV_MOVEMENT,

    NEXT_RENDERER,
    PREV_RENDERER,

    // Custom renderer related
    NEXT_RENDERER_CUSTOM_LOGIC_0,
    PREV_RENDERER_CUSTOM_LOGIC_0,
    NEXT_RENDERER_CUSTOM_LOGIC_1,
    PREV_RENDERER_CUSTOM_LOGIC_1,

    //
    PAUSE_CONT_RENDER,
    START_STOP_TRACE,
    CLOSE,

    // On-demand img-save
    SAVE_IMAGE,
    SAVE_IMAGE_HDR,

    // Movement Related
    // TODO: We need to distinguish the key action and mouse action
    // since Imgui has seperate function for mouse and key.
    // Currently but "MOUSE_" prefix here for verbosity
    //
    MOUSE_ROTATE_MODIFIER,
    MOUSE_TRANSLATE_MODIFIER,
    //
    MOVE_FORWARD,
    MOVE_BACKWARD,
    MOVE_RIGHT,
    MOVE_LEFT,
    FAST_MOVE_MODIFIER
};

using VisorKeyMap = std::map<VisorUserAction, VisorInputType>;

struct VisorConfig
{
    // DLL Related
    std::string dllName;
    std::string dllCreateFuncName = "ConstructVisor"s;
    std::string dllDeleteFuncName = "DestroyVisor"s;
    // Window Related
    bool        enforceIGPU = true;
    bool        displayHDR  = true;
    bool        realTime    = false;
    Vector2i    wSize       = Vector2i(1280, 720);
    // Technical
    uint32_t    commandBufferSize   = 8;
    uint32_t    responseBufferSize  = 8;

    // TODO: Move the default keymap to somwhere else
    VisorKeyMap keyMap;
};

class VisorI
{
    public:
    virtual             ~VisorI() = default;
    //
    virtual MRayError   MTInitialize(TransferQueue& transferQueue,
                                     BS::thread_pool*,
                                     const VisorConfig&,
                                     const std::string& processPath) = 0;
    virtual bool        MTIsTerminated() = 0;
    virtual void        MTWaitForInputs() = 0;
    virtual void        MTRender()  = 0;
    virtual void        MTDestroy() = 0;
    virtual void        TriggerEvent() = 0;
    virtual void        MTInitiallyStartRender(std::string_view renderConfigFile) = 0;
};