#pragma once

#include "Core/Vector.h"

#include <map>
#include <string>

struct MRayError;
class TimelineSemaphore;
class ThreadPool;
class TransferQueue;

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
    // These **must** be contigious so that
    // we can do indexed access (like GL_TEXTURE0 etc)
    // Also change "VisorMaxHotkeyCount" below
    NEXT_RENDERER_HOTKEY_0,
    NEXT_RENDERER_HOTKEY_1 = NEXT_RENDERER_HOTKEY_0 + 1,
    NEXT_RENDERER_HOTKEY_2 = NEXT_RENDERER_HOTKEY_0 + 2,
    NEXT_RENDERER_HOTKEY_3 = NEXT_RENDERER_HOTKEY_0 + 3,
    //
    PREV_RENDERER_HOTKEY_0,
    PREV_RENDERER_HOTKEY_1 = PREV_RENDERER_HOTKEY_0 + 1,
    PREV_RENDERER_HOTKEY_2 = PREV_RENDERER_HOTKEY_0 + 2,
    PREV_RENDERER_HOTKEY_3 = PREV_RENDERER_HOTKEY_0 + 3,
    //
    PAUSE_CONT_RENDER,
    START_STOP_TRACE,
    CLOSE,

    // On-demand img-save
    SAVE_IMAGE,
    SAVE_IMAGE_HDR,

    // Movement Related
    // TODO: We need to distinguish the key action and mouse action
    // since Imgui has separate function for mouse and key.
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

static constexpr uint32_t VisorMaxHotkeyCount = 4;

using VisorKeyMap = std::map<VisorUserAction, VisorInputType>;

struct VisorConfig
{
    // DLL Related
    std::string dllName;
    std::string dllCreateFuncName = "ConstructVisor";
    std::string dllDeleteFuncName = "DestroyVisor";
    // Window Related
    bool        enforceIGPU = true;
    bool        displayHDR  = true;
    bool        realTime    = false;
    bool        useWayland  = true;
    Vector2i    wSize       = Vector2i(1280, 720);
    // Technical
    uint32_t    commandBufferSize   = 8;
    uint32_t    responseBufferSize  = 8;

    // TODO: Move the default keymap to somewhere else
    VisorKeyMap keyMap;
};

class VisorI
{
    public:
    virtual             ~VisorI() = default;
    //
    virtual MRayError   MTInitialize(TransferQueue& transferQueue,
                                     TimelineSemaphore* syncSem,
                                     ThreadPool*, const VisorConfig&,
                                     const std::string& processPath) = 0;
    virtual bool        MTIsTerminated() = 0;
    virtual void        MTWaitForInputs() = 0;
    virtual bool        MTRender()  = 0;
    virtual void        MTDestroy() = 0;
    virtual void        TriggerEvent() = 0;
    virtual void        MTInitiallyStartRender(const Optional<std::string_view>& renderConfigPath,
                                               const Optional<std::string_view>& sceneFile) = 0;
};