#include "VisorGUI.h"
#include "VisorState.h"
#include "VulkanTypes.h"
#include "TonemapStage.h"

#include <Imgui/imgui.h>
#include <Imgui/imgui_internal.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_vulkan.h>

#include "../Resources/Fonts/IcoMoonFontTable.h"

#include "Core/MemAlloc.h"
#include "Core/BitFunctions.h"
#include "Core/Log.h"

#include <vulkan/vulkan.hpp>

enum class WindowLocationType
{
    TOP_LEFT = 0,
    TOP_RIGHT = 1,
    BOTTOM_LEFT = 2,
    BOTTOM_RIGHT = 3
};

Pair<ImVec2, ImVec2> CalculateInitialWindowLocation(WindowLocationType windowLocation)
{
    uint32_t location = static_cast<uint32_t>(windowLocation);
    static constexpr float PADDING = 10.0f;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImVec2 work_pos = viewport->WorkPos; // Use work area to avoid menu-bar/task-bar, if any!
    ImVec2 work_size = viewport->WorkSize;
    ImVec2 window_pos, window_pos_pivot;
    window_pos.x = (location & 1)
        ? (work_pos.x + work_size.x - PADDING)
        : (work_pos.x + PADDING);
    window_pos.y = (location & 2)
        ? (work_pos.y + work_size.y - PADDING)
        : (work_pos.y + PADDING);
    window_pos_pivot.x = (location & 1) ? 1.0f : 0.0f;
    window_pos_pivot.y = (location & 2) ? 1.0f : 0.0f;
    return {window_pos, window_pos_pivot};
}

Pair<Vector2, Vector2> GenAspectCorrectVP(const Vector2& fbSize,
                                          const Vector2& imgSize)
{
    Vector2 vpSize = Vector2::Zero();
    Vector2 vpOffset = Vector2::Zero();
    // Determine view-port by checking aspect ratio
    Float imgAspect = imgSize[0] / imgSize[1];
    Float screenAspect = fbSize[0] / fbSize[1];
    if(imgAspect > screenAspect)
    {
        Float ySize = std::round(fbSize[1] * screenAspect / imgAspect);
        Float yOffset = std::round((fbSize[1] - ySize) * Float(0.5));
        vpSize = Vector2(fbSize[0], ySize);
        vpOffset = Vector2(0, yOffset);
    }
    else
    {
        Float xSize = std::round(fbSize[0] * imgAspect / screenAspect);
        float xOffset = std::round((fbSize[0] - xSize) * Float(0.5));
        vpSize = Vector2(xSize, fbSize[1]);
        vpOffset = Vector2(xOffset, 0);
    }
    return {vpSize, vpOffset};
}

Pair<double, std::string_view> ConvertMemSizeToGUI(size_t size)
{
    // This function is overengineered for a GUI operation.
    // This probably has better precision? (probably not)
    // has high amount memory (TiB++ of memory).
    Pair<double, std::string_view> result;
    using namespace std::string_view_literals;
    size_t shiftVal = 0;
    if(size >= 1_TiB)
    {
        result.second = "TiB"sv;
        shiftVal = 40;
    }
    else if(size >= 1_GiB)
    {
        result.second = "GiB"sv;
        shiftVal = 30;
    }
    else if(size >= 1_MiB)
    {
        result.second = "MiB"sv;
        shiftVal = 20;
    }
    else if(size >= 1_KiB)
    {
        result.second = "KiB"sv;
        shiftVal = 10;
    }
    else
    {
        result.second = "Bytes"sv;
        shiftVal = 0;
    }

    size_t mask = ((size_t(1) << shiftVal) - 1);
    size_t integer = size >> shiftVal;
    size_t decimal = mask & size;
    // Sanity check
    static_assert(std::numeric_limits<double>::is_iec559,
                  "This overengineered function requires "
                  "IEE 754 floats.");
    static constexpr size_t DOUBLE_MANTISSA = 52;
    static constexpr size_t MANTISSA_MASK = (size_t(1) << DOUBLE_MANTISSA) - 1;
    size_t bitCount = Bit::RequiredBitsToRepresent(decimal);
    if(bitCount > DOUBLE_MANTISSA)
        decimal >>= (bitCount - DOUBLE_MANTISSA);
    else
        decimal <<= (DOUBLE_MANTISSA - bitCount);


    uint64_t dblFrac = std::bit_cast<uint64_t>(1.0);
    dblFrac |= decimal & MANTISSA_MASK;
    result.first = std::bit_cast<double>(dblFrac);
    result.first += static_cast<double>(integer) - 1.0;
    return result;
}

const VisorKeyMap VisorGUI::DefaultKeyMap =
{
    { VisorUserAction::TOGGLE_TOP_BAR, ImGuiKey::ImGuiKey_M },
    { VisorUserAction::TOGGLE_BOTTOM_BAR, ImGuiKey::ImGuiKey_N },
    { VisorUserAction::NEXT_CAM, ImGuiKey::ImGuiKey_Keypad6 },
    { VisorUserAction::PREV_CAM, ImGuiKey::ImGuiKey_Keypad4 },
    { VisorUserAction::TOGGLE_MOVEMENT_LOCK, ImGuiKey::ImGuiKey_Keypad5 },
    { VisorUserAction::PRINT_CUSTOM_CAMERA, ImGuiKey::ImGuiKey_KeypadDecimal },
    { VisorUserAction::NEXT_MOVEMENT, ImGuiKey::ImGuiKey_RightBracket },
    { VisorUserAction::PREV_MOVEMENT, ImGuiKey::ImGuiKey_LeftBracket },
    { VisorUserAction::NEXT_RENDERER, ImGuiKey::ImGuiKey_Keypad9 },
    { VisorUserAction::PREV_RENDERER, ImGuiKey::ImGuiKey_Keypad7 },
    { VisorUserAction::NEXT_RENDERER_CUSTOM_LOGIC_0, ImGuiKey::ImGuiKey_Keypad3 },
    { VisorUserAction::PREV_RENDERER_CUSTOM_LOGIC_0, ImGuiKey::ImGuiKey_Keypad1 },
    { VisorUserAction::NEXT_RENDERER_CUSTOM_LOGIC_1, ImGuiKey::ImGuiKey_KeypadAdd },
    { VisorUserAction::PREV_RENDERER_CUSTOM_LOGIC_1, ImGuiKey::ImGuiKey_KeypadSubtract },
    { VisorUserAction::PAUSE_CONT_RENDER, ImGuiKey::ImGuiKey_P },
    { VisorUserAction::START_STOP_TRACE, ImGuiKey::ImGuiKey_O },
    { VisorUserAction::CLOSE, ImGuiKey::ImGuiKey_Escape },
    { VisorUserAction::SAVE_IMAGE, ImGuiKey::ImGuiKey_G },
    { VisorUserAction::SAVE_IMAGE_HDR, ImGuiKey::ImGuiKey_H },
    // Movement
    { VisorUserAction::MOUSE_ROTATE_MODIFIER, ImGuiMouseButton_Left},
    { VisorUserAction::MOUSE_TRANSLATE_MODIFIER, ImGuiMouseButton_Middle},
    { VisorUserAction::MOVE_FORWARD, ImGuiKey::ImGuiKey_W},
    { VisorUserAction::MOVE_BACKWARD, ImGuiKey::ImGuiKey_S},
    { VisorUserAction::MOVE_RIGHT, ImGuiKey::ImGuiKey_D},
    { VisorUserAction::MOVE_LEFT, ImGuiKey::ImGuiKey_A},
    { VisorUserAction::FAST_MOVE_MODIFIER, ImGuiKey::ImGuiKey_LeftShift}
};

namespace ImGui
{
    bool ToggleButton(const char* name, bool& toggle,
                      bool decorate = true)
    {
        bool result = false;
        if(toggle == true)
        {
            ImVec4 hoverColor = ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered);

            if(decorate)
            {
                ImGui::PushID(name);
                ImGui::PushStyleColor(ImGuiCol_Button, hoverColor);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, hoverColor);
            }

            result = ImGui::Button(name);
            if(ImGui::IsItemClicked(0))
            {
                result = true;
                toggle = !toggle;
            }
            if(decorate)
            {
                ImGui::PopStyleColor(2);
                ImGui::PopID();
            };
        }
        else if(ImGui::Button(name))
        {
            result = true;
            toggle = true;
        }
        return result;
    }
}

TracerRunState DetermineTracerState(bool stopToggle,
                                    bool pauseToggle)
{
    if(stopToggle)
        return TracerRunState::STOPPED;
    else if(pauseToggle)
        return TracerRunState::PAUSED;
    return TracerRunState::RUNNING;
}

void SetButtonState(bool& stopToggle,
                    bool& runToggle,
                    bool& pauseToggle,
                    TracerRunState rs)
{
    stopToggle = false;
    runToggle = false;
    pauseToggle = false;
    switch(rs)
    {
        case TracerRunState::PAUSED:  pauseToggle = true; break;
        case TracerRunState::RUNNING: runToggle = true; break;
        case TracerRunState::STOPPED: stopToggle = true; break;
        default: break;
    }
}

MainStatusBar::MainStatusBar(const InputChecker& ic)
    : inputChecker(&ic)
    , paused(false)
    , running(false)
    , stopped(true)
{}

StatusBarChanges MainStatusBar::Render(const VisorState& visorState,
                                       bool camLocked)
{
    // This is here to align the GUI with programmatic changes to the
    // tracer state
    SetButtonState(stopped, running, paused,
                   visorState.currentRendererState);

    bool isChanged = false;
    // Handle keyboard related inputs
    if(inputChecker->CheckKeyPress(VisorUserAction::START_STOP_TRACE))
    {
        if(!paused)
        {
            isChanged = true;
            stopped = !stopped;
            running = !running;
        }
        else
        {
            isChanged = true;
            paused = false;
            stopped = true;
            running = false;
        }
    }
    if(inputChecker->CheckKeyPress(VisorUserAction::PAUSE_CONT_RENDER))
    {
        if(!stopped)
        {
            isChanged = true;
            paused = !paused;
            running = !running;
        }
    }

    using namespace std::string_literals;
    ImGuiWindowFlags window_flags = (ImGuiWindowFlags_NoScrollbar |
                                     ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_MenuBar);
    // Pre-init button state if it changed by keyboard
    //SetButtonState(stopped, running, paused, );
    int32_t camChange = 0;
    float height = ImGui::GetFrameHeight();
    if(ImGui::BeginViewportSideBar("##MainStatusBar", NULL,
                                   ImGuiDir_Down, height,
                                   window_flags))
    {
        if(ImGui::BeginMenuBar())
        {
            auto usedGPUMem = ConvertMemSizeToGUI(visorState.usedGPUMemoryBytes);
            auto totalGPUMem = ConvertMemSizeToGUI(visorState.tracer.totalGPUMemoryBytes);
            std::string memUsage = MRAY_FORMAT("{:.1f}{:s} / {:.1f}{:s}",
                                               usedGPUMem.first, usedGPUMem.second,
                                               totalGPUMem.first, totalGPUMem.second);

            ImGui::Text("%s", memUsage.c_str());
            ImGui::Separator();
            ImGui::Text("%s", (std::to_string(visorState.renderer.renderResolution[0]) + "x" +
                               std::to_string(visorState.renderer.renderResolution[1])).c_str());
            ImGui::Separator();
            ImGui::Text("%s", MRAY_FORMAT("{:>7.3f}{:s}",
                                          visorState.renderer.throughput,
                                          visorState.renderer.throughputSuffix).c_str());
            ImGui::Separator();
            ImGui::Text("%s", MRAY_FORMAT("{:>6.1f}{:s}",
                                          visorState.renderer.workPerPixel,
                                          visorState.renderer.workPerPixelSuffix).c_str());
            ImGui::Separator();

            std::string prefix = std::string(RENDERING_NAME);
            std::string body = (prefix + " " + visorState.scene.sceneName + "...");
            if(paused)
                body += " ("s + std::string(PAUSED_NAME) + ")"s;
            else if(stopped)
                body += " ("s + std::string(STOPPED_NAME) + ")"s;
            ImGui::Text("%s", body.c_str());

            float buttonSize = (ImGui::CalcTextSize(ICON_ICOMN_ARROW_LEFT).x +
                                ImGui::GetStyle().FramePadding.x * 2.0f);
            float spacingSize = ImGui::GetStyle().ItemSpacing.x;

            ImGui::SameLine(ImGui::GetWindowContentRegionMax().x -
                            (buttonSize * 5 + spacingSize * 6 + 2));

            ImGui::Separator();
            if(camLocked) ImGui::BeginDisabled();
            if(ImGui::Button(ICON_ICOMN_ARROW_LEFT) ||
               (!camLocked && inputChecker->CheckKeyPress(VisorUserAction::PREV_CAM)))
            {
                camChange--;
            }
            if(ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            {
                ImGui::BeginTooltip();
                ImGui::Text("Prev Camera");
                ImGui::EndTooltip();
            }

            if(ImGui::Button(ICON_ICOMN_ARROW_RIGHT) ||
               (!camLocked && inputChecker->CheckKeyPress(VisorUserAction::NEXT_CAM)))
            {
                camChange++;
            }
            if(ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            {
                ImGui::BeginTooltip();
                ImGui::Text("Next Camera");
                ImGui::EndTooltip();
            }
            if(camLocked) ImGui::EndDisabled();
            ImGui::Separator();

            if(ImGui::ToggleButton(ICON_ICOMN_STOP2, stopped))
            {
                if(running || paused)
                    isChanged = true;

                stopped = true;
                running = !stopped;
                paused = !stopped;

            }
            if(ImGui::ToggleButton(ICON_ICOMN_PAUSE2, paused))
            {
                if(!stopped)
                {
                    isChanged = true;
                    running = !paused;
                }
                else paused = false;
            }
            if(ImGui::ToggleButton(ICON_ICOMN_PLAY3, running))
            {
                if(stopped || paused) isChanged = true;

                running = true;
                stopped = false;
                paused = false;
            }
            ImGui::EndMenuBar();
        }
    }
    ImGui::End();



    TracerRunState newRunState = DetermineTracerState(stopped, paused);
    auto runStateResult = (isChanged)
        ? Optional<TracerRunState>(newRunState)
        : std::nullopt;
    auto camIndexResult = (camChange != 0)
        ? Optional<uint32_t>(camChange)
        : std::nullopt;
    return StatusBarChanges
    {
       .runState = runStateResult,
       .cameraIndex = camIndexResult
    };
}

void VisorGUI::ShowFrameOverlay(bool& isOpen,
                                const VisorState& visorState)
{
    static int location = 1;
    ImGuiWindowFlags window_flags = (ImGuiWindowFlags_NoDecoration |
                                     ImGuiWindowFlags_AlwaysAutoResize |
                                     ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_NoFocusOnAppearing |
                                     ImGuiWindowFlags_NoNav |
                                     ImGuiWindowFlags_NoMove);

    auto [window_pos, window_pos_pivot] = CalculateInitialWindowLocation(WindowLocationType::TOP_RIGHT);
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
    ImGui::SetNextWindowBgAlpha(0.33f); // Transparent background

    if(ImGui::Begin("##VisorOverlay", &isOpen, window_flags))
    {
        ImGui::Text("Frame       : %.2f ms", static_cast<double>(visorState.visor.frameTime));
        auto [sz, suffix] = ConvertMemSizeToGUI(visorState.visor.usedGPUMemory);
        std::string memUsage = MRAY_FORMAT("{:.1f}{:s}", sz, suffix);
        ImGui::Text("Memory      : %s", memUsage.c_str());

        const auto& sc = visorState.visor.swapchainInfo;
        ImGui::Separator();
        ImGui::Text("--Swapchain--");
        ImGui::Text("ColorSpace  : %s", vk::to_string(vk::ColorSpaceKHR(sc.colorSpace)).c_str());
        ImGui::Text("PresentMode : %s", vk::to_string(vk::PresentModeKHR(sc.presentMode)).c_str());
        ImGui::Text("Format      : %s", vk::to_string(vk::Format(sc.format)).c_str());
        ImGui::Text("Extent      : [%u, %u]", sc.extent.width, sc.extent.height);


        if(ImGui::BeginPopupContextWindow())
        {
            if(ImGui::MenuItem("Top-left", NULL, location == 0)) location = 0;
            if(ImGui::MenuItem("Top-right", NULL, location == 1)) location = 1;
            if(ImGui::MenuItem("Bottom-left", NULL, location == 2)) location = 2;
            if(ImGui::MenuItem("Bottom-right", NULL, location == 3)) location = 3;
            if(isOpen && ImGui::MenuItem("Close")) isOpen = false;
            ImGui::EndPopup();
        }
    }
    ImGui::End();
}

Optional<int32_t> VisorGUI::ShowRendererComboBox(const VisorState& visorState)
{
    if(visorState.tracer.rendererTypes.empty()) return std::nullopt;

    Optional<int32_t> result;
    const std::vector<std::string>& rTypes = visorState.tracer.rendererTypes;
    int32_t rCount = static_cast<int32_t>(rTypes.size());
    int32_t rIndex = visorState.currentRenderIndex;
    if(inputChecker.CheckKeyPress(VisorUserAction::NEXT_RENDERER) ||
       inputChecker.CheckKeyPress(VisorUserAction::PREV_RENDERER))
    {
        int32_t i = inputChecker.CheckKeyPress(VisorUserAction::PREV_RENDERER) ? -1 : 1;
        rIndex += i;
        rIndex = MathFunctions::Roll(rIndex, 0, rCount);
        result = rIndex;
    }
    const int32_t& curRIndex = (result.has_value())
                                    ? result.value()
                                    : visorState.currentRenderIndex;

    static const std::string RENDERER_DASHED = "Renderer-"s;
    float maxSize = std::transform_reduce(rTypes.cbegin(), rTypes.cend(), 0.0f,
    [](float a, float b) { return std::max(a, b); },
    [](const std::string& s)
    {
        return ImGui::CalcTextSize((RENDERER_DASHED + s).c_str()).x;
    });
    std::string prevName = RENDERER_DASHED + rTypes[static_cast<uint32_t>(curRIndex)];
    ImGui::SetNextItemWidth(maxSize + ImGui::GetStyle().FramePadding.x * 2.0f);
    if(ImGui::BeginCombo("##Renderers", prevName.c_str(),
                         ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightSmall))
    {
        for(int32_t i = 0; i < rCount; i++)
        {
            const auto& rendererName = visorState.tracer.rendererTypes[static_cast<uint32_t>(i)];
            bool isSelected = (i == curRIndex);
            if(ImGui::Selectable(rendererName.c_str(), &isSelected) &&
               i != curRIndex)
            {
                result = i;
            }
        }
        ImGui::EndCombo();
    }
    return result;
}

TopBarChanges VisorGUI::ShowTopMenu(const VisorState& visorState)
{
    auto CheckLogic = [&](int32_t index, uint32_t size,
                          VisorUserAction nextAction,
                          VisorUserAction prevAction) -> Optional<uint32_t>
    {
        Optional<uint32_t> result;
        if(size == 0) return result;

        int32_t count = static_cast<int32_t>(size);
        if(inputChecker.CheckKeyPress(nextAction) ||
           inputChecker.CheckKeyPress(prevAction))
        {
            int32_t i = inputChecker.CheckKeyPress(prevAction) ? -1 : 1;
            index += i;
            index = MathFunctions::Roll(index, 0, count);
            result = index;
        }
        return result;
    };

    TopBarChanges result;
    if(inputChecker.CheckKeyPress(VisorUserAction::TOGGLE_MOVEMENT_LOCK))
        camLocked = !camLocked;


    result.customLogicIndex0 = CheckLogic(visorState.currentRenderLogic0,
                                          visorState.renderer.customLogicSize0,
                                          VisorUserAction::NEXT_RENDERER_CUSTOM_LOGIC_0,
                                          VisorUserAction::PREV_RENDERER_CUSTOM_LOGIC_0);
    result.customLogicIndex1 = CheckLogic(visorState.currentRenderLogic1,
                                          visorState.renderer.customLogicSize1,
                                          VisorUserAction::NEXT_RENDERER_CUSTOM_LOGIC_1,
                                          VisorUserAction::PREV_RENDERER_CUSTOM_LOGIC_1);

    if(ImGui::BeginMainMenuBar())
    {
        ImGui::Text(" ");
        ImGui::ToggleButton("Tonemap", tmWindowOn);
        if(tmWindowOn && tonemapperGUI)
        {
            auto [wPos, wPivot] = CalculateInitialWindowLocation(WindowLocationType::TOP_LEFT);
            ImGui::SetNextWindowPos(wPos, ImGuiCond_Appearing, wPivot);
            result.newTMParams = tonemapperGUI->Render(tmWindowOn);
        }

        ImGui::Separator();

        result.rendererIndex = ShowRendererComboBox(visorState);

        static constexpr const char* VISOR_INFO_NAME = "VisorInfo ";
        float offsetX = (ImGui::GetWindowContentRegionMax().x -
                         ImGui::CalcTextSize(VISOR_INFO_NAME).x -
                         ImGui::CalcTextSize(ICON_ICOMN_LOCK).x -
                         ImGui::GetStyle().ItemSpacing.x * 4);
        ImGui::SameLine(offsetX);
        ImGui::Separator();

        if(camLocked) ImGui::ToggleButton(ICON_ICOMN_LOCK, camLocked, false);
        else ImGui::ToggleButton(ICON_ICOMN_UNLOCKED, camLocked, false);

        ImGui::ToggleButton(VISOR_INFO_NAME, fpsInfoOn);
        if(fpsInfoOn) ShowFrameOverlay(fpsInfoOn, visorState);

        ImGui::EndMainMenuBar();
    }
    return result;
}

StatusBarChanges VisorGUI::ShowStatusBar(const VisorState& visorState)
{
    return statusBar.Render(visorState, camLocked);
}

Optional<CameraTransform> VisorGUI::ShowMainImage(const VisorState& visorState)
{
    int32_t movementCount = static_cast<int32_t>(movementSchemes.size());
    if(inputChecker.CheckKeyPress(VisorUserAction::NEXT_MOVEMENT))
    {
        movementIndex++;
        movementIndex = MathFunctions::Roll(movementIndex, 0, movementCount);
    }
    else if(inputChecker.CheckKeyPress(VisorUserAction::PREV_MOVEMENT))
    {
        movementIndex--;
        movementIndex = MathFunctions::Roll(movementIndex, 0, movementCount);
    }

    Optional<CameraTransform> result;
    static const ImGuiWindowFlags flags = (ImGuiWindowFlags_NoDecoration |
                                           ImGuiWindowFlags_NoMove |
                                           ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoBackground |
                                           ImGuiWindowFlags_NoBringToFrontOnFocus |
                                           ImGuiWindowFlags_NoTitleBar |
                                           ImGuiWindowFlags_NoScrollbar |
                                           ImGuiWindowFlags_NoCollapse);
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2());
    if(ImGui::Begin("MainWindow", nullptr, flags))
    {
        if(mainImage)
        {
            Vector2 vpSize = Vector2(viewport->WorkSize.x,
                                     viewport->WorkSize.y);
            auto [size, offset] = GenAspectCorrectVP(vpSize, imgSize);

            ImGui::SetCursorPos({offset[0], offset[1]});
            ImGui::Image(std::bit_cast<ImTextureID>(mainImage),
                         {size[0], size[1]});
        }
        if(ImGui::IsWindowFocused() && !camLocked)
        {
            result = CurrentMovement().Update(inputChecker, visorState);
        }
    }
    ImGui::PopStyleVar();
    ImGui::End();
    return result;
}

MovementSchemeI& VisorGUI::CurrentMovement()
{
    return *movementSchemes[static_cast<uint32_t>(movementIndex)];
}

VisorGUI::VisorGUI(const VisorKeyMap* km)
    : inputChecker((km != nullptr) ? (*km) : DefaultKeyMap)
    , statusBar(inputChecker)
    , movementIndex(0)
{
    movementSchemes.emplace_back(std::make_unique<MovementSchemeFPS>());
    //movementSchemes.emplace_back(std::make_unique<MovementSchemeMaya>());
    //movementSchemes.emplace_back(std::make_unique<MovementSchemeImg>());
    assert(!movementSchemes.empty());
}

VisorGUI::VisorGUI(VisorGUI&& other)
    : inputChecker(std::move(other.inputChecker))
    , statusBar(inputChecker)
    , fpsInfoOn(other.fpsInfoOn)
    , bottomBarOn(other.bottomBarOn)
    , camLocked(other.camLocked)
    , tonemapperGUI(other.tonemapperGUI)
    , mainImage(other.mainImage)
    , imgSize(other.imgSize)
    , movementSchemes(std::exchange(other.movementSchemes, MovementSchemeList()))
    , movementIndex(other.movementIndex)
{}

VisorGUI& VisorGUI::operator=(VisorGUI&& other)
{
    assert(this != &other);
    inputChecker = std::move(other.inputChecker);
    statusBar = inputChecker;
    fpsInfoOn = other.fpsInfoOn;
    bottomBarOn = other.bottomBarOn;
    camLocked = other.camLocked;
    tonemapperGUI = other.tonemapperGUI;
    mainImage = other.mainImage;
    imgSize = other.imgSize;
    movementSchemes = std::exchange(other.movementSchemes, MovementSchemeList());
    movementIndex = other.movementIndex;

    return *this;
}

GUIChanges VisorGUI::Render(ImFont* windowScaledFont, const VisorState& visorState)
{
    using enum VisorUserAction;
    GUIChanges guiChanges;
    if(inputChecker.CheckKeyPress(CLOSE, false))
    {
        guiChanges.visorIsClosed = true;
        return guiChanges;
    }
    guiChanges.hdrSaveTrigger = inputChecker.CheckKeyPress(SAVE_IMAGE_HDR, false);
    guiChanges.sdrSaveTrigger = inputChecker.CheckKeyPress(SAVE_IMAGE, false);


    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::PushFont(windowScaledFont);

    if(inputChecker.CheckKeyPress(TOGGLE_TOP_BAR))
        topBarOn = !topBarOn;
    if(inputChecker.CheckKeyPress(TOGGLE_BOTTOM_BAR))
        bottomBarOn = !bottomBarOn;

    if(inputChecker.CheckKeyPress(PRINT_CUSTOM_CAMERA))
    {
        MRAY_LOG("\"gaze\"     : {},\n"
                 "\"position\" : {},\n"
                 "\"up\"       : {}",
                 visorState.transform.gazePoint.AsArray(),
                 visorState.transform.position.AsArray(),
                 visorState.transform.up.AsArray());
    }

    if(topBarOn)
        guiChanges.topBarChanges = ShowTopMenu(visorState);
    if(bottomBarOn)
        guiChanges.statusBarState = ShowStatusBar(visorState);
    guiChanges.transform = ShowMainImage(visorState);

    ImGui::PopFont();
    ImGui::Render();

    return guiChanges;
}

void VisorGUI::ChangeDisplayImage(const VulkanImage& img)
{
    if(mainImage) ImGui_ImplVulkan_RemoveTexture(mainImage);

    imgSize = Vector2(img.Extent());
    mainImage = ImGui_ImplVulkan_AddTexture(img.Sampler(), img.View(),
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VisorGUI::ChangeTonemapperGUI(GUITonemapperI* newTonemapperGUI)
{
    tonemapperGUI = newTonemapperGUI;
}