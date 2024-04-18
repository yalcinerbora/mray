#include "VisorGUI.h"
#include "VisorState.h"
#include "VulkanTypes.h"
#include "TonemapStage.h"

#include <Imgui/imgui.h>
#include <Imgui/imgui_internal.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_vulkan.h>

#include "../Resources/Fonts/IcoMoonFontTable.h"

namespace ImGui
{
    bool ToggleButton(const char* name, bool& toggle)
    {
        bool result = false;
        if(toggle == true)
        {
            ImVec4 hoverColor = ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered);

            ImGui::PushID(name);
            ImGui::PushStyleColor(ImGuiCol_Button, hoverColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, hoverColor);
            result = ImGui::Button(name);
            if(ImGui::IsItemClicked(0))
            {
                result = true;
                toggle = !toggle;
            }
            ImGui::PopStyleColor(2);
            ImGui::PopID();
        }
        else if(ImGui::Button(name))
        {
            result = true;
            toggle = true;
        }
        return result;
    }
}

RunState DetermineTracerState(bool stopToggle,
                              bool runToggle,
                              bool pauseToggle)
{
    if(stopToggle)
        return RunState::STOPPED;
    else if(pauseToggle)
        return RunState::PAUSED;
    return RunState::RUNNING;
}

void SetButtonState(bool& stopToggle,
                    bool& runToggle,
                    bool& pauseToggle,
                    RunState rs)
{
    stopToggle = false;
    runToggle = false;
    pauseToggle = false;
    switch(rs)
    {
        case RunState::PAUSED:  pauseToggle = true; break;
        case RunState::RUNNING: runToggle = true; break;
        case RunState::STOPPED: stopToggle = true; break;
        default: break;
    }
}

MainStatusBar::MainStatusBar()
    : paused(false)
    , running(true)
    , stopped(false)
{}

Optional<RunState> MainStatusBar::Render(const VisorState& visorState)
{
    using namespace std::string_literals;
    ImGuiWindowFlags window_flags = (ImGuiWindowFlags_NoScrollbar |
                                     ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_MenuBar);
    // Pre-init button state if it changed by keyboard
    //SetButtonState(stopped, running, paused, );

    bool isChanged = false;
    float height = ImGui::GetFrameHeight();
    if(ImGui::BeginViewportSideBar("##MainStatusBar", NULL,
                                   ImGuiDir_Down, height,
                                   window_flags))
    {
        if(ImGui::BeginMenuBar())
        {
            double usedGPUMemMiB = visorState.tracer.usedGPUMemoryMiB;
            double totalGPUMemGiB = visorState.tracer.totalGPUMemoryMiB / 1024.0;
            std::string memUsage = fmt::format("{:.1f}MiB / {:.1f}GiB",
                                               usedGPUMemMiB, totalGPUMemGiB);

            ImGui::Text("%s", memUsage.c_str());
            ImGui::Separator();
            ImGui::Text("%s", (std::to_string(visorState.renderer.renderResolution[0]) + "x" +
                               std::to_string(visorState.renderer.renderResolution[1])).c_str());
            ImGui::Separator();
            ImGui::Text("%s", fmt::format("{:>7.3f}{:s}",
                                          visorState.renderer.throughput,
                                          visorState.renderer.throughputSuffix).c_str());
            ImGui::Separator();
            ImGui::Text("%s", (fmt::format("{:>6.0f}{:s}",
                                           visorState.renderer.workPerPixel,
                                           visorState.renderer.workPerPixelSuffix).c_str()));
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
            ImGui::Button(ICON_ICOMN_ARROW_LEFT);
            if(ImGui::IsItemHovered() && GImGui->HoveredIdTimer > 1)
            {
                ImGui::BeginTooltip();
                ImGui::Text("Prev Frame");
                ImGui::EndTooltip();
            }

            ImGui::Button(ICON_ICOMN_ARROW_RIGHT);
            if(ImGui::IsItemHovered() && GImGui->HoveredIdTimer > 1)
            {
                ImGui::BeginTooltip();
                ImGui::Text("Next Frame");
                ImGui::EndTooltip();
            }
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

    RunState newRunState = DetermineTracerState(stopped, running, paused);
    return (isChanged) ? Optional<RunState>(newRunState) : std::nullopt;
}

void VisorGUI::ShowFrameOverlay(bool& isOpen, const VisorState& visorState)
{
    static int location = 1;
    ImGuiWindowFlags window_flags = (ImGuiWindowFlags_NoDecoration |
                                     ImGuiWindowFlags_AlwaysAutoResize |
                                     ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_NoFocusOnAppearing |
                                     ImGuiWindowFlags_NoNav |
                                     ImGuiWindowFlags_NoMove);

    const float PADDING = 10.0f;
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
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
    ImGui::SetNextWindowBgAlpha(0.33f); // Transparent background

    if(ImGui::Begin("##VisorOverlay", &isOpen, window_flags))
    {
        ImGui::Text("Frame  : %.2f", visorState.visor.frameTime);
        ImGui::Text("Memory : %.2f", visorState.visor.usedGPUMemoryMiB);


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

void VisorGUI::ShowTopMenu(bool& isOpen, const VisorState& visorState)
{
    if(ImGui::BeginMainMenuBar())
    {
        if(tonemapperGUI) tonemapperGUI->Render();

        static constexpr const char* VISOR_INFO_NAME = "VisorInfo";
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x -
                        ImGui::CalcTextSize(VISOR_INFO_NAME).x -
                        ImGui::GetStyle().ItemSpacing.x * 3);
        ImGui::Separator();
        ImGui::ToggleButton(VISOR_INFO_NAME, fpsInfoOn);
        if(fpsInfoOn) ShowFrameOverlay(fpsInfoOn, visorState);

        ImGui::EndMainMenuBar();
    }
}

Optional<RunState> VisorGUI::ShowStatusBar(bool& isOpen,
                                           const VisorState& visorState)
{

    return statusBar.Render(visorState);
    // Acquire state
}

void VisorGUI::ShowMainImage()
{
    static const ImGuiWindowFlags flags = (ImGuiWindowFlags_NoDecoration |
                                           ImGuiWindowFlags_NoMove |
                                           ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoBackground |
                                           ImGuiWindowFlags_NoTitleBar |
                                           ImGuiWindowFlags_NoScrollbar |
                                           ImGuiWindowFlags_NoCollapse);

    // We demonstrate using the full viewport area or the work area"
    // (without menu-bars, task-bars etc.)
    // Based on your use case you may want one or the other.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    if(ImGui::Begin("MainWindow", nullptr, flags))
    {
        //ImGui::Image(std::bit_cast<ImTextureID>(mainImage), {256.0f, 256.0f});
    }
    ImGui::End();
}

void VisorGUI::Render(ImFont* windowScaledFont, const VisorState& visorState)
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::PushFont(windowScaledFont);

    if(ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_M))
        topBarOn = !topBarOn;
    if(ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_N))
        bottomBarOn = !bottomBarOn;

    ImGuiWindowFlags window_flags = (ImGuiWindowFlags_NoScrollbar |
                                     ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_MenuBar);
    if(topBarOn) ShowTopMenu(topBarOn, visorState);
    if(bottomBarOn) ShowStatusBar(bottomBarOn, visorState);

    ShowMainImage();

    ImGui::PopFont();
    ImGui::Render();
}

void VisorGUI::ChangeDisplayImage(const VulkanImage& img)
{
    if(mainImage) ImGui_ImplVulkan_RemoveTexture(mainImage);
    //
    mainImage = ImGui_ImplVulkan_AddTexture(img.Sampler(), img.View(),
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VisorGUI::ChangeTonemapperGUI(GUITonemapperI* newTonemapperGUI)
{
    tonemapperGUI = newTonemapperGUI;
}