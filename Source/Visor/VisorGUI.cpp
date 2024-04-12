#include "VisorGUI.h"

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

Optional<RunState> MainStatusBar::Render(const TracerAnalyticData& tracerData,
                                         const SceneAnalyticData& sceneData)
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
            double usedGPUMemMiB = tracerData.usedGPUMemoryMiB;
            double totalGPUMemGiB = tracerData.totalGPUMemoryMiB / 1024.0;
            std::string memUsage = fmt::format("{:.1f}MiB / {:.1f}GiB",
                                               usedGPUMemMiB, totalGPUMemGiB);

            ImGui::Text("%s", memUsage.c_str());
            ImGui::Separator();
            ImGui::Text("%s", (std::to_string(tracerData.renderResolution[0]) + "x" +
                               std::to_string(tracerData.renderResolution[1])).c_str());
            ImGui::Separator();
            ImGui::Text("%s", fmt::format("{:>7.3f}{:s}",
                                          tracerData.throughput,
                                          tracerData.throughputSuffix).c_str());
            ImGui::Separator();
            ImGui::Text("%s", (fmt::format("{:>6.0f}{:s}",
                                           tracerData.workPerPixel,
                                           tracerData.workPerPixelSuffix).c_str()));
            ImGui::Separator();

            std::string prefix = std::string(RENDERING_NAME);
            std::string body = (prefix + " " + sceneData.sceneName + "...");
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

// Demonstrate creating a window covering the entire screen/viewport
static void ShowExampleAppFullscreen(bool* p_open)
{
    static bool use_work_area = true;
    static ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;

    // We demonstrate using the full viewport area or the work area (without menu-bars, task-bars etc.)
    // Based on your use case you may want one or the other.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(use_work_area ? viewport->WorkPos : viewport->Pos);
    ImGui::SetNextWindowSize(use_work_area ? viewport->WorkSize : viewport->Size);

    if(ImGui::Begin("Example: Fullscreen window", p_open, flags))
    {
        ImGui::Checkbox("Use work area instead of main area", &use_work_area);

        ImGui::CheckboxFlags("ImGuiWindowFlags_NoBackground", &flags, ImGuiWindowFlags_NoBackground);
        ImGui::CheckboxFlags("ImGuiWindowFlags_NoDecoration", &flags, ImGuiWindowFlags_NoDecoration);
        ImGui::Indent();
        ImGui::CheckboxFlags("ImGuiWindowFlags_NoTitleBar", &flags, ImGuiWindowFlags_NoTitleBar);
        ImGui::CheckboxFlags("ImGuiWindowFlags_NoCollapse", &flags, ImGuiWindowFlags_NoCollapse);
        ImGui::CheckboxFlags("ImGuiWindowFlags_NoScrollbar", &flags, ImGuiWindowFlags_NoScrollbar);
        ImGui::Unindent();

        if(p_open && ImGui::Button("Close this window"))
            *p_open = false;
    }
    ImGui::End();
}

void VisorGUI::Render(ImFont* windowScaledFont)
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
    if(topBarOn)
    {
        if(ImGui::BeginViewportSideBar("##MenuBar", NULL,
                                       ImGuiDir_Up,
                                       ImGui::GetFrameHeight(),
                                       window_flags))
        {
            if(ImGui::BeginMenuBar())
            {
                if(ImGui::Button("Tone Mapping"))
                {
                    //tmWindow.ToggleWindowOpen();
                }
                ImGui::EndMenuBar();
            }
        }
        ImGui::End();
    }

    if(bottomBarOn)
    {
        TracerAnalyticData ad =
        {
            .throughput = 20.2339494,
            .throughputSuffix = "MRays/sec",
            .workPerPixel = 1.2,
            .workPerPixelSuffix = "spp",
            .iterationTimeMS = 10.123131,
            .totalGPUMemoryMiB = 8000,
            .usedGPUMemoryMiB = 500,
            .renderResolution = {3840, 2160}
        };

        SceneAnalyticData sad =
        {
            .sceneName = "Kitchen.json",
            .sceneLoadTime = 10.2,
            .sceneUpdateTime = 0.3,
            .groupCounts = {},
            .accKeyMax = {0, 0},
            .workKeyMax = {0, 0}
        };

        Optional<RunState> newState = statusBar.Render(ad, sad);
        // Acquire state
    }

    /*ImGui::*/

    ImGui::ShowDemoWindow();
    //static bool isOpen = true;
    //ShowExampleAppFullscreen(&isOpen);


    // IssueCommand
    // -->

    // Rendering
    // ---------
    static int i = 0;
    MRAY_LOG("Rendering Frame!{}", i++);

    ImGui::PopFont();
    ImGui::Render();
}
