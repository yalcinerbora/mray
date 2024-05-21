#pragma once

#include <Imgui/imgui.h>
#include <map>

#include "VisorI.h"
#include "Common/AnalyticStructs.h"

class MovementSchemeI
{
    public:
    virtual                             ~MovementSchemeI() = default;
    // Interface
    virtual Optional<CameraTransform>   Update() = 0;
    virtual void                        ResetTransform(const CameraTransform&) = 0;
    virtual void                        ChangeSceneExtent(const AABB3& sceneExtent) = 0;
};

class MovementSchemeFPS : public MovementSchemeI
{
    private:
    static constexpr float  DefaultSensitivity = 0.0025f;
    static constexpr float  DefaultMovementRatio = 1.0f / 50.0f;
    static constexpr float  DefaultMoveRatioModifier = 1.5f;

    const std::map<VisorUserAction, ImGuiKey>& keyMap;
    CameraTransform currentTransform;

    public:
    // Constructors & Destructor
    MovementSchemeFPS(const std::map<VisorUserAction, ImGuiKey>&);

    // Members
    Optional<CameraTransform>   Update() override;
    void                        ResetTransform(const CameraTransform&) override;
    void                        ChangeSceneExtent(const AABB3& sceneExtent) override;
};

inline MovementSchemeFPS::MovementSchemeFPS(const std::map<VisorUserAction, ImGuiKey>& km)
    : keyMap(km)
{}

Optional<CameraTransform> MovementSchemeFPS::Update()
{
    ImVec2 v = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);

    //auto* viewport = ImGui::GetMainViewport();
}

void MovementSchemeFPS::ResetTransform(const CameraTransform& t)
{
    currentTransform = t;
}

void MovementSchemeFPS::ChangeSceneExtent(const AABB3& sceneExtent)
{

}