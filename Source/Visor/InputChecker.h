#pragma once

#include "VisorI.h"
#include <Imgui/imgui.h>

class InputChecker
{
    private:
    const VisorKeyMap* keyMap;

    public:
    // Constructors & Destructor
                InputChecker(const VisorKeyMap& keyMap);
    //
    bool        CheckKeyPress(VisorUserAction, bool repeat = false) const;
    bool        CheckKeyRelease(VisorUserAction) const;
    bool        CheckMouseDrag(VisorUserAction) const;
    Vector2     GetMousePos() const;
};

inline InputChecker::InputChecker(const VisorKeyMap& km)
    : keyMap(&km)
{}

inline bool InputChecker::CheckKeyPress(VisorUserAction a, bool repeat) const
{
    return ImGui::IsKeyPressed(ImGuiKey(keyMap->at(a)), repeat);
}

inline bool InputChecker::CheckKeyRelease(VisorUserAction a) const
{
    return ImGui::IsKeyReleased(ImGuiKey(keyMap->at(a)));
}

inline bool InputChecker::CheckMouseDrag(VisorUserAction a) const
{
    return ImGui::IsMouseDragging(keyMap->at(a));
}

inline Vector2 InputChecker::GetMousePos() const
{
    auto v = ImGui::GetMousePos();
    return Vector2(v.x, v.y);
}