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
    ImGuiKey key = ImGuiKey(keyMap->at(a));
    // TODO: ImGuiMod_Mask_ is probably internal? Check that is is stable etc.
    bool hasModifiers = ImGuiKey(key & ImGuiMod_Mask_) != ImGuiKey(0);
    if(hasModifiers)
        return ImGui::IsKeyChordPressed(key);
    else
        return ImGui::IsKeyPressed(key, repeat);
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