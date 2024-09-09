#pragma once

#include <Imgui/imgui.h>
#include <map>
#include <algorithm>

#include "Common/AnalyticStructs.h"
#include "Core/Quaternion.h"

#include "VisorI.h"
#include "VisorState.h"
#include "InputChecker.h"

class MovementSchemeI
{
    public:
    virtual                             ~MovementSchemeI() = default;
    // Interface
    virtual Optional<CameraTransform>   Update(const InputChecker&,
                                               const VisorState&) = 0;
};

class MovementSchemeFPS : public MovementSchemeI
{
    private:
    // TODO: Make these user defined later
    static constexpr Float  Sensitivity = Float(0.0025);
    static constexpr Float  MovementRatio = Float(1.0 / 20.0);
    static constexpr Float  MoveRatioModifier = Float(1.5);
    //
    Vector2                 prevMouse = Vector2::Zero();
    bool                    fastMove = false;

    public:
    // Members
    Optional<CameraTransform>   Update(const InputChecker&,
                                       const VisorState&) override;
};

inline Optional<CameraTransform> MovementSchemeFPS::Update(const InputChecker& inputChecker,
                                                           const VisorState& state)
{
    Optional<CameraTransform> result;

    Vector2 currMouse = inputChecker.GetMousePos();
    Vector2 diff = currMouse - prevMouse;

    // Mouse related
    if(inputChecker.CheckMouseDrag(VisorUserAction::MOUSE_ROTATE_MODIFIER) &&
       diff != Vector2::Zero())
    {
        result = state.transform;
        CameraTransform& transform = result.value();
        // X Rotation
        Vector3 lookDir = transform.gazePoint - transform.position;
        Quaternion rotateX(-diff[0] * Sensitivity, Vector3::YAxis());
        Vector3 rotated = rotateX.ApplyRotation(lookDir);
        transform.gazePoint = transform.position + rotated;

        // Y Rotation
        lookDir = transform.gazePoint - transform.position;
        Vector3 side = Vector3::Cross(transform.up, lookDir).NormalizeSelf();
        Quaternion rotateY(diff[1] * Sensitivity, side);
        rotated = rotateY.ApplyRotation((lookDir));
        transform.gazePoint = transform.position + rotated;

        // Redefine up
        // Enforce an up vector which is orthogonal to the xz plane
        transform.up = Vector3::Cross(rotated, side);
        transform.up[0] = 0.0f;
        transform.up[1] = (transform.up[1] < 0.0f) ? -1.0f : 1.0f;
        transform.up[2] = 0.0f;
    }

    // Key Related
    if(inputChecker.CheckKeyPress(VisorUserAction::FAST_MOVE_MODIFIER))
        fastMove = true;
    if(inputChecker.CheckKeyRelease(VisorUserAction::FAST_MOVE_MODIFIER))
        fastMove = false;

    const std::array<bool, 4> MovePresses =
    {
        inputChecker.CheckKeyPress(VisorUserAction::MOVE_FORWARD, true),
        inputChecker.CheckKeyPress(VisorUserAction::MOVE_BACKWARD, true),
        inputChecker.CheckKeyPress(VisorUserAction::MOVE_LEFT, true),
        inputChecker.CheckKeyPress(VisorUserAction::MOVE_RIGHT, true)
    };
    if(std::any_of(MovePresses.cbegin(), MovePresses.cend(),
                   [](auto a) { return a; }))
    {
        Vector3 sceneSpan = state.scene.sceneExtent.GeomSpan();
        uint32_t maxIndex = sceneSpan.Maximum();
        Float maxExtent = sceneSpan[maxIndex];
        Float movementRatio = maxExtent * MovementRatio;
        if(fastMove)
            movementRatio *= MoveRatioModifier;

        if(!result) result = state.transform;
        CameraTransform& transform = result.value();

        Vector3 lookDir = (transform.gazePoint - transform.position).NormalizeSelf();
        Vector3 side = Vector3::Cross(transform.up, lookDir).NormalizeSelf();
        if(MovePresses[0])
        {
            transform.position += lookDir * movementRatio;
            transform.gazePoint += lookDir * movementRatio;
        }
        if(MovePresses[1])
        {
            transform.position -= lookDir * movementRatio;
            transform.gazePoint -= lookDir * movementRatio;
        }
        if(MovePresses[2])
        {
            transform.position += side * movementRatio;
            transform.gazePoint += side *movementRatio;
        }
        if(MovePresses[3])
        {
            transform.position -= side * movementRatio;
            transform.gazePoint -= side * movementRatio;
        }
    }
    prevMouse = currMouse;
    return result;
}