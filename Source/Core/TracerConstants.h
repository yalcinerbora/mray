#pragma once

#include <string_view>

namespace TracerConstants
{
    // This is utilized by static vectors to evade heap,
    // this is currently "small" 65K x 65K textures are max
    static constexpr size_t MaxTextureMipCount = 16;
    // Primitive/Material pairs can be batched and a single
    // accelerator is constructed per batch this is the upper limit
    // of such pairs
    static constexpr size_t MaxPrimBatchPerSurface = 8;
    // Each "group" can have at most N different attributes
    // (i.e., a triangle primitive's "position" "normal" "uv" etc.)
    static constexpr size_t MaxAttributePerGroup = 16;
    // Same as above but for renderer
    static constexpr size_t MaxRendererAttributeCount = 32;
    // Renderer can define at most N work per Mat/Prim/Transform
    // triplet. Most of the time single work definition is enough.
    // but some renderers (path tracer renderer) may define multiple
    // works, and change the work logic according to input parameters.
    static constexpr size_t MaxRenderWorkPerTriplet = 4;
    // Accelerator report hits as barycentric coordinates for triangles,
    // and spherical coordinates for spheres. So two is currently enough.
    // Volume hits will require 3 (local space x, y, z) for hits probably
    // but these can be generated from position.
    // In future, there may be different primitives require more that two
    // hit parametrization (This should rarely be an issue since surfaces
    // are inherently 2D), and this may be changed
    static constexpr size_t MaxHitFloatCount = 2;
    // Maximum camera size, a renderer will allocate this
    // much memory, and camera should fit into this.
    // This will be compile time checked.
    static constexpr size_t MaxCameraInstanceByteSize = 512;
    // Maximum nested volumes allowed by the renderer
    // This is used for light/camera surfaces
    // when these are nested inside a volume
    // These are unfortunately authored.
    static constexpr size_t MaxNestedVolumes = 8;

    static constexpr std::string_view IdentityTransName  = "(T)Identity";
    static constexpr std::string_view NullLightName      = "(L)Null";
    static constexpr std::string_view EmptyPrimName      = "(P)Empty";
    static constexpr std::string_view VacuumMediumName   = "(Md)Vacuum";
    static constexpr std::string_view PassthroughMatName = "(Mt)Passthrough";

    static constexpr std::string_view LIGHT_PREFIX      = "(L)";
    static constexpr std::string_view TRANSFORM_PREFIX  = "(T)";
    static constexpr std::string_view PRIM_PREFIX       = "(P)";
    static constexpr std::string_view MAT_PREFIX        = "(Mt)";
    static constexpr std::string_view CAM_PREFIX        = "(C)";
    static constexpr std::string_view MEDIUM_PREFIX     = "(Md)";
    static constexpr std::string_view ACCEL_PREFIX      = "(A)";
    static constexpr std::string_view RENDERER_PREFIX   = "(R)";
    static constexpr std::string_view WORK_PREFIX       = "(W)";
}
