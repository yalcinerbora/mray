#pragma once

#include <string_view>

namespace NodeNames
{
    using namespace std;
    using namespace literals;

    static constexpr string_view SCENE_EXT      = "mscene"sv;
    static constexpr string_view ANIM_EXT       = "manim"sv;
    // Common Base Arrays
    static constexpr string_view CAMERA_BASE                = "Cameras"sv;
    static constexpr string_view LIGHT_BASE                 = "Lights"sv;
    static constexpr string_view MEDIUM_BASE                = "Mediums"sv;
    static constexpr string_view TEXTURE_BASE               = "Textures"sv;
    static constexpr string_view ACCELERATOR_BASE           = "Accelerators"sv;
    static constexpr string_view TRANSFORM_BASE             = "Transforms"sv;
    static constexpr string_view PRIMITIVE_BASE             = "Primitives"sv;
    static constexpr string_view MATERIAL_BASE              = "Materials"sv;
    static constexpr string_view SURFACE_DATA_BASE          = "SurfaceData"sv;
    // Boundaries
    static constexpr string_view BASE_ACCELERATOR           = "BaseAccelerator"sv;
    static constexpr string_view BASE_BOUNDARY_LIGHT        = "BaseBoundaryLight"sv;
    static constexpr string_view BASE_MEDIUM                = "BaseMedium"sv;
    static constexpr string_view BASE_BOUNDARY_TRANSFORM    = "BaseBoundaryTransform"sv;

    static constexpr string_view SURFACE_BASE               = "Surfaces"sv;
    static constexpr string_view LIGHT_SURFACE_BASE         = "LightSurfaces"sv;
    static constexpr string_view CAMERA_SURFACE_BASE        = "CameraSurfaces"sv;

    // Common Names
    static constexpr string_view ID             = "id"sv;
    static constexpr string_view TYPE           = "type"sv;
    static constexpr string_view NAME           = "name"sv;
    static constexpr string_view TAG            = "tag"sv;
    // Common Names
    static constexpr string_view POSITION       = "position"sv;
    static constexpr string_view DATA           = "data"sv;
    // Surface Related Names
    static constexpr string_view TRANSFORM      = "transform"sv;
    static constexpr string_view PRIMITIVE      = "primitive"sv;
    static constexpr string_view ACCELERATOR    = "accelerator"sv;
    static constexpr string_view MATERIAL       = "material"sv;
    // Material & Light Common Names
    static constexpr string_view MEDIUM         = "medium"sv;
    static constexpr string_view LIGHT          = "light"sv;
    static constexpr string_view CAMERA         = "camera"sv;
    // Texture Related Names
    static constexpr string_view TEXTURE_IS_CACHED  = "isCached"sv;
    static constexpr string_view TEXTURE_FILTER     = "filter"sv;
    static constexpr string_view TEXTURE_FILE       = "file"sv;
    static constexpr string_view TEXTURE_SIGNED     = "signed"sv;
    static constexpr string_view TEXTURE_MIPMAP     = "generateMipmaps"sv;
    static constexpr string_view TEXTURE_CHANNEL    = "channels"sv;
    static constexpr string_view TEXTURE_NAME       = "texture"sv;
    // Light Related
    static constexpr string_view LIGHT_TYPE_PRIMITIVE   = "Primitive"sv;
    // Identity Transform Type Name
    static constexpr string_view TRANSFORM_IDENTITY     = "Identity"sv;
}