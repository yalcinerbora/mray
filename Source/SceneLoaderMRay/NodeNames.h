#pragma once

#include <string_view>

namespace NodeNames
{
    using namespace std::string_view_literals;

    // Common Base Arrays
    static constexpr auto CAMERA_LIST    = "Cameras"sv;
    static constexpr auto LIGHT_LIST     = "Lights"sv;
    static constexpr auto MEDIUM_LIST    = "Mediums"sv;
    static constexpr auto TEXTURE_LIST   = "Textures"sv;
    static constexpr auto TRANSFORM_LIST = "Transforms"sv;
    static constexpr auto PRIMITIVE_LIST = "Primitives"sv;
    static constexpr auto MATERIAL_LIST  = "Materials"sv;
    // Surfaces
    static constexpr auto BOUNDARY              = "Boundary"sv;
    static constexpr auto SURFACE_LIST          = "Surfaces"sv;
    static constexpr auto LIGHT_SURFACE_LIST    = "LightSurfaces"sv;
    static constexpr auto CAMERA_SURFACE_LIST   = "CameraSurfaces"sv;

    // Json Node Common Names
    static constexpr auto ID    = "id"sv;
    static constexpr auto TYPE  = "type"sv;
    static constexpr auto FILE  = "file"sv;
    static constexpr auto TAG   = "tag"sv;
    // Primitive Type related
    static constexpr auto INNER_INDEX           = "innerIndex"sv;
    static constexpr auto NODE_PRIM_TRI         = "nodeTriangle"sv;
    static constexpr auto NODE_PRIM_TRI_INDEXED = "nodeTriangleIndexed"sv;
    static constexpr auto NODE_PRIM_SPHERE      = "nodeSphere"sv;

    // Common Names
    static constexpr auto POSITION      = "position"sv;
    static constexpr auto DATA          = "data"sv;
    // Surface Related Names
    static constexpr auto TRANSFORM     = "transform"sv;
    static constexpr auto PRIMITIVE     = "primitive"sv;
    static constexpr auto MATERIAL      = "material"sv;
    static constexpr auto ALPHA_MAP     = "alphaMap"sv;
    static constexpr auto CULL_FACE     = "cullBackFace"sv;
    // Material & Light Common Names
    static constexpr auto MEDIUM_FRONT  = "mediumFront"sv;
    static constexpr auto MEDIUM_BACK   = "mediumBack"sv;
    static constexpr auto MEDIUM        = "medium"sv;
    static constexpr auto LIGHT         = "light"sv;
    static constexpr auto CAMERA        = "camera"sv;
    // Texture Related Names
    static constexpr auto TEXTURE_CHANNEL   = "channels"sv;
    static constexpr auto TEXTURE_NAME      = "texture"sv;
    // Light Related
    static constexpr auto LIGHT_TYPE_PRIMITIVE      = "Primitive"sv;
    static constexpr auto TRANSFORM_TYPE_IDENTITY   = "Identity"sv;

    // Texture Node Related Names
    static constexpr auto TEX_NODE_FILE                 = "file"sv;
    static constexpr auto TEX_NODE_AS_SIGNED            = "asSigned"sv;
    static constexpr bool TEX_NODE_AS_SIGNED_DEFAULT    = false;
    static constexpr auto TEX_NODE_IS_COLOR             = "isColor"sv;
    static constexpr bool TEX_NODE_IS_COLOR_DEFAULT     = true;
    static constexpr auto TEX_NODE_IGNORE_CLAMP         = "ignoreResClamp"sv;
    static constexpr bool TEX_NODE_IGNORE_CLAMP_DEFAULT = false;
    static constexpr auto TEX_NODE_EDGE_RESOLVE         = "edgeResolve"sv;
    static constexpr auto TEX_NODE_INTERPOLATION        = "interpolation"sv;
    static constexpr auto TEX_NODE_COLOR_SPACE          = "colorSpace"sv;
    static constexpr auto TEX_NODE_GAMMA                = "gamma"sv;
    static constexpr auto TEX_NODE_READ_MODE            = "readMode"sv;
    static constexpr auto TEX_NODE_IS_3D                = "is3D"sv;
    static constexpr bool TEX_NODE_IS_3D_DEFAULT        = false;
    static constexpr auto TEX_NODE_CHANNELS             = "channels"sv;
}