#pragma once

#include <string_view>

namespace NodeNames
{
    using namespace std;
    using namespace literals;

    // Common Base Arrays
    static constexpr string_view CAMERA_LIST        = "Cameras"sv;
    static constexpr string_view LIGHT_LIST         = "Lights"sv;
    static constexpr string_view MEDIUM_LIST        = "Mediums"sv;
    static constexpr string_view TEXTURE_LIST       = "Textures"sv;
    static constexpr string_view TRANSFORM_LIST     = "Transforms"sv;
    static constexpr string_view PRIMITIVE_LIST     = "Primitives"sv;
    static constexpr string_view MATERIAL_LIST      = "Materials"sv;
    // Surfaces
    static constexpr string_view BOUNDARY               = "Boundary"sv;
    static constexpr string_view SURFACE_LIST           = "Surfaces"sv;
    static constexpr string_view LIGHT_SURFACE_LIST     = "LightSurfaces"sv;
    static constexpr string_view CAMERA_SURFACE_LIST    = "CameraSurfaces"sv;

    // Json Node Common Names
    static constexpr string_view ID             = "id"sv;
    static constexpr string_view TYPE           = "type"sv;
    static constexpr string_view FILE           = "file"sv;
    static constexpr string_view TAG            = "tag"sv;
    // Primitive Type related
    static constexpr string_view INNER_INDEX            = "innerIndex"sv;
    static constexpr string_view NODE_PRIM_TRI          = "nodeTriangle"sv;
    static constexpr string_view NODE_PRIM_TRI_INDEXED  = "nodeTriangleIndexed"sv;
    static constexpr string_view NODE_PRIM_SPHERE       = "nodeSphere"sv;

    // Common Names
    static constexpr string_view POSITION       = "position"sv;
    static constexpr string_view DATA           = "data"sv;
    // Surface Related Names
    static constexpr string_view TRANSFORM      = "transform"sv;
    static constexpr string_view PRIMITIVE      = "primitive"sv;
    static constexpr string_view MATERIAL       = "material"sv;
    static constexpr string_view ALPHA_MAP      = "alphaMap"sv;
    static constexpr string_view CULL_FACE      = "cullBackFace"sv;
    // Material & Light Common Names
    static constexpr string_view MEDIUM_FRONT   = "mediumFront"sv;
    static constexpr string_view MEDIUM_BACK    = "mediumBack"sv;
    static constexpr string_view MEDIUM         = "medium"sv;
    static constexpr string_view LIGHT          = "light"sv;
    static constexpr string_view CAMERA         = "camera"sv;
    // Texture Related Names
    static constexpr string_view TEXTURE_CHANNEL    = "channels"sv;
    static constexpr string_view TEXTURE_NAME       = "texture"sv;
    // Light Related
    static constexpr string_view LIGHT_TYPE_PRIMITIVE       = "Primitive"sv;
    static constexpr string_view TRANSFORM_TYPE_IDENTITY    = "Identity"sv;

    // Texture Node Related Names
    static constexpr string_view TEX_NODE_FILE          = "file"sv;
    static constexpr string_view TEX_NODE_AS_SIGNED     = "asSigned"sv;
    static constexpr bool TEX_NODE_AS_SIGNED_DEFAULT    = false;
    static constexpr string_view TEX_NODE_IS_COLOR      = "isColor"sv;
    static constexpr bool TEX_NODE_IS_COLOR_DEFAULT     = true;
    static constexpr string_view TEX_NODE_IGNORE_CLAMP  = "ignoreResClamp"sv;
    static constexpr bool TEX_NODE_IGNORE_CLAMP_DEFAULT = false;
    static constexpr string_view TEX_NODE_EDGE_RESOLVE  = "edgeResolve"sv;
    static constexpr string_view TEX_NODE_INTERPOLATION = "interpolation"sv;
    static constexpr string_view TEX_NODE_COLOR_SPACE   = "colorSpace"sv;
    static constexpr string_view TEX_NODE_GAMMA         = "gamma"sv;
    static constexpr string_view TEX_NODE_READ_MODE     = "readMode";
    static constexpr string_view TEX_NODE_IS_3D         = "is3D";
    static constexpr bool TEX_NODE_IS_3D_DEFAULT        = false;
    static constexpr string_view TEX_NODE_CHANNELS      = "channels";

    //// Identity Transform Type Name
    //static constexpr string_view TRANSFORM_IDENTITY     = "Identity"sv;
}