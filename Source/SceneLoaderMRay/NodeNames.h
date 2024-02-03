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
    static constexpr string_view NAME           = "name"sv;
    static constexpr string_view TAG            = "tag"sv;
    // Primitive Type related
    static constexpr string_view INNER_INDEX            = "innerIndex"sv;
    static constexpr string_view NODE_PRIMITIVE         = ".nodeTriangle"sv;
    static constexpr string_view INDEXED_NODE_PRIMITIVE = ".nodeTriangleIndexed"sv;

    // Common Names
    static constexpr string_view POSITION       = "position"sv;
    static constexpr string_view DATA           = "data"sv;
    // Surface Related Names
    static constexpr string_view TRANSFORM      = "transform"sv;
    static constexpr string_view PRIMITIVE      = "primitive"sv;
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