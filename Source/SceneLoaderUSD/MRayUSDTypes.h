#pragma once

#include <vector>
#include <variant>
#include <map>

#include <pxr/usd/usd/prim.h>

#include "Core/Definitions.h"
#include "Core/Matrix.h"

struct MRayUSDPrimSurface
{
    using SubGeomMaterials = std::vector<std::pair<uint32_t, pxr::SdfPath>>;
    //
    pxr::UsdPrim        surfacePrim;
    pxr::UsdPrim        uniquePrim;
    Matrix4x4           surfaceTransform;
    SubGeomMaterials    subGeometryMaterialKeys;
};

struct CollapsedPrims
{
    std::vector<MRayUSDPrimSurface> geomLightSurfaces;
    std::vector<MRayUSDPrimSurface> surfaces;
    std::set<pxr::UsdPrim>          uniquePrims;
};

struct MRayUSDFallbackMaterial
{
    pxr::GfVec3f color;
    // TODO: As a sophisticated platform, I did expect usd to have color space for
    // display color. But it does not. This stays here to ease of change
    MRayColorSpaceEnum colorSpace = MRayColorSpaceEnum::MR_DEFAULT;
};

using MRayUSDBoundMaterial = std::variant<pxr::UsdPrim, MRayUSDFallbackMaterial>;
using MRayUSDMaterialMap = std::map<pxr::UsdPrim, MRayUSDBoundMaterial>;

