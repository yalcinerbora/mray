#pragma once

#include <vector>
#include <map>

#include <pxr/usd/usd/prim.h>

#include "Core/Definitions.h"
#include "Core/Matrix.h"        // IWYU pragma: keep
#include "Core/Variant.h"

struct MRayUSDPrimSurface
{
    using SubGeomMaterials = std::vector<Pair<uint32_t, pxr::SdfPath>>;
    //
    bool                cullFace = false;
    pxr::UsdPrim        surfacePrim;
    pxr::UsdPrim        uniquePrim;
    Optional<Matrix4x4> surfaceTransform;
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

using MRayUSDBoundMaterial = Variant<pxr::UsdPrim, MRayUSDFallbackMaterial>;
using MRayUSDMaterialMap = std::map<pxr::UsdPrim, MRayUSDBoundMaterial>;

