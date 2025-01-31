#pragma once

#include <BS/BS_thread_pool.hpp>

#include "MeshLoader/EntryPoint.h"
#include "Core/SceneLoaderI.h"
#include "Core/TracerI.h"
#include "Core/Flag.h"
#include "Core/Expected.h"

#include <pxr/usd/usd/common.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdLux/tokens.h>

// CTAD to the rescue!
static const std::array MRayEquavilentUSDTypes =
{
    // Comments are in MRayTerms
    // Primitives
    std::pair(pxr::UsdGeomTokens->Mesh,         "Triangle"),
    std::pair(pxr::UsdGeomTokens->Sphere,       "Sphere"),
    // Cameras
    std::pair(pxr::UsdGeomTokens->Camera,       "Pinhole"),
    // Lights
    std::pair(pxr::UsdLuxTokens->GeometryLight, "Primitive"),
    std::pair(pxr::UsdLuxTokens->DomeLight,     "Skysphere_Spherical"),
};

// Materials are little bit complex, it will be handled differently,
// One std surface is "USDPreviewSurface" it is mega shader-like
// material. We will try to bisect it. These are the lookups that will be
// used.
//
// Obviously this is static currently. We will need to make this dynamic later
// (by making MRay to inform us with material types etc.)
// This will require quite a bit of design to support stuff, but we'll see.
enum class MRayUSDMaterialType
{
    DIFFUSE,
    SPECULAR_DIFFUSE_COMBO,
    PURE_REFLECT,
    PURE_REFRACT
};
static const std::array MRayMaterialTypes =
{
    std::pair(MRayUSDMaterialType::DIFFUSE,                 "Lambert"),
    std::pair(MRayUSDMaterialType::SPECULAR_DIFFUSE_COMBO,  "Unreal"),
    std::pair(MRayUSDMaterialType::PURE_REFLECT,            "Reflect"),
    std::pair(MRayUSDMaterialType::PURE_REFRACT,            "Refract"),
};

class SceneLoaderUSD : public SceneLoaderI
{
    private:
    BS::thread_pool&        threadPool;
    pxr::UsdStageRefPtr     loadedStage;
    //
    TracerIdPack            resultingIdPack;

    public:
                            SceneLoaderUSD(BS::thread_pool&);
                            SceneLoaderUSD(const SceneLoaderUSD&) = delete;
    SceneLoaderUSD&         operator=(const SceneLoaderUSD&) = delete;

    // Scene Loading
    Expected<TracerIdPack>  LoadScene(TracerI& tracer,
                                      const std::string& filePath) override;
    Expected<TracerIdPack>  LoadScene(TracerI& tracer,
                                      std::istream& sceneData) override;

    void                    ClearScene() override;
};

//