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