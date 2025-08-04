#pragma once

#ifdef MRAY_GCC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wcpp"
#endif

#include <pxr/usd/usd/common.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdLux/tokens.h>

#ifdef MRAY_GCC
    #pragma GCC diagnostic pop
#endif

#include "Core/SceneLoaderI.h"
#include "Core/TracerI.h"
#include "Core/Expected.h"

class ThreadPool;

#ifndef PXR_STATIC
    #error PXR_STATIC must be defined!
#endif

// CTAD to the rescue!
static const std::array MRayEquivalentUSDTypes =
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
    ThreadPool&             threadPool;
    pxr::UsdStageRefPtr     loadedStage;
    //
    TracerIdPack            resultingIdPack;

    public:
                            SceneLoaderUSD(ThreadPool&);
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