#include "SceneLoaderUSD.h"

#include "MRayUSDTypes.h"
#include "MeshProcessor.h"
#include "MaterialProcessor.h"

#include "Core/TracerI.h"
#include "Core/Log.h"
#include "Core/Timer.h"
#include "Core/TypeNameGenerators.h"
#include "Core/Error.hpp"
#include "Core/Algorithm.h"

#include "ImageLoader/EntryPoint.h"
// Traversal
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/xformCache.h>
// Supported Geom Schemas
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/camera.h>
// Materials
#include <pxr/usd/usdShade/materialBindingAPI.h>
// Lights
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/lightAPI.h>
// Error Related
#include <pxr/base/tf/errorMark.h>

using namespace TypeNameGen::Runtime;

Matrix4x4 ConvertToMRayMatrix(const pxr::GfMatrix4d& matIn)
{
    // USD says their matrix is row-major order but their vectors are
    // row vectors I think. Because the matrix translation portion is on bottom row.
    // So we transpose
    return Matrix4x4(Span<const double, 16>(matIn.data(), 16)).Transpose();
}

void PrintPrims(const CollapsedPrims& meshMatPrims,
                const CollapsedPrims& sphereMatPrims,
                const MRayUSDMaterialMap& uniqueMaterials,
                //
                Optional<MRayUSDPrimSurface> domeLight,
                //
                const std::vector<MRayUSDPrimSurface>& cameras,
                const pxr::UsdStageRefPtr& loadedStage)
{
    auto PrintPrimSurface = [&](const std::vector<MRayUSDPrimSurface>& primSurf)
    {
        for(const auto& s : primSurf)
        {
            MRAY_LOG("{}", s.surfacePrim.GetPath().GetString());
            for(const auto& [subGeoIndex, mk] : s.subGeometryMaterialKeys)
            {
                pxr::UsdPrim matKey = loadedStage->GetPrimAtPath(mk);
                const auto& material = uniqueMaterials.at(matKey);
                if(std::holds_alternative<pxr::UsdPrim>(material))
                    MRAY_LOG("    {} | {}", subGeoIndex,
                             std::get<pxr::UsdPrim>(material).GetPath().GetString());
                else
                {
                    auto fallbackMat = std::get<MRayUSDFallbackMaterial>(material);
                    MRAY_LOG("    {} | [[{}, {}, {}], {}]",  subGeoIndex,
                             fallbackMat.color[0], fallbackMat.color[1], fallbackMat.color[2],
                             MRayColorSpaceStringifier::ToString(fallbackMat.colorSpace));
                }
            }
        }
    };

    static constexpr auto HEADER = "//=====================//\n"
                                   "//{: ^21}//\n"
                                   "//=====================//";
    //
    MRAY_LOG(HEADER, "Meshes");
    PrintPrimSurface(meshMatPrims.surfaces);
    //
    MRAY_LOG(HEADER, "Spheres");
    PrintPrimSurface(sphereMatPrims.surfaces);
    //
    MRAY_LOG(HEADER, "Materials");
    for(const auto& [_, mat] : uniqueMaterials)
    {
        if(std::holds_alternative<pxr::UsdPrim>(mat))
            MRAY_LOG("[M]: {}", std::get<pxr::UsdPrim>(mat).GetPath().GetString());
        else
        {
            MRayUSDFallbackMaterial fallbackMat = std::get<MRayUSDFallbackMaterial>(mat);
            MRAY_LOG("[F]: [{}, {}, {}] | {}",
                     fallbackMat.color[0], fallbackMat.color[1], fallbackMat.color[2],
                     MRayColorSpaceStringifier::ToString(fallbackMat.colorSpace));
        }
    }
    //
    MRAY_LOG(HEADER, "Mesh Lights");
    PrintPrimSurface(meshMatPrims.geomLightSurfaces);
    //
    MRAY_LOG(HEADER, "Sphere Lights");
    PrintPrimSurface(sphereMatPrims.geomLightSurfaces);
    //
    MRAY_LOG(HEADER, "Dome Light");
    if(domeLight)
        MRAY_LOG("{}", domeLight->surfacePrim.GetPath().GetString());
    //
    MRAY_LOG(HEADER, "Cameras");
    for(const auto& cam : cameras)
        MRAY_LOG("{}", cam.surfacePrim.GetPath().GetString());
}

void ExpandGeomsAndFindMaterials(CollapsedPrims& subGeomPack,
                                 MRayUSDMaterialMap& uniqueMaterials,
                                 Span<MRayUSDPrimSurface> surfacesRange)
{
    bool warnLightIndependent = false;
    bool warnLightMaterialTint = false;
    bool warnMaterialDisplayColorVarying = false;
    bool warnMaterialDisplayNotFound = false;
    bool warnUsingDisplayColor = false;
    bool warnUnsupportedMaterial = false;
    bool warnEdgeSubset = false;

    // TODO: It is not avail on the token pool (UsdShadeTokens)?
    // Check later.
    pxr::TfToken usdPrevSurfToken = pxr::TfToken("UsdPreviewSurface");

    auto BindMaterial = [&](MRayUSDPrimSurface& surface,
                            pxr::UsdShadeMaterial boundMaterial,
                            uint32_t subGeomIndex = std::numeric_limits<uint32_t>::max())
    {
        pxr::TfToken shaderId;
        boundMaterial.ComputeSurfaceSource().GetShaderId(&shaderId);
        // Check if the material is usd preview surface, if not fallback to
        // display color stuff
        if(boundMaterial && shaderId == usdPrevSurfToken)
        {
            pxr::UsdPrim matPrim = boundMaterial.GetPrim();
            surface.subGeometryMaterialKeys.emplace_back(subGeomIndex, matPrim.GetPath());
            if(matPrim.IsInstanceProxy())
                matPrim = matPrim.GetPrimInPrototype();
            uniqueMaterials.emplace(matPrim, matPrim);
            return;
        }
        else warnUnsupportedMaterial = true;

        warnUsingDisplayColor = true;
        using DisplayColorUSD = pxr::VtArray<pxr::GfVec3f>;

        // No material found, fallback to display color
        // iff is constant, if not do a middle gray
        MRayUSDFallbackMaterial fallbackMat = {};
        pxr::UsdGeomGprim gPrim(surface.surfacePrim);
        pxr::UsdGeomPrimvar displayColor = gPrim.GetDisplayColorPrimvar();
        if(!displayColor.HasValue())
        {
            // Warn about no display color,
            warnMaterialDisplayNotFound = true;
            fallbackMat.color = pxr::GfVec3f(0.5);
        }
        else if(displayColor.GetInterpolation() != pxr::UsdGeomTokens->constant)
        {
            // Warn about display color is not constant
            warnMaterialDisplayColorVarying = true;
        }
        DisplayColorUSD displayColorValue;
        displayColor.Get<DisplayColorUSD>(&displayColorValue);
        fallbackMat.color = displayColorValue.AsConst()[0];

        // Here we key the material via the instance proxy, assuming
        // Instance proxies can be override the instance prim's display value
        surface.subGeometryMaterialKeys.emplace_back(subGeomIndex, surface.surfacePrim.GetPath());
        uniqueMaterials.emplace(surface.surfacePrim, fallbackMat);
    };

    using MatUSD = pxr::UsdShadeMaterial;
    using MatBindAPI = pxr::UsdShadeMaterialBindingAPI;
    MatBindAPI::BindingsCache bindingsCache;
    MatBindAPI::CollectionQueryCache collQueryCache;
    for(const MRayUSDPrimSurface& surface : surfacesRange)
    {
        const auto& surfacePrim = surface.surfacePrim;
        auto gPrim = pxr::UsdGeomGprim(surfacePrim);
        auto subsets = pxr::UsdGeomSubset::GetAllGeomSubsets(gPrim);

        // I give up, fatally crash if you see this shit
        bool hasEdgeSubset = std::any_of(subsets.cbegin(), subsets.cend(),
                                         [](const pxr::UsdGeomSubset& geomSubset)
        {
            pxr::TfToken elemType;
            geomSubset.GetElementTypeAttr().Get(&elemType);
            // TODO: Wtf are these? Maybe for point clouds etc.
            return (elemType == pxr::UsdGeomTokens->edge ||
                    elemType == pxr::UsdGeomTokens->point);
        });
        // We can't recover from this, so skip this mesh
        if(hasEdgeSubset)
        {
            warnEdgeSubset = true;
            continue;
        }

        // Check the light API to filter the light
        const pxr::UsdLuxLightAPI lightAPI(surface.surfacePrim);
        if(lightAPI)
        {
            pxr::TfToken lightBindMode;
            lightAPI.GetMaterialSyncModeAttr().Get(&lightBindMode);
            if(lightBindMode == pxr::UsdLuxTokens->independent)
            {
                // Warn about "independent" is not supported
                warnLightIndependent = true;
            }
            else if(lightBindMode == pxr::UsdLuxTokens->materialGlowTintsLight)
            {
                // Warn about "materialGlowTintsLight" is not supported
                warnLightMaterialTint = true;
            }
            else if(lightBindMode == pxr::UsdLuxTokens->noMaterialResponse)
            {
                // If this is set no need to continue,
                subGeomPack.geomLightSurfaces.emplace_back(surface.surfacePrim, surface.uniquePrim,
                                                           surface.surfaceTransform);
                auto& newLight = subGeomPack.geomLightSurfaces.back();
                newLight.subGeometryMaterialKeys.emplace_back(std::numeric_limits<uint32_t>::max(),
                                                              surface.surfacePrim.GetPath());
                continue;
            }
        }

        // Copy the surface
        subGeomPack.surfaces.push_back(surface);
        // Copy the unique prim
        subGeomPack.uniquePrims.emplace(surface.uniquePrim);
        if(subsets.empty())
        {
            MatBindAPI matBinder(surfacePrim);
            MatUSD boundMaterial = matBinder.ComputeBoundMaterial(&bindingsCache,
                                                                  &collQueryCache);
            BindMaterial(subGeomPack.surfaces.back(), boundMaterial);
        }
        else for(uint32_t i = 0; i < subsets.size(); i++)
        {
            const auto& subset = subsets[i];
            MatBindAPI matBinder(subset);
            MatUSD boundMaterial = matBinder.ComputeBoundMaterial(&bindingsCache,
                                                                  &collQueryCache);
            BindMaterial(subGeomPack.surfaces.back(), boundMaterial, i);
        }
    }

    // Warn about stuff.
    static constexpr auto LightWarningLog =
        "[MRayUSD]: Some geometry lights have the tag "
        "\"{}\". MRay supports geometry lights with no material response. "
        "These geometries contribute to the via the materials only. "
        "These material-backed geometries are not considered in NEE system. ";

    if(warnLightIndependent)
        MRAY_WARNING_LOG(LightWarningLog, "independent");
    if(warnLightMaterialTint)
        MRAY_WARNING_LOG(LightWarningLog, "materialGlowTintsLight");
    if(warnUsingDisplayColor)
        MRAY_WARNING_LOG("[MRayUSD]: Some geometries had no material bound. The "
                         "\"displayColor\" attribute colored Lambert materials "
                         "are attached to these scenes.");
    if(warnMaterialDisplayNotFound)
        MRAY_WARNING_LOG("[MRayUSD]: When trying to find \"displayColor\" to "
                         "create fallback material for non material-bounded "
                         "geometries, \"displayColor\" is not found. "
                         "Color is set to middle gray (linear).");
    if(warnMaterialDisplayColorVarying)
        MRAY_WARNING_LOG("[MRayUSD]: When trying to find \"displayColor\" to create "
                         "fallback material for non material-bounded geometries, "
                         "\"displayColor\" is not constant. MRay does not support "
                         "per-vertex / per-face varying display colors (yet). "
                         "First value of the queried array will be used.");
    if(warnEdgeSubset)
        MRAY_WARNING_LOG("[MRayUSD]: Some meshes' geometry subset element type is "
                         "set to \"edge\" or \"point\". MRay does not understand edge "
                         "or point sets. Only \"\face\" sets are allowed. These "
                         "meshes are skipped.");
    if(warnUnsupportedMaterial)
        MRAY_WARNING_LOG("[MRayUSD]: Some meshes' have unkown bound shader. "
                         "MRay supports \"USDPreviewSurface\" shaders only. "
                         "These surfaces will utilize meshes' display color property.");
}

SceneLoaderUSD::SceneLoaderUSD(BS::thread_pool& tp)
    : threadPool(tp)
{}

Expected<TracerIdPack> SceneLoaderUSD::LoadScene(TracerI& tracer,
                                                 const std::string& filePath)
{
    Timer t; t.Start();
    MRayError e = MRayError::OK;

    // Load the stage!
    pxr::TfErrorMark pxrErrorMark;
    if(!pxr::UsdStage::IsSupportedFile(filePath))
        return MRayError("\"{}\" is not a supported USD format", filePath);
    loadedStage = pxr::UsdStage::Open(filePath);
    // TODO: Is this ok?
    if(!loadedStage)
    {
        std::string errors;
        if(!pxrErrorMark.IsClean())
        {
            MRayError pxrErrs = MRayError("Unable to Open {}:\n", filePath);
            for(const pxr::TfError& pxrErr : pxrErrorMark)
            {
                using namespace std::literals;
                pxrErrs.AppendInfo(" "s + pxrErr.GetCommentary() + "\n");
            }
            // Do not propagate any errors to further callers.
            pxrErrorMark.Clear();
            return pxrErrs;
        }
        else return MRayError("Unable to Open {}", filePath);
    }

    // Hopefully everything is ok.
    //
    // I've checked the imaging pipeline, but it was unecessarily
    // complicated? (It is because of the transitioning from Hd-1.0
    // to Hd-2.0 probably). Hd instancer uses scene delegate but
    // not scene index?
    //
    // Anyway, we raw traverse the scene. It may prone to logical
    // errors but this will be rewritten when Hd-2.0 is mature enough.
    pxr::UsdGeomXformCache transformCache;
    std::vector<MRayUSDPrimSurface> surfaces;
    surfaces.reserve(4096);
    bool warnUnkownTypes = false;
    // Traversal
    auto iterator = loadedStage->Traverse(pxr::UsdTraverseInstanceProxies());
    for(auto i = iterator.begin(); i != iterator.end(); i++)
    {
        const pxr::UsdPrim prim = *i;
        // Similar to the main loop, skip point instancers
        if(prim.IsA<pxr::UsdGeomPointInstancer>())
        {
            warnUnkownTypes = true;
            i.PruneChildren();
            continue;
        }
        // For the rest, assume these are transforms only
        // We do not swicth/case here, but these are seperated for readibility
        bool isGeometry = (prim.IsA<pxr::UsdGeomMesh>() ||
                           prim.IsA<pxr::UsdGeomSphere>());
        bool isLight = prim.IsA<pxr::UsdLuxDomeLight>();
        bool isCamera = prim.IsA<pxr::UsdGeomCamera>();

        pxr::TfToken visibility;
        if(isGeometry && pxr::UsdGeomImageable(prim).GetVisibilityAttr().Get(&visibility) &&
           visibility == pxr::UsdGeomTokens->invisible)
        {
            i.PruneChildren();
            continue;
        }
        // TODO: How to validate these?
        bool isIntermediate = true;
        if(!(isGeometry || isLight || isCamera || isIntermediate))
        {
            warnUnkownTypes = true;
            continue;
        }
        //
        Matrix4x4 toWorld = ConvertToMRayMatrix(transformCache.GetLocalToWorldTransform(prim));
        pxr::UsdPrim uniquePrim = prim.IsInstanceProxy() ? prim.GetPrimInPrototype() : prim;
        surfaces.emplace_back(prim, uniquePrim, toWorld);
    }
    transformCache.Clear();

    // Issue warning about un
    if(warnUnkownTypes)
        MRAY_WARNING_LOG("[MRayUSD]: Unsupported geometries detected and skipped. "
                         "MRay currently supports \"Mesh\" "
                         "\"Sphere\" geometries, \"DomeLight\" and "
                         "\"Geometry-backed\" lights.");

    // Partition the prims
    auto PrimTypeCompUSD = [](const MRayUSDPrimSurface& left,
                              const MRayUSDPrimSurface& right)
    {
        return left.surfacePrim.GetTypeName() < right.surfacePrim.GetTypeName();
    };
    // Sort by type
    std::sort(surfaces.begin(), surfaces.end(), PrimTypeCompUSD);
    // N-way Partition
    auto primTypeRange = Algo::PartitionRange(surfaces.begin(),
                                              surfaces.end(),
                                              PrimTypeCompUSD);

    // Resolve the Geometries
    MRayUSDMaterialMap uniqueMaterials;
    CollapsedPrims meshMatPrims;
    CollapsedPrims sphereMatPrims;
    for(const auto& startEnd : primTypeRange)
    {
        Span<MRayUSDPrimSurface> range(surfaces.begin() + startEnd[0],
                                       startEnd[1] - startEnd[0]);
        if(range.front().surfacePrim.IsA<pxr::UsdGeomMesh>())
            ExpandGeomsAndFindMaterials(meshMatPrims, uniqueMaterials, range);
        else if(range.front().surfacePrim.IsA<pxr::UsdGeomSphere>())
            ExpandGeomsAndFindMaterials(sphereMatPrims, uniqueMaterials, range);
    }

    // Resolve Lights & Cameras
    // We do not need to do anything here just split the range by type
    Optional<MRayUSDPrimSurface> domeLight;
    std::vector<MRayUSDPrimSurface> cameras;
    for(const auto& startEnd : primTypeRange)
    {
        Span<MRayUSDPrimSurface> range(surfaces.begin() + startEnd[0],
                                       startEnd[1] - startEnd[0]);

        if(range.front().surfacePrim.IsA<pxr::UsdGeomCamera>())
        {
            cameras.reserve(range.size());
            cameras.insert(cameras.end(), range.begin(), range.end());
        }
        else if(range.front().surfacePrim.IsA<pxr::UsdLuxDomeLight>())
        {
            if(range.size() > 1)
                MRAY_WARNING_LOG("There is more than one \"DomeLight\" in the scene. "
                                 "MRay support only single boundary light "
                                 "dome light. First light will be used "
                                 "(According to the traversal).");
            domeLight = range.front();
        }
    }
    surfaces.clear();


    // Report
    PrintPrims(meshMatPrims, sphereMatPrims, uniqueMaterials,
               domeLight, cameras, loadedStage);

    // Now do the processing
    std::map<pxr::UsdPrim, std::vector<PrimBatchId>> outMeshPrimBatches;
    e = ProcessUniqueMeshes(outMeshPrimBatches, tracer, threadPool, meshMatPrims);
    if(e) return e;

    std::map<pxr::UsdPrim, std::vector<PrimBatchId>> outSpherePrimBatches;
    e = ProcessUniqueSpheres(outSpherePrimBatches, tracer, threadPool, meshMatPrims);
    if(e) return e;

    // Process the materials
    std::map<pxr::UsdPrim, MRayUSDMatAlphaPack> outMaterials;
    ProcessUniqueMaterials(outMaterials, tracer, threadPool, uniqueMaterials);

    //
    t.Split();
    return MRayError("Not yet implemented");
    //return resultinIdPack;
}

Expected<TracerIdPack> SceneLoaderUSD::LoadScene(TracerI&, std::istream&)
{
    return MRayError("SceneLoaderUSD does not support streaming scene data!");
}

void SceneLoaderUSD::ClearScene()
{}