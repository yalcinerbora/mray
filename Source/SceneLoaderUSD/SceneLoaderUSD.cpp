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
#include <pxr/usd/usdGeom/metrics.h>
// Materials
#include <pxr/usd/usdShade/materialBindingAPI.h>
// Lights
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/lightAPI.h>
// Error Related
#include <pxr/base/tf/errorMark.h>

using namespace TypeNameGen::Runtime;

enum class MRayUSDGeomMatResolveWarningsEnum : uint32_t
{
    LIGHT_INDEPENDENT,
    LIGHT_MATERIAL_TINT,
    DISPLAY_COLOR_VARYING,
    DISPLAY_COLOR_NOT_FOUND,
    USING_DISPLAY_COLOR,
    UNSUPPORTED_MATERIAL,
    EDGE_SUBSET,

    END
};
using MRayUSDGeomMatResolWarnings = std::bitset<uint32_t(MRayUSDGeomMatResolveWarningsEnum::END)>;

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
            MRAY_LOG("[{:>8s}] {}",
                     s.cullFace ? "CullFace" : "NoCull",
                     s.surfacePrim.GetPath().GetString());
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

MRayUSDGeomMatResolWarnings ExpandGeomsAndFindMaterials(CollapsedPrims& subGeomPack,
                                                        MRayUSDMaterialMap& uniqueMaterials,
                                                        Span<MRayUSDPrimSurface> surfacesRange)
{
    using enum MRayUSDGeomMatResolveWarningsEnum;
    MRayUSDGeomMatResolWarnings warnings;

    // TODO: It is not avail on the token pool (UsdShadeTokens)?
    // Check later.
    pxr::TfToken usdPrevSurfToken = pxr::TfToken("UsdPreviewSurface");

    auto BindMaterial = [&](MRayUSDPrimSurface& surface,
                            pxr::UsdShadeMaterial boundMaterial,
                            uint32_t subGeomIndex = 0)
    {
        // Check if the material is usd preview surface, if not fallback to
        // display color stuff
        if(boundMaterial)
        {
            auto shader = boundMaterial.ComputeSurfaceSource();
            pxr::TfToken shaderId;
            shader.GetShaderId(&shaderId);
            if(shaderId == usdPrevSurfToken)
            {
                pxr::UsdPrim matPrim = boundMaterial.GetPrim();
                surface.subGeometryMaterialKeys.emplace_back(subGeomIndex, matPrim.GetPath());
                if(matPrim.IsInstanceProxy())
                    matPrim = matPrim.GetPrimInPrototype();
                uniqueMaterials.emplace(matPrim, matPrim);
                return;
            }
        }
        else warnings[uint32_t(UNSUPPORTED_MATERIAL)] = true;

        warnings[uint32_t(USING_DISPLAY_COLOR)] = true;
        // No material found, fallback to display color
        // iff is constant, if not do a middle gray
        MRayUSDFallbackMaterial fallbackMat = {};
        pxr::UsdGeomGprim gPrim(surface.surfacePrim);
        pxr::UsdGeomPrimvar displayColor = gPrim.GetDisplayColorPrimvar();
        if(!displayColor || !displayColor.HasAuthoredValue())
        {
            // Warn about no display color,
            warnings[uint32_t(DISPLAY_COLOR_NOT_FOUND)] = true;
            fallbackMat.color = pxr::GfVec3f(0.5);
        }
        else
        {
            using DisplayColorUSD = pxr::VtArray<pxr::GfVec3f>;
            DisplayColorUSD displayColorValue;
            displayColor.Get<DisplayColorUSD>(&displayColorValue);
            fallbackMat.color = displayColorValue.AsConst()[0];

            // Warn about display color is not constant
            warnings[uint32_t(DISPLAY_COLOR_VARYING)] = (displayColor.GetInterpolation() !=
                                                         pxr::UsdGeomTokens->constant);
        }
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
            warnings[uint32_t(EDGE_SUBSET)] = true;
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
                warnings[uint32_t(LIGHT_INDEPENDENT)] = true;
            }
            else if(lightBindMode == pxr::UsdLuxTokens->materialGlowTintsLight)
            {
                // Warn about "materialGlowTintsLight" is not supported
                warnings[uint32_t(LIGHT_MATERIAL_TINT)] = true;
            }
            else if(lightBindMode == pxr::UsdLuxTokens->noMaterialResponse)
            {
                // If this is set no need to continue,
                subGeomPack.geomLightSurfaces.emplace_back(false,
                                                           surface.surfacePrim, surface.uniquePrim,
                                                           surface.surfaceTransform);
                auto& newLight = subGeomPack.geomLightSurfaces.back();
                newLight.subGeometryMaterialKeys.emplace_back(std::numeric_limits<uint32_t>::max(),
                                                              surface.surfacePrim.GetPath());
                continue;
            }
        }

        // Copy the surface
        subGeomPack.surfaces.push_back(surface);
        // Copy the unique prim (conditional)
        subGeomPack.uniquePrims.emplace(surface.uniquePrim);
        // Set cull face flag
        bool isDoubleSided = false;
        gPrim.GetDoubleSidedAttr().Get<bool>(&isDoubleSided);
        subGeomPack.surfaces.back().cullFace = !isDoubleSided;
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
    return warnings;
}

MRayError ProcessCameras(CameraGroupId& camGroupId,
                         std::vector<std::pair<pxr::UsdPrim, CameraId>>& outCamIds,
                         TracerI& tracer, BS::thread_pool&,
                         const std::vector<MRayUSDPrimSurface>& cameras,
                         const pxr::UsdStageRefPtr& loadedStage)
{
    size_t camCount = std::max(size_t(1), cameras.size());

    using namespace std::string_view_literals;
    static constexpr auto CAMERA_NAME = "Pinhole"sv;
    auto CAMERA_TYPE = TypeNameGen::CompTime::CameraTypeName<CAMERA_NAME>;
    camGroupId = tracer.CreateCameraGroup(std::string(CAMERA_TYPE));

    using AList = std::vector<AttributeCountList>;
    auto camIds = tracer.ReserveCameras(camGroupId, AList(camCount, {1, 1, 1, 1}));

    TransientData fovPlaneBuffer(std::in_place_type_t<Vector4>(), camCount);
    TransientData gazeBuffer(std::in_place_type_t<Vector3>(), camCount);
    TransientData positionBuffer(std::in_place_type_t<Vector3>(), camCount);
    TransientData upBuffer(std::in_place_type_t<Vector3>(), camCount);
    fovPlaneBuffer.ReserveAll();
    gazeBuffer.ReserveAll();
    positionBuffer.ReserveAll();
    upBuffer.ReserveAll();
    Span fovPlaneSpan = fovPlaneBuffer.AccessAs<Vector4>();
    Span gazeSpan = gazeBuffer.AccessAs<Vector3>();
    Span positionSpan = positionBuffer.AccessAs<Vector3>();
    Span upSpan = upBuffer.AccessAs<Vector3>();
    std::fill(gazeSpan.begin(), gazeSpan.end(), -Vector3::ZAxis());
    std::fill(positionSpan.begin(), positionSpan.end(), Vector3::Zero());
    std::fill(upSpan.begin(), upSpan.end(), Vector3::YAxis());
    if(cameras.empty())
    {
        using MathConstants::DegToRadCoef;
        MRAY_WARNING_LOG("[MRayUSD]: There is no camera on USD Scene. "
                         "Creating a generic camera (16:9)");
        // TODO: This should be costly, maybe we generate this during traversal
        pxr::UsdGeomImageable rootImg(loadedStage->GetPseudoRoot());
        pxr::GfBBox3d bb = rootImg.ComputeWorldBound(pxr::UsdTimeCode::Default(),
                                                       pxr::UsdGeomTokens->default_);
        pxr::GfRange3d bbox = bb.ComputeAlignedRange();
        pxr::GfVec3d bboxSize = bbox.GetSize();
        pxr::GfVec3d mid = bbox.GetMidpoint();
        bool zUp = (pxr::UsdGeomGetStageUpAxis(loadedStage) == pxr::UsdGeomTokens->z);
        // Convert to MRay vectors, it is little bit easier to express
        // (Countinue double since USD returned double)
        // Find the max
        Vector3d extent = Vector3d(bboxSize[0], bboxSize[1], bboxSize[2]);
        Vector3d gaze = Vector3d(mid[0], mid[1], mid[2]);
        // Orient the camera to max index
        uint32_t axis = (zUp ? Vector2d(extent[0], extent[1])
                             : Vector2d(extent[0], extent[2])).Maximum();
        uint32_t dirAxis = (axis == 1) ? 0 : 1;
        axis = (!zUp && axis == 1) ? axis + 1 : axis;
        dirAxis = (!zUp && dirAxis == 1) ? dirAxis + 1 : dirAxis;
        Vector3d dir = Vector3d::Zero();
        dir[dirAxis] = 1;
        // Get fov half to calculate the cam position (cam will cover
        // entire bbox
        double widthHalf = extent[axis] * 0.5;
        static constexpr double fovX = 70.0 * DegToRadCoef<double>();
        double tanXHalf = std::tan(fovX * 0.5);
        double posDist = widthHalf / tanXHalf;
        Vector3d pos = gaze - dir * posDist;
        // Find fovY (aspect correct)
        double fovY = 2.0 * std::atan(tanXHalf * 0.5625);
        Vector4 fovAndPlanes(fovX, fovY, 0.1, 5000);
        //
        if(zUp)
        {
            gaze = TransformGen::ZUpToYUp(gaze);
            pos = TransformGen::ZUpToYUp(pos);
        }
        // Write all!
        fovPlaneSpan[0] = fovAndPlanes;
        positionSpan[0] = Vector3(pos[0], pos[1], pos[2]);
        gazeSpan[0] = Vector3(gaze[0], gaze[1], gaze[2]);
    }
    else for(size_t i = 0; i < cameras.size(); i++)
    {
        const auto& cam = cameras[i];
        auto camPrim = pxr::UsdGeomCamera(cam.surfacePrim);
        // For camera matrices, embed to the camera position vector
        // direction etc.
        // This is not always true, but most of the time
        // up vector is always anchore
        Vector3 front = -Vector3::ZAxis();
        Vector3 pos = Vector3::Zero();
        pos = Vector3(*cam.surfaceTransform * Vector4(pos, 1));
        front = *cam.surfaceTransform * front;
        gazeSpan[i] = pos + front;
        positionSpan[i] = pos;
        upSpan[i] = Vector3::YAxis();

        pxr::GfCamera pxrCam = camPrim.GetCamera(pxr::UsdTimeCode());
        float fovX = pxrCam.GetFieldOfView(pxr::GfCamera::FOVHorizontal);
        float fovY = pxrCam.GetFieldOfView(pxr::GfCamera::FOVVertical);
        fovX *= MathConstants::DegToRadCoef<float>();
        fovY *= MathConstants::DegToRadCoef<float>();
        pxr::GfRange1f nearFar = pxrCam.GetClippingRange();

        Vector4 fovAndPlanes(fovX, fovY, nearFar.GetMin(), nearFar.GetMax());
        fovPlaneSpan[i] = fovAndPlanes;
    }
    CommonIdRange range = CommonIdRange(std::bit_cast<CommonId>(camIds.front()),
                                        std::bit_cast<CommonId>(camIds.back()));
    tracer.CommitCamReservations(camGroupId);
    tracer.PushCamAttribute(camGroupId, range, 0, std::move(fovPlaneBuffer));
    tracer.PushCamAttribute(camGroupId, range, 1, std::move(gazeBuffer));
    tracer.PushCamAttribute(camGroupId, range, 2, std::move(positionBuffer));
    tracer.PushCamAttribute(camGroupId, range, 3, std::move(upBuffer));

    if(cameras.empty())
        outCamIds.emplace_back(pxr::UsdPrim(), camIds[0]);
    else for(size_t i = 0; i < cameras.size(); i++)
        outCamIds.emplace_back(cameras[i].surfacePrim, camIds[i]);

    return MRayError::OK;
}

MRayError FindLightTextures(std::map<pxr::UsdPrim, MRayUSDTexture>& extraTextures,
                            const std::vector<MRayUSDPrimSurface>&,
                            const std::vector<MRayUSDPrimSurface>&,
                            const Optional<MRayUSDPrimSurface>& domeLight)
{
    if(domeLight)
    {
        pxr::UsdLuxDomeLight lightPrim(domeLight->uniquePrim);
        pxr::UsdAttribute fileA = lightPrim.GetTextureFileAttr();
        pxr::SdfAssetPath path; fileA.Get(&path);
        std::string filePath = path.GetResolvedPath();
        MRayUSDTexture tex =
        {
            .absoluteFilePath = filePath,
            .imageSubChannel = ImageSubChannelType::RGB,
            .isNormal = false,
            .params = MRayTextureParameters
            {
                // Random ass type this will be overriden
                .pixelType = MRayPixelType<MRayPixelEnum::MR_R_HALF>(),
                .colorSpace = MRayColorSpaceEnum::MR_DEFAULT,
                .gamma = Float(1),
                // Ignore the resolution clamping
                .ignoreResClamp = true,
                .isColor = AttributeIsColor::IS_COLOR,
                .edgeResolve = MRayTextureEdgeResolveEnum::MR_WRAP,
                .interpolation = MRayTextureInterpEnum::MR_LINEAR,
                .readMode = MRayTextureReadMode::MR_PASSTHROUGH
            }
        };
        extraTextures.emplace(domeLight->uniquePrim, tex);
    }
    // TODO: Do the rest

    return MRayError::OK;
}

MRayError ProcessLights(std::vector<std::pair<LightGroupId, LightId>>&,
                        LightGroupId& domeLightGroupId, LightId& domeLightId,
                        TracerI& tracer, BS::thread_pool&,
                        const std::vector<MRayUSDPrimSurface>& meshGeomLights,
                        const std::vector<MRayUSDPrimSurface>& sphereGeomLights,
                        const Optional<MRayUSDPrimSurface>& domeLight,
                        const std::map<pxr::UsdPrim, std::vector<PrimBatchId>>&,
                        const std::map<pxr::UsdPrim, std::vector<PrimBatchId>>&,
                        const std::map<pxr::UsdPrim, TextureId>& uniqueTextureIds)
{
    using namespace std::string_view_literals;
    static constexpr auto SKY_LIGHT_NAME = "Skysphere_Spherical"sv;
    auto SKY_LIGHT_TYPE = TypeNameGen::CompTime::LightTypeName<SKY_LIGHT_NAME>;
    domeLightGroupId = tracer.CreateLightGroup(std::string(SKY_LIGHT_TYPE));

    domeLightId = tracer.ReserveLight(domeLightGroupId, {1});
    TransientData lightBuffer(std::in_place_type_t<Vector3>{}, 1);
    std::vector<Optional<TextureId>> lightTexture(1);
    lightBuffer.ReserveAll();
    if(!domeLight)
    {
        MRAY_WARNING_LOG("[MRayUSD]: There is no boundary light on USD Scene. "
                         "Creating a uniform white background");
        lightBuffer.AccessAs<Vector3>()[0] = Vector3(10);
    }
    else
    {
        pxr::UsdLuxDomeLight lightPrim(domeLight->uniquePrim);
        TextureId texId = uniqueTextureIds.at(domeLight->uniquePrim);
        lightTexture[0] = texId;
    }
    tracer.CommitLightReservations(domeLightGroupId);
    tracer.PushLightAttribute(domeLightGroupId,
                              CommonIdRange(std::bit_cast<CommonId>(domeLightId),
                                            std::bit_cast<CommonId>(domeLightId)),
                              0, std::move(lightBuffer), lightTexture);

    // TODO: Implement later
    if(!meshGeomLights.empty() || !sphereGeomLights.empty())
    {
        MRAY_WARNING_LOG("[MRayUSD]: MRay detected geometry lights (in MRay terms, primitive-backed "
                         "lights) in the scene. Altough these are supported in MRay, wiring is not yet "
                         "implemented. Please use a single dome light");
    }
    return MRayError::OK;
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
        surfaces.emplace_back(false, prim, uniquePrim, toWorld);
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
    MRayUSDGeomMatResolWarnings geomExpandWarnings;
    for(const auto& startEnd : primTypeRange)
    {
        Span<MRayUSDPrimSurface> range(surfaces.begin() + startEnd[0],
                                       startEnd[1] - startEnd[0]);
        if(range.front().surfacePrim.IsA<pxr::UsdGeomMesh>())
            geomExpandWarnings |= ExpandGeomsAndFindMaterials(meshMatPrims,
                                                              uniqueMaterials,
                                                              range);
        else if(range.front().surfacePrim.IsA<pxr::UsdGeomSphere>())
            geomExpandWarnings |= ExpandGeomsAndFindMaterials(sphereMatPrims,
                                                              uniqueMaterials,
                                                              range);
    }
    // Warn about stuff.
    static constexpr auto LightWarningLog =
        "[MRayUSD]: Some geometry lights have the tag "
        "\"{}\". MRay supports geometry lights with no material response. "
        "These geometries contribute to the via the materials only. "
        "These material-backed geometries are not considered in NEE system. ";
    using enum MRayUSDGeomMatResolveWarningsEnum;
    if(geomExpandWarnings[uint32_t(LIGHT_INDEPENDENT)])
        MRAY_WARNING_LOG(LightWarningLog, "independent");
    if(geomExpandWarnings[uint32_t(LIGHT_MATERIAL_TINT)])
        MRAY_WARNING_LOG(LightWarningLog, "materialGlowTintsLight");
    if(geomExpandWarnings[uint32_t(USING_DISPLAY_COLOR)])
        MRAY_WARNING_LOG("[MRayUSD]: Some geometries had no material bound. The "
                         "\"displayColor\" attribute colored Lambert materials "
                         "are attached to these scenes.");
    if(geomExpandWarnings[uint32_t(UNSUPPORTED_MATERIAL)])
        MRAY_WARNING_LOG("[MRayUSD]: Some meshes' have unkown bound shader. "
                         "MRay supports \"USDPreviewSurface\" shaders only. "
                         "These surfaces will utilize meshes' display color property.");
    if(geomExpandWarnings[uint32_t(DISPLAY_COLOR_NOT_FOUND)])
        MRAY_WARNING_LOG("[MRayUSD]: When trying to find \"displayColor\" to "
                         "create fallback material for non material-bounded "
                         "geometries, \"displayColor\" is not found. "
                         "Color is set to middle gray (linear).");
    if(geomExpandWarnings[uint32_t(DISPLAY_COLOR_VARYING)])
        MRAY_WARNING_LOG("[MRayUSD]: When trying to find \"displayColor\" to create "
                         "fallback material for non material-bounded geometries, "
                         "\"displayColor\" is not constant. MRay does not support "
                         "per-vertex / per-face varying display colors (yet). "
                         "First value of the queried array will be used.");
    if(geomExpandWarnings[uint32_t(EDGE_SUBSET)])
        MRAY_WARNING_LOG("[MRayUSD]: Some meshes' geometry subset element type is "
                         "set to \"edge\" or \"point\". MRay does not understand edge "
                         "or point sets. Only \"\face\" sets are allowed. These "
                         "meshes are skipped.");

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
                MRAY_WARNING_LOG("[MRayUSD]: There is more than one \"DomeLight\" in the scene. "
                                 "MRay support only single boundary light "
                                 "(dome light). First light will be used "
                                 "(According to the USD traversal).");
            domeLight = range.front();
        }
    }
    surfaces.clear();

    // Report
    //PrintPrims(meshMatPrims, sphereMatPrims, uniqueMaterials,
    //           domeLight, cameras, loadedStage);

    // Now do the processing
    PrimGroupId meshPrimGroupId;
    std::map<pxr::UsdPrim, std::vector<PrimBatchId>> uniqueMeshPrimBatches;
    e = ProcessUniqueMeshes(meshPrimGroupId,
                            uniqueMeshPrimBatches, tracer, threadPool,
                            meshMatPrims.uniquePrims);
    if(e) return e;
    //
    PrimGroupId spherePrimGroupId;
    std::map<pxr::UsdPrim, std::vector<PrimBatchId>> uniqueSpherePrimBatches;
    e = ProcessUniqueSpheres(spherePrimGroupId,
                             uniqueSpherePrimBatches, tracer, threadPool,
                             sphereMatPrims.uniquePrims);
    if(e) return e;

    // Find extra textures
    // From lights
    std::map<pxr::UsdPrim, MRayUSDTexture> extraTextures;
    e = FindLightTextures(extraTextures,
                          meshMatPrims.geomLightSurfaces,
                          sphereMatPrims.geomLightSurfaces,
                          domeLight);
    if(e) return e;

    // Process the materials
    std::map<pxr::UsdPrim, MRayUSDMatAlphaPack> outMaterials;
    std::map<pxr::UsdPrim, TextureId> uniqueTextureIds;
    e = ProcessUniqueMaterials(outMaterials, uniqueTextureIds,
                               tracer, threadPool, uniqueMaterials,
                               extraTextures);
    if(e) return e;

    // Process cameras
    // Sort the camera's by name for consistent loads
    // Some compositions weill reorder the cameras
    std::sort(cameras.begin(), cameras.end(), 
    [](const MRayUSDPrimSurface& l, const MRayUSDPrimSurface& r) -> bool
    {
        return l.surfacePrim < r.surfacePrim;
    });
    CameraGroupId camGroupId;
    std::vector<std::pair<pxr::UsdPrim, CameraId>> outCameras;
    e = ProcessCameras(camGroupId, outCameras, tracer,
                       threadPool, cameras, loadedStage);
    if(e) return e;

    LightGroupId boundaryLightGroup;
    LightId boundaryLight;
    std::vector<std::pair<LightGroupId, LightId>> outLights;
    e = ProcessLights(outLights, boundaryLightGroup, boundaryLight,
                      tracer, threadPool,
                      meshMatPrims.geomLightSurfaces,
                      sphereMatPrims.geomLightSurfaces,
                      domeLight, uniqueMeshPrimBatches,
                      uniqueSpherePrimBatches,
                      uniqueTextureIds);
    if(e) return e;

    // For mesh prims, pre-apply transformations if this prim is used once
    // (no instancing)        
    {
        std::vector<pxr::UsdPrim> uniqueMeshPrims;
        uniqueMeshPrims.reserve(uniqueMeshPrimBatches.size());
        for(const auto& meshPrim : uniqueMeshPrimBatches)
            uniqueMeshPrims.push_back(meshPrim.first);
        std::sort(uniqueMeshPrims.begin(), uniqueMeshPrims.end());
        
        std::vector<PrimBatchId> meshPrimBatches;
        std::vector<Matrix4x4>  meshTransforms;
        // Traverse through unique prim batches, add transform application list
        for(auto& surface : meshMatPrims.surfaces)
        {
            auto [start, end] = std::equal_range(uniqueMeshPrims.cbegin(), 
                                                 uniqueMeshPrims.cend(),
                                                 surface.uniquePrim);
            if(std::distance(end, start) > 1) continue;

            Matrix4x4 transform = *surface.surfaceTransform;
            if(pxr::UsdGeomGetStageUpAxis(loadedStage) == pxr::UsdGeomTokens->z)
                transform = TransformGen::ZUpToYUpMat<Float>() * transform;

            for(PrimBatchId pbId : uniqueMeshPrimBatches.at(surface.uniquePrim))
            {
                meshPrimBatches.push_back(pbId);
                meshTransforms.push_back(transform);
            }                
            surface.surfaceTransform = std::nullopt;
        }
        // Flush up to this point (load operations are async, we will apply
        // transformations on loaded meshes)
        tracer.Flush();
        tracer.TransformPrimitives(meshPrimGroupId,
                                   std::move(meshPrimBatches),
                                   std::move(meshTransforms));
    }
    
    std::vector<uint32_t> meshTransformOffsets(meshMatPrims.surfaces.size() + 1);
    meshTransformOffsets[0] = 0;
    std::transform_inclusive_scan
    (
        meshMatPrims.surfaces.cbegin(),
        meshMatPrims.surfaces.cend(),
        meshTransformOffsets.begin() + 1,
        std::plus<uint32_t>{},
        [](const MRayUSDPrimSurface& s) ->uint32_t
        {
            return s.surfaceTransform.has_value() ? 1u : 0u;
        }
    );

    // Load Transforms
    std::array<size_t, 7> allSizes;
    allSizes[0] = 0;
    allSizes[1] = meshTransformOffsets.back();
    allSizes[2] = meshMatPrims.geomLightSurfaces.size();
    allSizes[3] = sphereMatPrims.surfaces.size();
    allSizes[4] = sphereMatPrims.geomLightSurfaces.size();
    allSizes[5] = 0; // cameras.size();
    allSizes[6] = domeLight.has_value() ? 1 : 0;
    std::inclusive_scan(allSizes.cbegin(), allSizes.cend(), allSizes.begin());
    size_t totalSurfSize = allSizes.back();
    //
    TransientData matrixBuffer(std::in_place_type_t<Matrix4x4>(), totalSurfSize);
    matrixBuffer.ReserveAll();
    Span allMatrices = matrixBuffer.AccessAs<Matrix4x4>();
    Span meshSurfMats = allMatrices.subspan(allSizes[0], allSizes[1] - allSizes[0]);
    Span meshLightSurfMats = allMatrices.subspan(allSizes[1], allSizes[2] - allSizes[1]);
    Span sphereSurfMats = allMatrices.subspan(allSizes[2], allSizes[3] - allSizes[2]);
    Span sphereLightSurfMats = allMatrices.subspan(allSizes[3], allSizes[4] - allSizes[3]);
    Span cameraSurfMats = allMatrices.subspan(allSizes[4], allSizes[5] - allSizes[4]);
    Span domeLightSurfMats = allMatrices.subspan(allSizes[5], allSizes[6] - allSizes[5]);

    for(size_t i = 0; i < meshSurfMats.size(); i++)
    {
        const auto& s = meshMatPrims.surfaces[i];
        if(!s.surfaceTransform.has_value()) continue;

        meshSurfMats[meshTransformOffsets[i]] = *meshMatPrims.surfaces[i].surfaceTransform;
    }        
    for(size_t i = 0; i < meshLightSurfMats.size(); i++)
        meshLightSurfMats[i] = *meshMatPrims.geomLightSurfaces[i].surfaceTransform;
    for(size_t i = 0; i < sphereSurfMats.size(); i++)
        sphereSurfMats[i] = *sphereMatPrims.surfaces[i].surfaceTransform;
    for(size_t i = 0; i < sphereLightSurfMats.size(); i++)
        sphereLightSurfMats[i] = *sphereMatPrims.geomLightSurfaces[i].surfaceTransform;
    for(size_t i = 0; i < cameraSurfMats.size(); i++)
        cameraSurfMats[i] = *cameras[i].surfaceTransform;
    if(domeLight.has_value())
        domeLightSurfMats[0] = *domeLight->surfaceTransform;

    // Convert to Y up if required
    if(pxr::UsdGeomGetStageUpAxis(loadedStage) == pxr::UsdGeomTokens->z)
    {
        for(auto& mat : allMatrices)
            mat = TransformGen::ZUpToYUpMat<Float>() * mat;
    }

    using TypeNameGen::Runtime::AddTransformPrefix;
    TransGroupId tgId = tracer.CreateTransformGroup(AddTransformPrefix("Single"));
    std::vector<AttributeCountList> transformAttribCounts(totalSurfSize, {1});
    std::vector<TransformId> tIds = tracer.ReserveTransformations(tgId, transformAttribCounts);
    tracer.CommitTransReservations(tgId);
    //
    CommonIdRange range(std::bit_cast<CommonId>(tIds.front()),
                        std::bit_cast<CommonId>(tIds.back()));
    tracer.PushTransAttribute(tgId, range, 0, std::move(matrixBuffer));
    Span allTIds = Span(tIds.cbegin(), tIds.size());
    Span meshSurfTIds = allTIds.subspan(allSizes[0], allSizes[1] - allSizes[0]);
    Span meshLightSurfTIds= allTIds.subspan(allSizes[1], allSizes[2] - allSizes[1]);
    Span sphereSurfTIds = allTIds.subspan(allSizes[2], allSizes[3] - allSizes[2]);
    Span sphereLightSurfTIds= allTIds.subspan(allSizes[3], allSizes[4] - allSizes[3]);
    Span cameraSurfTIds = allTIds.subspan(allSizes[4], allSizes[5] - allSizes[4]);
    Span domeLightSurfTIds = allTIds.subspan(allSizes[5], allSizes[6] - allSizes[5]);

    // Wire the meshes
    std::vector<Pair<pxr::UsdPrim, SurfaceId>> surfaceList;
    surfaceList.reserve(totalSurfSize);
    for(size_t i = 0; i < meshMatPrims.surfaces.size(); i++)
    {
        const auto& prim = meshMatPrims.surfaces[i];
        const auto& subGeomMaterials = prim.subGeometryMaterialKeys;
        size_t surfCount = Math::DivideUp(subGeomMaterials.size(),
                                          TracerConstants::MaxPrimBatchPerSurface);
        for(size_t batchI = 0; batchI < surfCount; batchI++)
        {
            size_t start = batchI * TracerConstants::MaxPrimBatchPerSurface;
            size_t end = (batchI + 1) * TracerConstants::MaxPrimBatchPerSurface;
            end = std::min(subGeomMaterials.size(), end);
            //
            SurfaceParams surface;
            surface.transformId = TracerConstants::IdentityTransformId;
            if(prim.surfaceTransform.has_value())
                surface.transformId = meshSurfTIds[meshTransformOffsets[i]];
            for(size_t j = start; j < end; j++)
            {
                const auto& [geomIndex, matPath] = prim.subGeometryMaterialKeys[j];
                auto primName = prim.uniquePrim;
                const auto& mat = outMaterials.at(loadedStage->GetPrimAtPath(matPath));
                const auto& primBatchId = uniqueMeshPrimBatches.at(primName)[geomIndex];
                surface.materials.push_back(mat.materialId);
                surface.primBatches.push_back(primBatchId);
                surface.cullFaceFlags.push_back(prim.cullFace);
                surface.alphaMaps.push_back(mat.alphaMap);
            }
            SurfaceId sId = tracer.CreateSurface(surface);
            surfaceList.emplace_back(prim.surfacePrim, sId);
        }
    }
    //
    for(size_t i = 0; i < sphereMatPrims.surfaces.size(); i++)
    {
        const auto& prim = sphereMatPrims.surfaces[i];
        const auto& subGeomMaterials = prim.subGeometryMaterialKeys;
        size_t surfCount = Math::DivideUp(subGeomMaterials.size(),
                                          TracerConstants::MaxPrimBatchPerSurface);
        for(size_t batchI = 0; batchI < surfCount; batchI++)
        {
            size_t start = batchI * TracerConstants::MaxPrimBatchPerSurface;
            size_t end = (batchI + 1) * TracerConstants::MaxPrimBatchPerSurface;
            end = std::min(subGeomMaterials.size(), end);
            //
            SurfaceParams surface;
            surface.transformId = sphereSurfTIds[i];
            for(size_t j = start; j < end; j++)
            {
                auto [geomIndex, matPath] = prim.subGeometryMaterialKeys[j];
                auto primName = prim.uniquePrim;
                const auto& mat = outMaterials.at(loadedStage->GetPrimAtPath(matPath));
                const auto& primBatchId = uniqueSpherePrimBatches.at(primName)[geomIndex];
                surface.materials.push_back(mat.materialId);
                surface.primBatches.push_back(primBatchId);
                surface.cullFaceFlags.push_back(!mat.alphaMap.has_value());
                surface.alphaMaps.push_back(mat.alphaMap);
            }
            SurfaceId sId = tracer.CreateSurface(surface);
            surfaceList.emplace_back(prim.surfacePrim, sId);
        }
    }

    // Camera Surfaces
    std::vector<Pair<pxr::UsdPrim, CamSurfaceId>> camSurfaceList;
    camSurfaceList.reserve(outCameras.size());
    for(size_t i = 0; i < outCameras.size(); i++)
    {
        const auto& outCam = outCameras[i];
        CameraSurfaceParams surface =
        {
            .cameraId = outCam.second,
            .transformId = cameraSurfTIds.empty()
                                ? TracerConstants::IdentityTransformId
                                : cameraSurfTIds[0],
            .mediumId = TracerConstants::VacuumMediumId,
        };
        CamSurfaceId csId = tracer.CreateCameraSurface(surface);
        camSurfaceList.emplace_back(outCam.first, csId);
    }

    // Set the boundary surface
    tracer.SetBoundarySurface
    (
        LightSurfaceParams
        {
            .lightId = boundaryLight,
            .transformId = domeLightSurfTIds.empty()
                            ? TracerConstants::IdentityTransformId
                            : domeLightSurfTIds[0],
            .mediumId = TracerConstants::VacuumMediumId
        }
    );
    //
    TracerIdPack result;
    std::vector<char> stringConcat;
    std::vector<Vector2ul> stringRange;
    stringConcat.reserve(4096);
    stringRange.reserve(4096);

    // TODO: Too many string alloc/dealloc
    // Change this later
    uint32_t globalCounter = 0;
    for(const auto& [prim, batchList] : uniqueMeshPrimBatches)
    {
        std::string s = prim.GetPath().GetString();
        for(uint32_t i = 0; i < batchList.size(); i++)
        {
            std::string v = s + std::to_string(i);
            auto loc = stringConcat.insert(stringConcat.end(),
                                           v.cbegin(), v.cend());
            auto start = std::distance(loc, stringConcat.begin());
            auto end = stringConcat.size();
            stringRange.emplace_back(start, end);
            result.prims.emplace(globalCounter++,
                                 Pair(meshPrimGroupId, batchList[i]));
        }
    }
    for(const auto& [prim, batchList] : uniqueSpherePrimBatches)
    {
        std::string s = prim.GetPath().GetString();
        for(uint32_t i = 0; i < batchList.size(); i++)
        {
            std::string v = s + std::to_string(i);
            auto loc = stringConcat.insert(stringConcat.end(),
                                           v.cbegin(), v.cend());
            auto start = std::distance(loc, stringConcat.begin());
            auto end = stringConcat.size();
            stringRange.emplace_back(start, end);
            result.prims.emplace(globalCounter++,
                                 Pair(spherePrimGroupId, batchList[i]));
        }
    }
    // Cameras
    for(const auto& [camPrim, camId] : outCameras)
    {
        std::string s = (cameras.empty())
                            ? "GENERATED"
                            : camPrim.GetPath().GetString();
        auto loc = stringConcat.insert(stringConcat.end(),
                                        s.cbegin(), s.cend());
        auto start = std::distance(loc, stringConcat.begin());
        auto end = stringConcat.size();
        stringRange.emplace_back(start, end);
        result.cams.emplace(globalCounter++,
                            Pair(camGroupId, camId));
    }
    // Lights
    {
        std::string s = (domeLight) ? domeLight->surfacePrim.GetPath().GetString()
                                    : "GENERATED";
        auto loc = stringConcat.insert(stringConcat.end(),
                                       s.cbegin(), s.cend());
        auto start = std::distance(loc, stringConcat.begin());
        auto end = stringConcat.size();
        stringRange.emplace_back(start, end);
        result.lights.emplace(globalCounter++,
                              Pair(boundaryLightGroup, boundaryLight));
    }
    // Materials
    for(const auto& [matPrim, matPack] : outMaterials)
    {
        std::string s = matPrim.GetPath().GetString();
        auto loc = stringConcat.insert(stringConcat.end(),
                                       s.cbegin(), s.cend());
        auto start = std::distance(loc, stringConcat.begin());
        auto end = stringConcat.size();
        stringRange.emplace_back(start, end);
        result.mats.emplace(globalCounter++,
                            Pair(matPack.groupId, matPack.materialId));
    }
    // Textures
    for(const auto& [texPrim, texId] : uniqueTextureIds)
    {
        std::string s = texPrim.GetPath().GetString();
        auto loc = stringConcat.insert(stringConcat.end(),
                                       s.cbegin(), s.cend());
        auto start = std::distance(loc, stringConcat.begin());
        auto end = stringConcat.size();
        stringRange.emplace_back(start, end);
        result.textures.emplace(SceneTexId(globalCounter++), texId);
    }
    // TODO: Mediums (Not loaded)
    // TODO: Transforms
    // Surfaces
    for(const auto& [surfPrim, surfId] : surfaceList)
    {
        std::string s = surfPrim.GetPath().GetString();
        auto loc = stringConcat.insert(stringConcat.end(),
                                       s.cbegin(), s.cend());
        auto start = std::distance(loc, stringConcat.begin());
        auto end = stringConcat.size();
        stringRange.emplace_back(start, end);
        result.surfaces.emplace_back(globalCounter++, surfId);
    }
    // TODO: Light Surfaces
    // CameraSurfaces is required to select cameras etc
    for(const auto& [camPrim, camSurfId] : camSurfaceList)
    {
        std::string s = (camPrim) ? camPrim.GetPath().GetString()
                                  : "GENERATED";
        auto loc = stringConcat.insert(stringConcat.end(),
                                       s.cbegin(), s.cend());
        auto start = std::distance(loc, stringConcat.begin());
        auto end = stringConcat.size();
        stringRange.emplace_back(start, end);
        result.camSurfaces.emplace_back(globalCounter++, camSurfId);
    }

    result.concatStrings = std::move(stringConcat);
    result.stringRanges = std::move(stringRange);
    t.Split();
    result.loadTimeMS = t.Elapsed<Millisecond>();
    return result;
}

Expected<TracerIdPack> SceneLoaderUSD::LoadScene(TracerI&, std::istream&)
{
    return MRayError("SceneLoaderUSD does not support streaming scene data!");
}

void SceneLoaderUSD::ClearScene()
{}