#pragma once

#include "MeshLoader/EntryPoint.h"
#include "Core/SceneLoaderI.h"
#include "Core/Expected.h"

#include "JsonNode.h"

class ThreadPool;
class TracerI;

class SceneLoaderMRay : public SceneLoaderI
{
    public:
    using TypeMappedNodes   = std::map<std::string, std::vector<JsonNode>>;
    using TexMappedNodes    = std::vector<Pair<SceneTexId, JsonNode>>;

    template <class Map>
    struct MutexedMap
    {
        std::mutex  mutex;
        Map         map;
    };

    private:
    using PrimIdMappings        = typename TracerIdPack::PrimIdMappings;
    using CamIdMappings         = typename TracerIdPack::CamIdMappings;
    using LightIdMappings       = typename TracerIdPack::LightIdMappings;
    using TransformIdMappings   = typename TracerIdPack::TransformIdMappings;
    using MaterialIdMappings    = typename TracerIdPack::MaterialIdMappings;
    using MediumIdMappings      = typename TracerIdPack::MediumIdMappings;
    using TextureIdMappings     = typename TracerIdPack::TextureIdMappings;

    private:
    std::string         scenePath;
    nlohmann::json      sceneJson;
    ThreadPool&         threadPool;

    // Temporary Internal Data
    TypeMappedNodes     primNodes;
    TypeMappedNodes     cameraNodes;
    TypeMappedNodes     transformNodes;
    TypeMappedNodes     lightNodes;
    TypeMappedNodes     materialNodes;
    TypeMappedNodes     mediumNodes;
    TexMappedNodes      textureNodes;

    // Scene id to -> Tracer id mappings
    MutexedMap<TransformIdMappings> transformMappings;
    MutexedMap<MediumIdMappings>    mediumMappings;
    MutexedMap<PrimIdMappings>      primMappings;
    MutexedMap<MaterialIdMappings>  matMappings;
    MutexedMap<CamIdMappings>       camMappings;
    MutexedMap<LightIdMappings>     lightMappings;
    TextureIdMappings               texMappings;

    LightSurfaceId  mRayBoundaryLightSurface;
    std::vector<Pair<uint32_t, SurfaceId>>         mRaySurfaces;
    std::vector<Pair<uint32_t, LightSurfaceId>>    mRayLightSurfaces;
    std::vector<Pair<uint32_t, CamSurfaceId>>      mRayCamSurfaces;

    static LightSurfaceStruct               LoadBoundary(const nlohmann::json&);
    static std::vector<SurfaceStruct>       LoadSurfaces(const nlohmann::json&);
    static std::vector<CameraSurfaceStruct> LoadCamSurfaces(const nlohmann::json&, uint32_t boundaryMediumId);
    static std::vector<LightSurfaceStruct>  LoadLightSurfaces(const nlohmann::json&, uint32_t boundaryMediumId);

    static void DryRunLightsForPrim(std::vector<uint32_t>&,
                                    const TypeMappedNodes&,
                                    const TracerI&);
    template <class TracerInterfaceFunc, class AnnotateFunc>
    static void DryRunNodesForTex(std::vector<SceneTexId>&,
                                  const TypeMappedNodes&,
                                  const TracerI&,
                                  AnnotateFunc&&,
                                  TracerInterfaceFunc&&);

    void        LoadTextures(TracerI&, ErrorList&);
    void        LoadMediums(TracerI&, ErrorList&);
    void        LoadMaterials(TracerI&, ErrorList&);
    void        LoadTransforms(TracerI&, ErrorList&);
    void        LoadPrimitives(TracerI&, ErrorList&);
    void        LoadCameras(TracerI&, ErrorList&);
    void        LoadLights(TracerI&, ErrorList&);

    void        CreateTypeMapping(const TracerI&,
                                  const SceneSurfList&,
                                  const SceneCamSurfList&,
                                  const SceneLightSurfList&,
                                  const LightSurfaceStruct& boundary);

    void        CreateSurfaces(TracerI&, const std::vector<SurfaceStruct>&);
    void        CreateLightSurfaces(TracerI&, const std::vector<LightSurfaceStruct>&,
                                    const LightSurfaceStruct& boundary);
    void        CreateCamSurfaces(TracerI&, const std::vector<CameraSurfaceStruct>&);

    MRayError       LoadAll(TracerI&);
    MRayError       OpenFile(const std::string& filePath);
    MRayError       ReadStream(std::istream& sceneData);

    TracerIdPack    MoveIdPack(double durationMS);
    void            ClearIntermediateBuffers();

    public:
                            SceneLoaderMRay(ThreadPool& pool);
                            SceneLoaderMRay(const SceneLoaderMRay&) = delete;
    SceneLoaderMRay&        operator=(const SceneLoaderMRay&) = delete;

    // Scene Loading
    Expected<TracerIdPack>  LoadScene(TracerI& tracer,
                                      const std::string& filePath) override;
    Expected<TracerIdPack>  LoadScene(TracerI& tracer,
                                      std::istream& sceneData) override;

    void                    ClearScene() override;
};

//