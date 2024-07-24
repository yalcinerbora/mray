#pragma once

#include <BS/BS_thread_pool.hpp>

#include "MeshLoader/EntryPoint.h"
#include "Core/SceneLoaderI.h"
#include "Core/TracerI.h"
#include "Core/Flag.h"
#include "Core/Expected.h"

#include "JsonNode.h"

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

    struct ExceptionList
    {
        private:
        static constexpr size_t MaxExceptionSize = 128;
        using ExceptionArray = std::array<MRayError, MaxExceptionSize>;

        public:
        std::atomic_size_t  size = 0;
        ExceptionArray      exceptions;
        void                AddException(MRayError&&);
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
    BS::thread_pool&    threadPool;

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

    std::vector<SurfaceId>          mRaySurfaces;
    std::vector<LightSurfaceId>     mRayLightSurfaces;
    std::vector<CamSurfaceId>       mRayCamSurfaces;

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

    void        LoadTextures(TracerI&, ExceptionList&);
    void        LoadMediums(TracerI&, ExceptionList&);
    void        LoadMaterials(TracerI&, ExceptionList&,
                              uint32_t boundaryMediumId);
    void        LoadTransforms(TracerI&, ExceptionList&);
    void        LoadPrimitives(TracerI&, ExceptionList&);
    void        LoadCameras(TracerI&, ExceptionList&);
    void        LoadLights(TracerI&, ExceptionList&);

    void        CreateTypeMapping(const TracerI&,
                                  const SceneSurfList&,
                                  const SceneCamSurfList&,
                                  const SceneLightSurfList&,
                                  const LightSurfaceStruct& boundary);

    void        CreateSurfaces(TracerI&, const std::vector<SurfaceStruct>&);
    void        CreateLightSurfaces(TracerI&, const std::vector<LightSurfaceStruct>&);
    void        CreateCamSurfaces(TracerI&, const std::vector<CameraSurfaceStruct>&);

    MRayError       LoadAll(TracerI&);
    MRayError       OpenFile(const std::string& filePath);
    MRayError       ReadStream(std::istream& sceneData);

    TracerIdPack    MoveIdPack(double durationMS);
    void            ClearIntermediateBuffers();

    public:
                            SceneLoaderMRay(BS::thread_pool& pool);
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