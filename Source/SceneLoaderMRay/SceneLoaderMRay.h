#pragma once

#include <BS/BS_thread_pool.hpp>

#include "MeshLoader/EntryPoint.h"
#include "Core/SceneLoaderI.h"
#include "Core/TracerI.h"

#include "JsonNode.h"

class JsonNode;

struct SceneNode
{
    nlohmann::json& jsonNode;
    uint32_t        innerIndex;
};

class SceneLoaderMRay : public SceneLoaderI
{
    public:
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

    using TypeMappedNodes       = std::map<std::string_view, std::vector<JsonNode>>;
    using TexTypeMappedNodes    = std::map<NodeTexStruct, JsonNode>;

    using PrimIdMappings        = std::map<uint32_t, PrimBatchId>;
    using CamIdMappings         = std::map<uint32_t, CameraId>;
    using LightIdMappings       = std::map<uint32_t, LightId>;
    using TransformIdMappings   = std::map<uint32_t, TransformId>;
    using MaterialIdMappings    = std::map<uint32_t, MaterialId>;
    using MediumIdMappings      = std::map<uint32_t, MediumId>;
    using TextureIdMappings     = std::map<NodeTexStruct, TextureId>;

    template <class Map>
    struct MutexedMap
    {
        std::mutex  mutex;
        Map         map;
    };

    static std::string  SceneRelativePathToAbsolute(std::string_view sceneRelativePath,
                                                    std::string_view scenePath);

    private:
    std::string         scenePath;
    nlohmann::json      sceneJson;
    BS::thread_pool&    threadPool;

    TypeMappedNodes     primNodes;
    TypeMappedNodes     cameraNodes;
    TypeMappedNodes     transformNodes;
    TypeMappedNodes     lightNodes;
    TypeMappedNodes     materialNodes;
    TypeMappedNodes     mediumNodes;

    TexTypeMappedNodes  textureNodes;

    //Scene id to -> Tracer id maps
    MutexedMap<TransformIdMappings> transformMappings;
    MutexedMap<MediumIdMappings>    mediumMappings;
    MutexedMap<PrimIdMappings>      primMappings;
    MutexedMap<MaterialIdMappings>  matMappings;
    TextureIdMappings               texMappings;

    static LightSurfaceStruct               LoadBoundary(const nlohmann::json&);
    static std::vector<SurfaceStruct>       LoadSurfaces(const nlohmann::json&);
    static std::vector<CameraSurfaceStruct> LoadCamSurfaces(const nlohmann::json&, uint32_t boundaryMediumId);
    static std::vector<LightSurfaceStruct>  LoadLightSurfaces(const nlohmann::json&, uint32_t boundaryMediumId);

    static void DryRunLightsForPrim(std::vector<uint32_t>&,
                                    const TypeMappedNodes&,
                                    const TracerI&);
    template <class TracerInterfaceFunc>
    static void DryRunNodesForTex(std::vector<NodeTexStruct>&,
                                  const TypeMappedNodes&,
                                  const TracerI&,
                                  TracerInterfaceFunc&&);

    void        LoadTextures(TracerI&, ExceptionList&);
    void        LoadMediums(TracerI&, ExceptionList&);
    void        LoadMaterials(TracerI&, ExceptionList&);
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

    MRayError   LoadAll(TracerI&);
    MRayError   OpenFile(const std::string& filePath);
    MRayError   ReadStream(std::istream& sceneData);

    public:
                SceneLoaderMRay(BS::thread_pool& pool);

    Pair<MRayError, double> LoadScene(TracerI& tracer,
                                      const std::string& filePath) override;

    Pair<MRayError, double> LoadScene(TracerI& tracer, std::istream& sceneData) override;
};