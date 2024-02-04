#pragma once

#include <BS/BS_thread_pool.hpp>

#include "MeshLoader/EntryPoint.h"
#include "Core/SceneLoaderI.h"
#include "Core/TracerI.h"

#include "JsonNode.h"

class MRayJsonNode;

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

    using TypeMappedNodes       = std::map<std::string_view, std::vector<MRayJsonNode>>;

    using PrimIdMappings        = std::map<uint32_t, PrimBatchId>;
    using CamIdMappings         = std::map<uint32_t, CameraId>;
    using LightIdMappings       = std::map<uint32_t, LightId>;
    using TransformIdMappings   = std::map<uint32_t, TransformId>;
    using MaterialIdMappings    = std::map<uint32_t, MaterialId>;
    using MediumIdMappings      = std::map<uint32_t, MediumId>;
    using TextureIdMappings     = std::map<uint32_t, TextureId>;

    template <class Map>
    struct MutexedMap
    {
        std::mutex  mutex;
        Map         map;
    };

    static std::string          SceneRelativePathToAbsolute(std::string_view sceneRelativePath,
                                                            std::string_view scenePath);

    private:
    std::string         scenePath;
    nlohmann::json      sceneJson;
    BS::thread_pool&    threadPool;
    MeshLoaderPoolPtr   meshLoaderPool;

    TypeMappedNodes     primNodes;
    TypeMappedNodes     cameraNodes;
    TypeMappedNodes     transformNodes;
    TypeMappedNodes     lightNodes;
    TypeMappedNodes     textureNodes;
    TypeMappedNodes     materialNodes;
    TypeMappedNodes     mediumNodes;

    MutexedMap<PrimIdMappings>  primMappings;

    static LightSurfaceStruct               LoadBoundary(const nlohmann::json&);
    static std::vector<SurfaceStruct>       LoadSurfaces(const nlohmann::json&);
    static std::vector<CameraSurfaceStruct> LoadCamSurfaces(const nlohmann::json&, uint32_t boundaryMediumId);
    static std::vector<LightSurfaceStruct>  LoadLightSurfaces(const nlohmann::json&, uint32_t boundaryMediumId);


    MRayError   LoadAll(TracerI&);
    MRayError   OpenFile(const std::string& filePath);
    MRayError   ReadStream(std::istream& sceneData);

    void        CreateTypeMapping(const std::vector<SurfaceStruct>&,
                                  const std::vector<CameraSurfaceStruct>&,
                                  const std::vector<LightSurfaceStruct>&,
                                  const LightSurfaceStruct& boundary);

    void        LoadTextures(TracerI&, ExceptionList&);
    void        LoadMediums(TracerI&, ExceptionList&);
    void        LoadMaterials(TracerI&, ExceptionList&);
    void        LoadTransforms(TracerI&, ExceptionList&);
    void        LoadPrimitives(TracerI&, ExceptionList&);
    void        LoadCameras(TracerI&, ExceptionList&);
    void        LoadLights(TracerI&, ExceptionList&);

    void        CreateSurfaces(TracerI&, const std::vector<SurfaceStruct>&);
    void        CreateLightSurfaces(TracerI&, const std::vector<LightSurfaceStruct>&);
    void        CreateCamSurfaces(TracerI&, const std::vector<CameraSurfaceStruct>&);

    public:
                SceneLoaderMRay(BS::thread_pool& pool);

    Pair<MRayError, double> LoadScene(TracerI& tracer,
                                      const std::string& filePath) override;

    Pair<MRayError, double> LoadScene(TracerI& tracer, std::istream& sceneData) override;
};