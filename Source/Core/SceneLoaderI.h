#pragma once

#include <string>
#include <map>
#include "Error.h"
#include "Types.h"
#include "TracerI.h"

#include "ImageLoader/ImageLoaderI.h"

enum class SceneTexId : uint32_t {};

struct TracerIdPack
{
    using PrimIdMappings        = std::map<uint32_t, Pair<PrimGroupId, PrimBatchId>>;
    using CamIdMappings         = std::map<uint32_t, Pair<CameraGroupId, CameraId>>;
    using LightIdMappings       = std::map<uint32_t, Pair<LightGroupId, LightId>>;
    using TransformIdMappings   = std::map<uint32_t, Pair<TransGroupId, TransformId>>;
    using MaterialIdMappings    = std::map<uint32_t, Pair<MatGroupId, MaterialId>>;
    using MediumIdMappings      = std::map<uint32_t, Pair<MediumGroupId, MediumId>>;
    using TextureIdMappings     = std::map<SceneTexId, TextureId>;

    PrimIdMappings      prims;
    CamIdMappings       cams;
    LightIdMappings     lights;
    TransformIdMappings transforms;
    MaterialIdMappings  mats;
    MediumIdMappings    mediums;
    TextureIdMappings   textures;

    std::vector<SurfaceId>      surfaces;
    std::vector<CamSurfaceId>   camSurfaces;
    std::vector<LightSurfaceId> lightSurfaces;
    LightSurfaceId              boundarySurface;

    double  loadTimeMS = 0.0;
};

class SceneLoaderI
{
    public:
    virtual ~SceneLoaderI() = default;

    virtual Expected<TracerIdPack>  LoadScene(TracerI& tracer,
                                              const std::string& filePath) = 0;
    virtual Expected<TracerIdPack>  LoadScene(TracerI& tracer,
                                              std::istream& sceneData) = 0;
    virtual void                    ClearScene() = 0;
};

namespace BS
{
    class thread_pool;
}

using SceneLoaderConstructorArgs = std::tuple<BS::thread_pool&>;