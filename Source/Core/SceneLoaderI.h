#pragma once

#include <string>
#include <map>
#include "Error.h"
#include "Types.h"
#include "TracerI.h"

enum class TextureAccessLayout
{
    R, G, B, A,
    RG, GB, BA,
    RGB, GBA,
    RGBA
};

struct NodeTexStruct
{
    uint32_t            texId;
    TextureAccessLayout channelLayout;

    auto operator<=>(const NodeTexStruct& rhs) const = default;
};

struct TracerIdPack
{
    using PrimIdMappings        = std::map<uint32_t, Pair<PrimGroupId, PrimBatchId>>;
    using CamIdMappings         = std::map<uint32_t, Pair<CameraGroupId, CameraId>>;
    using LightIdMappings       = std::map<uint32_t, Pair<LightGroupId, LightId>>;
    using TransformIdMappings   = std::map<uint32_t, Pair<TransGroupId, TransformId>>;
    using MaterialIdMappings    = std::map<uint32_t, Pair<MatGroupId, MaterialId>>;
    using MediumIdMappings      = std::map<uint32_t, Pair<MediumGroupId, MediumId>>;
    using TextureIdMappings     = std::map<NodeTexStruct, TextureId>;

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

    double loadTimeMS = 0.0;
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

using SceneLoaderConstructorArgs = Tuple<BS::thread_pool&>;