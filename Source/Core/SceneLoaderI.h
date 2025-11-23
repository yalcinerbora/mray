#pragma once

#include <string>
#include <map>
#include "Expected.h"
#include "Types.h"
#include "TracerI.h"

class ThreadPool;

enum class SceneTexId : uint32_t {};
enum class SceneVolId : uint32_t {};

// TODO: Some formats (well probably all formats)
// Have a string as "name" parameter. MRay scene format
// has integer identifiers, change this later to support
// strings.
struct TracerIdPack
{
    using PrimIdMappings        = std::map<uint32_t, Pair<PrimGroupId, PrimBatchId>>;
    using CamIdMappings         = std::map<uint32_t, Pair<CameraGroupId, CameraId>>;
    using LightIdMappings       = std::map<uint32_t, Pair<LightGroupId, LightId>>;
    using TransformIdMappings   = std::map<uint32_t, Pair<TransGroupId, TransformId>>;
    using MaterialIdMappings    = std::map<uint32_t, Pair<MatGroupId, MaterialId>>;
    using MediumIdMappings      = std::map<uint32_t, Pair<MediumGroupId, MediumId>>;
    using TextureIdMappings     = std::map<SceneTexId, TextureId>;
    using VolumeIdMappings      = std::map<SceneVolId, VolumeId>;

    std::vector<char>           concatStrings;
    std::vector<Vector2ul>      stringRanges;

    PrimIdMappings      prims;
    CamIdMappings       cams;
    LightIdMappings     lights;
    TransformIdMappings transforms;
    MaterialIdMappings  mats;
    MediumIdMappings    mediums;
    TextureIdMappings   textures;
    VolumeIdMappings    volumes;
    std::vector<Pair<uint32_t, SurfaceId>>      surfaces;
    std::vector<Pair<uint32_t, CamSurfaceId>>   camSurfaces;
    std::vector<Pair<uint32_t, LightSurfaceId>> lightSurfaces;
    Pair<uint32_t, LightSurfaceId>              boundarySurface;
    VolumeId                                    boundaryVolume;

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

using SceneLoaderConstructorArgs = TypePack<ThreadPool&>;