#pragma once

#include "Core/Types.h"
#include "Core/SceneLoaderI.h"
#include "Core/Variant.h"

#include "Common/JsonCommon.h" // IWYU pragma: keep

#include <nlohmann/json.hpp>

#include "ImageLoader/ImageLoaderI.h"

#include "NodeNames.h" // IWYU pragma: keep

static constexpr uint32_t EMPTY_TRANSFORM = std::numeric_limits<uint32_t>::max();
static constexpr uint32_t EMPTY_MEDIUM = std::numeric_limits<uint32_t>::max();

struct TracerSceneView
{
    struct LightSurfaces
    {
        TransformId tId;
        MediumId    mId;
        LightId     lId;
    };

    struct CameraSurfaces
    {
        TransformId tId;
        MediumId    mId;
        CameraId    cId;
    };

    struct Surfaces
    {
        std::array<PrimBatchId, TracerConstants::MaxPrimBatchPerSurface> primBatches;
        std::array<MaterialId, TracerConstants::MaxPrimBatchPerSurface> materials;
        TransformId             tId;
        Optional<TextureId>     alphaTexture;
        bool                    cullFace;
    };

    LightSurfaces               boundaryLight;
    std::vector<CameraSurfaces> cameras;
    std::vector<LightSurfaces>  lights;
    std::vector<Surfaces>       surfaces;
};

struct SurfaceStruct
{
    private:
    static constexpr auto PPS = TracerConstants::MaxPrimBatchPerSurface;
    public:
    static constexpr size_t MATERIAL_INDEX = 0;
    static constexpr size_t PRIM_INDEX = 1;
    using IdPair        = Tuple<uint32_t, uint32_t>;
    using IdPairList    = std::array<IdPair, PPS>;
    using TextureList   = std::array<Optional<SceneTexId>, PPS>;
    using CullList      = std::array<bool, PPS>;

    //
    uint32_t        transformId;
    IdPairList      matPrimBatchPairs;
    TextureList     alphaMaps;
    CullList        doCullBackFace;
    int8_t          pairCount;
};
using SceneSurfList = std::vector<SurfaceStruct>;

struct EndpointSurfaceStruct
{
    uint32_t        mediumId;
    uint32_t        transformId;
};

struct LightSurfaceStruct : public EndpointSurfaceStruct
{
    uint32_t lightId;
};
using SceneLightSurfList = std::vector<LightSurfaceStruct>;

struct CameraSurfaceStruct : public EndpointSurfaceStruct
{
    uint32_t cameraId;
};
using SceneCamSurfList = std::vector<CameraSurfaceStruct>;

// Json converters
void from_json(const nlohmann::json&, SceneTexId&);
void from_json(const nlohmann::json&, SurfaceStruct&);
void from_json(const nlohmann::json&, LightSurfaceStruct&);
void from_json(const nlohmann::json&, CameraSurfaceStruct&);
void from_json(const nlohmann::json&, MRayTextureEdgeResolveEnum&);
void from_json(const nlohmann::json&, MRayTextureInterpEnum&);
void from_json(const nlohmann::json&, MRayColorSpaceEnum&);
void from_json(const nlohmann::json&, MRayTextureReadMode&);
void from_json(const nlohmann::json&, ImageSubChannelType&);

class JsonNode
{
    private:
    const nlohmann::json*           node;
    bool                            isMultiNode;
    uint32_t                        innerIndex;

    public:
                            JsonNode(const nlohmann::json& node,
                                     uint32_t innerIndex = 0);

    // Id-based comparison
    auto                    operator<=>(const JsonNode other) const;


    const nlohmann::json&   RawNode() const;
    std::string_view        Type() const;
    std::string_view        Tag() const;
    uint32_t                Id() const;
    // Inner node unspecific data access
    template<class T>
    T                       CommonData(std::string_view name) const;
    template<class T>
    TransientData           CommonDataArray(std::string_view name) const;

    // Inner index related size checking
    size_t                  CheckDataArraySize(std::string_view name) const;
    size_t                  CheckOptionalDataArraySize(std::string_view name) const;
    bool                    CheckOptionalData(std::string_view name) const;
    // Inner index related data loading
    template<class T>
    T                       AccessData(std::string_view name) const;
    template<class T>
    TransientData           AccessDataArray(std::string_view name) const;
    // Optional Data
    template<class T>
    Optional<T>             AccessOptionalData(std::string_view name) const;
    template<class T>
    Optional<TransientData> AccessOptionalDataArray(std::string_view name) const;
    // Texturable (either data T, or texture struct)
    template<class T>
    Variant<SceneTexId, T>  AccessTexturableData(std::string_view name) const;
    SceneTexId              AccessTexture(std::string_view name) const;
    Optional<SceneTexId>    AccessOptionalTexture(std::string_view name) const;
};

#include "JsonNode.hpp"