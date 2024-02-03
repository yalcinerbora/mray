#pragma once

#include <nlohmann/json.hpp>

#include "Core/Log.h"
#include "Core/Error.h"
#include "Core/Types.h"
#include "Core/TracerI.h"

#include "NodeNames.h"

using IdPair = Pair<uint32_t, uint32_t>;
using IdPairList = std::array<IdPair, TracerConstants::MaxPrimBatchPerSurface>;

enum class TextureChannelType
{
    R, G, B, A
};

enum class TextureAccessLayout
{
    R, G, B, A,
    RG, GB, BA,
    RGB, GBA,
    RGBA
};

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
    static constexpr int MATERIAL_INDEX = 0;
    static constexpr int PRIM_INDEX = 1;

    uint32_t        transformId;
    IdPairList      matPrimBatchPairs;
    int8_t          pairCount;
};

struct EndpointSurfaceStruct
{
    uint32_t        mediumId;
    uint32_t        transformId;
};

struct LightSurfaceStruct : public EndpointSurfaceStruct
{
    uint32_t lightId;
};

struct CameraSurfaceStruct : public EndpointSurfaceStruct
{
    uint32_t cameraId;
};

struct NodeTexStruct
{
    uint32_t            texId;
    TextureAccessLayout channelLayout;
};

// Json converters
void from_json(const nlohmann::json&, NodeTexStruct&);

void from_json(const nlohmann::json&, SurfaceStruct&);

void from_json(const nlohmann::json&, LightSurfaceStruct&);

void from_json(const nlohmann::json&, CameraSurfaceStruct&);

template<ArrayLikeC T>
void from_json(const nlohmann::json&, T&);

TextureAccessLayout LoadTextureAccessLayout(const nlohmann::json& node);

class MRayJsonNode
{
    private:
    const nlohmann::json&   node;
    bool                    isMultiNode;
    uint32_t                innerIndex;

    public:
                        MRayJsonNode(nlohmann::json& node,
                                     uint32_t innerIndex = 0);

    std::string_view    Type() const;
    std::string_view    Tag() const;
    uint32_t            Id() const;
    // Inner node unspecific data access
    template<class T>
    T                   CommonData(std::string_view name) const;
    template<class T>
    MRayInput           CommonDataArray(std::string_view name) const;
    // Inner index related data loading
    template<class T>
    T                   AccessData(std::string_view name) const;
    template<class T>
    MRayInput           AccessDataArray(std::string_view name) const;
    // Optional Data
    template<class T>
    Optional<T>         AccessOptionalData(std::string_view name) const;
    template<class T>
    Optional<MRayInput> AccessOptionalDataArray(std::string_view name) const;
    // Texturable (either data T, or texture struct)
    template<class T>
    Variant<NodeTexStruct, T>               AccessTexturableData(std::string_view name);
    template<class T>
    std::vector<Variant<NodeTexStruct, T>>  AccessTexturableDataArray(std::string_view name);
};

#include "JsonNode.hpp"