#pragma once

#include "MRayUSDTypes.h"

#include <map>
#include <pxr/usd/usdShade/input.h>

#include "Core/Error.h"
#include "Core/TracerI.h"

#include "ImageLoader/ImageLoaderI.h"

class TracerI;

// Materials are little bit complex, it will be handled differently,
// One std surface is "USDPreviewSurface" it is mega shader-like
// material. We will try to bisect it. These are the lookups that will be
// used.
//
// Obviously this is static currently. We will need to make this dynamic later
// (by making MRay to inform us with material types etc.)
// This will require quite a bit of design to support stuff, but we'll see.
enum class MRayUSDMaterialType
{
    DIFFUSE                 = 0,
    SPECULAR_DIFFUSE_COMBO  = 1,
    PURE_REFLECT            = 2,
    PURE_REFRACT            = 3
};

struct MRayUSDMatAlphaPack
{
    MatGroupId          groupId;
    MaterialId          materialId;
    Optional<TextureId> alphaMap;
};

struct MRayUSDTextureTerminal
{
    pxr::UsdPrim        texShaderNode;
    ImageSubChannelType channelRead;
};

template<class T>
using MaterialTerminalVariant = Variant<MRayUSDTextureTerminal, T>;

// Hard-coded material definition
struct MRayUSDMaterialProps
{
    MRayUSDMaterialType type;
    MaterialTerminalVariant<pxr::GfVec3f> albedo;
    MaterialTerminalVariant<pxr::GfVec3f> normal    = pxr::GfVec3f(0, 1, 1);
    MaterialTerminalVariant<float> metallic         = 0.0f;
    MaterialTerminalVariant<float> roughness        = 1.0f;
    MaterialTerminalVariant<float> opacity          = 1.0f;
    MaterialTerminalVariant<float> iorOrSpec        = 1.0f;
};

inline auto operator<=>(const MRayTextureParameters& l,
                        const MRayTextureParameters& r);

struct MRayUSDTexture
{
    std::string             absoluteFilePath;
    ImageSubChannelType     imageSubChannel;
    bool                    isNormal;
    MRayTextureParameters   params;
    //
    auto operator<=>(const MRayUSDTexture& t) const;
};

struct MaterialConverter
{
    private:
    template<class T>
    MaterialTerminalVariant<T>
    PunchThroughGraphFindTexture(const auto& inputOutput,
                                 const T& defaultValue);
    //
    template<class T>
    MaterialTerminalVariant<T>
    GetTexturedAttribute(const pxr::UsdShadeInput& input,
                         const T& defaultValue);
    //
    MRayUSDMaterialProps    ResolveMatPropsSingle(const MRayUSDBoundMaterial& m);
    MRayUSDTexture          ReadTextureNode(const pxr::UsdPrim& texNodePrim,
                                            ImageSubChannelType imageSubChannel,
                                            bool isColor, bool isNormal);

    public:
    bool warnMultipleTerminals = false;
    bool warnNonDirectTexConnection = false;
    bool warnSpecularMode = false;
    bool warnDielectricMaterialSucks = false;

    std::vector<MRayUSDMaterialProps>
    ResolveMatProps(const MRayUSDMaterialMap& uniqueMaterials);
    //
    FlatSet<Pair<pxr::UsdPrim, MRayUSDTexture>>
    ResolveTextures(const std::vector<MRayUSDMaterialProps>&);
    //
    MRayError LoadTextures(std::map<pxr::UsdPrim, TextureId>& result,
                           FlatSet<Pair<pxr::UsdPrim, MRayUSDTexture>>&& tex,
                           TracerI& tracer, ThreadPool& threadPool);

    std::map<pxr::UsdPrim, MRayUSDMatAlphaPack>
    CreateMaterials(TracerI& tracer,
                    const std::vector<pxr::UsdPrim>& flatMatNames,
                    const std::vector<MRayUSDMaterialProps>& flatMatProps,
                    const std::map<pxr::UsdPrim, TextureId>& loadedTextures);
};

MRayError ProcessUniqueMaterials(std::map<pxr::UsdPrim, MRayUSDMatAlphaPack>& outMaterials,
                                 std::map<pxr::UsdPrim, TextureId>& uniqueTextureIds,
                                 TracerI& tracer, ThreadPool& threadPool,
                                 const MRayUSDMaterialMap& uniqueMaterials,
                                 const std::map<pxr::UsdPrim, MRayUSDTexture>& extraTextures);

inline auto operator<=>(const MRayTextureParameters& l,
                        const MRayTextureParameters& r)
{
    #define MRAY_GEN_TUPLE(a) \
        Tuple(a.pixelType.Name(), a.colorSpace, a.gamma, \
              a.ignoreResClamp, a.isColor, a.edgeResolve, \
              a.interpolation, a.readMode)

    // Little bit of overkill but w/e
    return (MRAY_GEN_TUPLE(l) <=> MRAY_GEN_TUPLE(r));
    #undef MRAY_GEN_TUPLE
}

inline auto MRayUSDTexture::operator<=>(const MRayUSDTexture& t) const
{
    return (Tuple(absoluteFilePath, params) <=>
            Tuple(t.absoluteFilePath, t.params));
}