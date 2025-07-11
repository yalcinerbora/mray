#include <pxr/usd/usd/primDefinition.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>
#include <barrier>

#include "MaterialProcessor.h"

#include "ImageLoader/EntryPoint.h"

#include "Core/Algorithm.h"
#include "Core/TypeNameGenerators.h"
#include "Core/ThreadPool.h"

using MatKeyValPair = std::pair<const pxr::UsdPrim*, const MRayUSDMaterialProps*>;

static const std::array MRayMaterialTypes =
{
    std::pair(MRayUSDMaterialType::DIFFUSE,                 "Lambert"),
    std::pair(MRayUSDMaterialType::SPECULAR_DIFFUSE_COMBO,  "Unreal"),
    std::pair(MRayUSDMaterialType::PURE_REFLECT,            "Reflect"),
    std::pair(MRayUSDMaterialType::PURE_REFRACT,            "Refract"),
};

// Usd have some "reflection" functionality (via macros),
// crashes intellisense though. Just elbow grease should suffice.
struct MRayUSDShadeTokensT
{
    pxr::TfToken diffuseColor           = pxr::TfToken("diffuseColor");
    pxr::TfToken emissiveColor          = pxr::TfToken("emissiveColor");
    pxr::TfToken useSpecularWorkflow    = pxr::TfToken("useSpecularWorkflow");
    pxr::TfToken specularColor          = pxr::TfToken("specularColor");
    pxr::TfToken metallic               = pxr::TfToken("metallic");
    pxr::TfToken roughness              = pxr::TfToken("roughness");
    pxr::TfToken clearcoat              = pxr::TfToken("clearcoat");
    pxr::TfToken clearcoatRoughness     = pxr::TfToken("clearcoatRoughness");
    pxr::TfToken opacity                = pxr::TfToken("opacity");
    pxr::TfToken opacityMode            = pxr::TfToken("opacityMode");
    pxr::TfToken opacityThreshold       = pxr::TfToken("opacityThreshold");
    pxr::TfToken ior                    = pxr::TfToken("ior");
    pxr::TfToken normal                 = pxr::TfToken("normal");
    pxr::TfToken displacement           = pxr::TfToken("displacement");
    pxr::TfToken occlusion              = pxr::TfToken("occlusion");
    //
    pxr::TfToken transparent    = pxr::TfToken("transparent");
    pxr::TfToken presence       = pxr::TfToken("presence");
    //
    pxr::TfToken UsdUVTexture   = pxr::TfToken("UsdUVTexture");
    pxr::TfToken file           = pxr::TfToken("file");
    pxr::TfToken st             = pxr::TfToken("st");
    pxr::TfToken wrapS          = pxr::TfToken("wrapS");
    pxr::TfToken wrapT          = pxr::TfToken("wrapT");
    pxr::TfToken fallback       = pxr::TfToken("fallback");
    //
    pxr::TfToken black          = pxr::TfToken("black");
    pxr::TfToken clamp          = pxr::TfToken("clamp");
    pxr::TfToken repeat         = pxr::TfToken("repeat");
    pxr::TfToken mirror         = pxr::TfToken("mirror");
    pxr::TfToken useMetadata    = pxr::TfToken("useMetadata");
    //
    pxr::TfToken raw        = pxr::TfToken("raw");
    pxr::TfToken sRGB       = pxr::TfToken("sRGB");
    pxr::TfToken autoToken  = pxr::TfToken("auto");
    // Thse should be supported, but not yet
    pxr::TfToken scale              = pxr::TfToken("scale");
    pxr::TfToken bias               = pxr::TfToken("bias");
    pxr::TfToken sourceColorSpace   = pxr::TfToken("sourceColorSpace");
    //
    pxr::TfToken r      = pxr::TfToken("r");
    pxr::TfToken g      = pxr::TfToken("g");
    pxr::TfToken b      = pxr::TfToken("b");
    pxr::TfToken rgb    = pxr::TfToken("rgb");
    // Only float2 currently
    pxr::TfToken UsdPrimvarReader_float2    = pxr::TfToken("UsdPrimvarReader_float2");
    pxr::TfToken varname                    = pxr::TfToken("varname");
    // These will be ignored
    pxr::TfToken UsdTransform2d = pxr::TfToken("UsdTransform2d");
    pxr::TfToken in             = pxr::TfToken("in");
    pxr::TfToken rotation       = pxr::TfToken("rotation");
    pxr::TfToken translation    = pxr::TfToken("translation");
};

const auto& MRayUSDShadeTokens()
{
    // Object will be initialized at first call,
    // so we can get away with internal USD structs
    // are initialized or not.
    static const MRayUSDShadeTokensT Tokens;
    return Tokens;
}

template<class T>
T GetInput(const pxr::UsdShadeInput& input, const T& defaultValue)
{
    T result;
    if(input) input.Get<T>(&result);
    else result = defaultValue;
    return result;
}

template<class T>
ImageSubChannelType ResolveSubchannel(pxr::TfToken& outName)
{
    static_assert(std::is_same_v<T, pxr::GfVec3f> || std::is_same_v<T, float>,
                  "Not yet implemented!");

    const auto& tokens = MRayUSDShadeTokens();
    if constexpr(std::is_same_v<T, float>)
    {
        if(outName == tokens.rgb ||
           outName == tokens.r)
            return ImageSubChannelType::R;
        else if(outName == tokens.g)
            return ImageSubChannelType::G;
        else // if(outName == tokens.b)
            return ImageSubChannelType::B;
    }
    else if constexpr(std::is_same_v<T, pxr::GfVec3f>)
    {
        // TODO: Error out when channels mismatch
        return ImageSubChannelType::RGB;
    }
}


template<class T>
std::pair<T, Optional<TextureId>>
ReadUSDMatAttribute(const MaterialTerminalVariant<T>& mt,
                    const std::map<pxr::UsdPrim, TextureId>& texLookup)
{
    using MTT = MRayUSDTextureTerminal;
    using ResultT = std::pair<T, Optional<TextureId>>;

    ResultT result = ResultT(T(), std::nullopt);
    if(std::holds_alternative<MTT>(mt))
    {
        const pxr::UsdPrim& texName = std::get<MTT>(mt).texShaderNode;
        TextureId tId = texLookup.at(texName);
        result.second = tId;
    }
    else
    {
        result.first = std::get<T>(mt);
    }
    return result;
}

void CreateDiffuseMaterials(Span<MRayUSDMatAlphaPack> matIdsOut, TracerI& tracer,
                            Span<const MatKeyValPair> matPairs,
                            const std::map<pxr::UsdPrim, TextureId>& texLookup)
{
    assert(matIdsOut.size() == matPairs.size());
    using namespace TypeNameGen::Runtime;
    using enum MRayUSDMaterialType;
    std::string_view name = MRayMaterialTypes.at(static_cast<uint32_t>(DIFFUSE)).second;
    std::string decoratedName = AddMaterialPrefix(name);
    MatGroupId mgId = tracer.CreateMaterialGroup(decoratedName);
    const auto attributeInfoList = tracer.AttributeInfo(mgId);
    assert(attributeInfoList.size() == 2);

    // We are statically doing the attribute checking,
    // which is wrong (if tracer changes this should change as well)
    // Tracer will throw an exception in this case. This may be done
    // in more generic way later. (USD material can be anything, so it will
    // be hard)
    std::vector<AttributeCountList> attribCounts(matPairs.size());
    std::vector<Optional<TextureId>> albedoTextures(matPairs.size());
    std::vector<Optional<TextureId>> normalTextures(matPairs.size());
    std::vector<Optional<TextureId>> alphaMaps(matPairs.size());
    TransientData albedoBuffer(std::in_place_type<Vector3>, matPairs.size());
    albedoBuffer.ReserveAll();
    for(size_t i = 0; i < matPairs.size(); i++)
    {
        Span<Vector3> albedoSpan = albedoBuffer.AccessAs<Vector3>();
        const auto& pair = matPairs[i];

        auto[albedo, albedoTex] = ReadUSDMatAttribute(pair.second->albedo,texLookup);
        albedoTextures[i] = albedoTex;
        albedoSpan[i] = Vector3(albedo[0], albedo[1], albedo[2]);
        // Albedo is mandatory
        attribCounts[i].push_back(1);

        auto [normal, normalTex] = ReadUSDMatAttribute(pair.second->normal, texLookup);
        normalTextures[i] = normalTex;
        attribCounts[i].push_back(normalTex ? 1 : 0);

        auto [_, opacityTex] = ReadUSDMatAttribute(pair.second->opacity, texLookup);
        alphaMaps[i] = opacityTex;
    }

    std::vector<MaterialId> matIds = tracer.ReserveMaterials(mgId, attribCounts);
    tracer.CommitMatReservations(mgId);

    CommonIdRange range(std::bit_cast<CommonId>(matIds.front()),
                        std::bit_cast<CommonId>(matIds.back()));
    tracer.PushMatAttribute(mgId, range, 0, std::move(albedoBuffer),
                            albedoTextures);
    tracer.PushMatAttribute(mgId, range, 1,
                            TransientData(std::in_place_type<Vector3>, 0),
                            normalTextures);

    for(size_t i = 0; i < matIdsOut.size(); i++)
    {
        matIdsOut[i].materialId = matIds[i];
        matIdsOut[i].alphaMap = alphaMaps[i];
    }
}

void CreateUnrealMaterials(Span<MRayUSDMatAlphaPack> matIdsOut, TracerI& tracer,
                           Span<const MatKeyValPair> matPairs,
                           const std::map<pxr::UsdPrim, TextureId>& texLookup)
{
    assert(matIdsOut.size() == matPairs.size());
    using namespace TypeNameGen::Runtime;
    using enum MRayUSDMaterialType;
    std::string_view name = MRayMaterialTypes.at(static_cast<uint32_t>(SPECULAR_DIFFUSE_COMBO)).second;
    std::string decoratedName = AddMaterialPrefix(name);
    MatGroupId mgId = tracer.CreateMaterialGroup(decoratedName);
    const auto attributeInfoList = tracer.AttributeInfo(mgId);
    assert(attributeInfoList.size() == 5);

    // We are statically doing the attribute checking,
    // which is wrong (if tracer changes this should change as well)
    // Tracer will throw an exception in this case. This may be done
    // in more generic way later. (USD material can be anything, so it will
    // be hard)
    std::vector<AttributeCountList> attribCounts(matPairs.size());
    std::vector<Optional<TextureId>> albedoTextures(matPairs.size());
    std::vector<Optional<TextureId>> normalTextures(matPairs.size());
    std::vector<Optional<TextureId>> roughnessTextures(matPairs.size());
    std::vector<Optional<TextureId>> specularTextures(matPairs.size());
    std::vector<Optional<TextureId>> metallicTextures(matPairs.size());
    std::vector<Optional<TextureId>> alphaMaps(matPairs.size());

    TransientData albedoBuffer(std::in_place_type<Vector3>, matPairs.size());
    TransientData roughnessBuffer(std::in_place_type<Float>, matPairs.size());
    TransientData specularBuffer(std::in_place_type<Float>, matPairs.size());
    TransientData metallicBuffer(std::in_place_type<Float>, matPairs.size());
    albedoBuffer.ReserveAll();
    roughnessBuffer.ReserveAll();
    specularBuffer.ReserveAll();
    metallicBuffer.ReserveAll();
    Span<Vector3> albedoSpan = albedoBuffer.AccessAs<Vector3>();
    Span<Float> roughnessSpan = roughnessBuffer.AccessAs<Float>();
    Span<Float> specularSpan = specularBuffer.AccessAs<Float>();
    Span<Float> metallicSpan = metallicBuffer.AccessAs<Float>();
    for(size_t i = 0; i < matPairs.size(); i++)
    {
        const auto& pair = matPairs[i];
        // Albedo
        auto [albedo, albedoTex] = ReadUSDMatAttribute(pair.second->albedo, texLookup);
        albedoTextures[i] = albedoTex;
        albedoSpan[i] = Vector3(albedo[0], albedo[1], albedo[2]);
        attribCounts[i].push_back(1);
        // Normal
        auto [normal, normalTex] = ReadUSDMatAttribute(pair.second->normal, texLookup);
        normalTextures[i] = normalTex;
        attribCounts[i].push_back(normalTex ? 1 : 0);
        // Roughness
        auto [roughness, roughnessTex] = ReadUSDMatAttribute(pair.second->roughness, texLookup);
        roughnessTextures[i] = roughnessTex;
        roughnessSpan[i] = roughness;
        attribCounts[i].push_back(1);
        // Specular
        auto [specular, specularTex] = ReadUSDMatAttribute(pair.second->iorOrSpec, texLookup);
        specularTextures[i] = specularTex;
        specularSpan[i] = specular;
        attribCounts[i].push_back(1);
        // Metallic
        auto [metallic, metallicTex] = ReadUSDMatAttribute(pair.second->metallic, texLookup);
        metallicTextures[i] = metallicTex;
        metallicSpan[i] = metallic;
        attribCounts[i].push_back(1);

        auto [_, opacityTex] = ReadUSDMatAttribute(pair.second->opacity, texLookup);
        alphaMaps[i] = opacityTex;
    }

    std::vector<MaterialId> matIds = tracer.ReserveMaterials(mgId, attribCounts);
    tracer.CommitMatReservations(mgId);

    CommonIdRange range(std::bit_cast<CommonId>(matIds.front()),
                        std::bit_cast<CommonId>(matIds.back()));
    tracer.PushMatAttribute(mgId, range, 0, std::move(albedoBuffer),
                            albedoTextures);
    tracer.PushMatAttribute(mgId, range, 1,
                            TransientData(std::in_place_type<Vector3>, 0),
                            normalTextures);
    tracer.PushMatAttribute(mgId, range, 2, std::move(roughnessBuffer),
                            roughnessTextures);
    tracer.PushMatAttribute(mgId, range, 3, std::move(specularBuffer),
                            specularTextures);
    tracer.PushMatAttribute(mgId, range, 4, std::move(metallicBuffer),
                            metallicTextures);

    for(size_t i = 0; i < matIdsOut.size(); i++)
    {
        matIdsOut[i].materialId = matIds[i];
        matIdsOut[i].alphaMap = alphaMaps[i];
    }
}

void CreateRefractMaterials(Span<MRayUSDMatAlphaPack> matIdsOut, TracerI& tracer,
                            Span<const MatKeyValPair> matPairs,
                            const std::map<pxr::UsdPrim, TextureId>& texLookup)
{
    assert(matIdsOut.size() == matPairs.size());
    using namespace TypeNameGen::Runtime;
    using enum MRayUSDMaterialType;
    std::string_view name = MRayMaterialTypes.at(static_cast<uint32_t>(PURE_REFRACT)).second;
    std::string decoratedName = AddMaterialPrefix(name);
    MatGroupId mgId = tracer.CreateMaterialGroup(decoratedName);
    const auto attributeInfoList = tracer.AttributeInfo(mgId);
    assert(attributeInfoList.size() == 2);

    // We are statically doing the attribute checking,
    // which is wrong (if tracer changes this should change as well)
    // Tracer will throw an exception in this case. This may be done
    // in more generic way later. (USD material can be anything, so it will
    // be hard)
    std::vector<AttributeCountList> attribCounts(matPairs.size());
    TransientData cauchyFrontBuffer(std::in_place_type<Vector3>, matPairs.size());
    TransientData cauchyBackBuffer(std::in_place_type<Vector3>, matPairs.size());
    cauchyBackBuffer.ReserveAll();
    cauchyFrontBuffer.ReserveAll();
    for(size_t i = 0; i < matPairs.size(); i++)
    {
        Span<Vector3> iorBackSpan = cauchyBackBuffer.AccessAs<Vector3>();
        Span<Vector3> iorFrontSpan = cauchyFrontBuffer.AccessAs<Vector3>();
        const auto& pair = matPairs[i];
        auto [ior, iorTex] = ReadUSDMatAttribute(pair.second->iorOrSpec, texLookup);
        assert(!iorTex.has_value());

        iorBackSpan[i] = Vector3(ior, 0, 0);
        attribCounts[i].push_back(1);
        // In USD you cannot set front facing index of refraction directly
        // Just give 1 for now
        iorFrontSpan[i] = Vector3(1, 0, 0);
        attribCounts[i].push_back(1);
    }

    std::vector<MaterialId> matIds = tracer.ReserveMaterials(mgId, attribCounts);
    tracer.CommitMatReservations(mgId);

    CommonIdRange range(std::bit_cast<CommonId>(matIds.front()),
                        std::bit_cast<CommonId>(matIds.back()));
    tracer.PushMatAttribute(mgId, range, 0, std::move(cauchyBackBuffer));
    tracer.PushMatAttribute(mgId, range, 1, std::move(cauchyFrontBuffer));
    for(size_t i = 0; i < matIdsOut.size(); i++)
    {
        matIdsOut[i].materialId = matIds[i];
        matIdsOut[i].alphaMap = std::nullopt;
    }
}

void CreateReflectMaterials(Span<MRayUSDMatAlphaPack> matIdsOut, TracerI& tracer,
                            Span<const MatKeyValPair> matPairs,
                            const std::map<pxr::UsdPrim, TextureId>& texLookup)
{
    assert(matIdsOut.size() == matPairs.size());
    using namespace TypeNameGen::Runtime;
    using enum MRayUSDMaterialType;
    std::string_view name = MRayMaterialTypes.at(static_cast<uint32_t>(PURE_REFLECT)).second;
    std::string decoratedName = AddMaterialPrefix(name);
    MatGroupId mgId = tracer.CreateMaterialGroup(decoratedName);
    const auto attributeInfoList = tracer.AttributeInfo(mgId);
    assert(attributeInfoList.size() == 0);

    // We are statically doing the attribute checking,
    // which is wrong (if tracer changes this should change as well)
    // Tracer will throw an exception in this case. This may be done
    // in more generic way later. (USD material can be anything, so it will
    // be hard)
    std::vector<AttributeCountList> attribCounts(matPairs.size());
    std::vector<MaterialId> matIds = tracer.ReserveMaterials(mgId, attribCounts);
    std::vector<Optional<TextureId>> alphaMaps(matPairs.size());
    for(size_t i = 0; i < matPairs.size(); i++)
    {
        const auto& pair = matPairs[i];
        auto [_, opacityTex] = ReadUSDMatAttribute(pair.second->opacity, texLookup);
        alphaMaps[i] = opacityTex;
    }

    tracer.CommitMatReservations(mgId);
    for(size_t i = 0; i < matIdsOut.size(); i++)
    {
        matIdsOut[i].materialId = matIds[i];
        matIdsOut[i].alphaMap = alphaMaps[i];
    }
}

MRayUSDTexture MaterialConverter::ReadTextureNode(const pxr::UsdPrim& texNodePrim,
                                                  ImageSubChannelType imageSubChannel,
                                                  bool isColor, bool isNormal)
{
    const auto& tokens = MRayUSDShadeTokens();
    auto WrapToEdgeResolve = [&tokens](const pxr::TfToken& usdWrap)
    {
        // TODO: Warn about unsupported ops
        if(usdWrap == tokens.useMetadata||
           usdWrap == tokens.repeat ||
           usdWrap == tokens.black)
            return MRayTextureEdgeResolveEnum::MR_WRAP;
        else if(usdWrap == tokens.mirror)
            return MRayTextureEdgeResolveEnum::MR_MIRROR;
        else if(usdWrap == tokens.clamp)
            return MRayTextureEdgeResolveEnum::MR_CLAMP;
        return MRayTextureEdgeResolveEnum::MR_END;
    };
    auto ConvertColorSpace = [&tokens](const pxr::TfToken& colorSpace)
    {
        if(colorSpace == tokens.sRGB)
            return std::pair(MRayColorSpaceEnum::MR_REC_709, Float(2.2));
        else
            return std::pair(MRayColorSpaceEnum::MR_DEFAULT, Float(1));
    };

    pxr::UsdShadeShader tNode(texNodePrim);
    // TODO: Ignoring wrapT, MRay only have single parameter to resolve
    // everything (due to MRay being lazy)
    auto wrapS = GetInput(tNode.GetInput(tokens.wrapS), tokens.repeat);
    //auto wrapT = GetInput(tNode.GetInput(tokens.wrapT), tokens.repeat);
    auto csConverted = ConvertColorSpace(GetInput(tNode.GetInput(tokens.sourceColorSpace),
                                                  tokens.autoToken));

    // Resolve file
    auto fileSdf = GetInput(tNode.GetInput(tokens.file), pxr::SdfAssetPath());
    std::string file = fileSdf.GetResolvedPath();
    assert(!file.empty());
    // TODO: load "st" and check if it is properly connected
    return MRayUSDTexture
    {
        .absoluteFilePath = file,
        .imageSubChannel = imageSubChannel,
        .isNormal = isNormal,
        .params = MRayTextureParameters
        {
            .pixelType = MRayPixelType<MRayPixelEnum::MR_R_HALF>(),
            .colorSpace = csConverted.first,
            .gamma = csConverted.second,
            .ignoreResClamp = false,
            .isColor = isColor ? AttributeIsColor::IS_COLOR
                               : AttributeIsColor::IS_PURE_DATA,
            // TODO: load "st" and check if it is properly connected
            .edgeResolve = WrapToEdgeResolve(wrapS),
            .interpolation = MRayTextureInterpEnum::MR_LINEAR,
            .readMode = MRayTextureReadMode::MR_PASSTHROUGH
        }
    };
}

template<class T>
MaterialTerminalVariant<T>
MaterialConverter::PunchThroughGraphFindTexture(const auto& inputOutput, const T& defaultValue)
{
    auto attribList = pxr::UsdShadeUtils::GetValueProducingAttributes(inputOutput);
    //
    if(attribList.size() > 1) warnMultipleTerminals = true;
    //
    pxr::TfToken name;
    pxr::UsdPrim prim = attribList[0].GetPrim();
    if(prim.IsA<pxr::UsdShadeShader>() &&
       pxr::UsdShadeShader(prim).GetShaderId(&name) &&
       name == MRayUSDShadeTokens().UsdUVTexture)
    {
        // Ladies and gentlemen, we got him
        // Resolve the channel read
        auto outName = pxr::UsdShadeOutput(attribList[0]).GetBaseName();
        return MRayUSDTextureTerminal
        {
            .texShaderNode = prim,
            .channelRead = ResolveSubchannel<T>(outName)
        };
    }
    else if constexpr(std::is_same_v<decltype(inputOutput), pxr::UsdShadeInput>)
    {
        warnNonDirectTexConnection = true;
        return PunchThroughGraphFindTexture(pxr::UsdShadeOutput(attribList[0]),
                                            defaultValue);
    }
    else if constexpr(std::is_same_v<decltype(inputOutput), pxr::UsdShadeOutput>)
    {
        warnNonDirectTexConnection = true;
        return PunchThroughGraphFindTexture(pxr::UsdShadeInput(attribList[0]),
                                            defaultValue);
    }
    return defaultValue;
}

template<class T>
MaterialTerminalVariant<T>
MaterialConverter::GetTexturedAttribute(const pxr::UsdShadeInput& input,
                                        const T& defaultValue)
{
    if(!input) return defaultValue;
    // Punch-through the graph and find the connected texture
    // Or directly return the the value
    if(!input.HasConnectedSource())
    {
        T result; input.Get<T>(&result);
        return result;
    }
    else return PunchThroughGraphFindTexture<T>(input, defaultValue);
}

MRayUSDMaterialProps MaterialConverter::ResolveMatPropsSingle(const MRayUSDBoundMaterial& m)
{
    const auto& tokens = MRayUSDShadeTokens();
    if(std::holds_alternative<MRayUSDFallbackMaterial>(m))
    {
        return MRayUSDMaterialProps
        {
            .type = MRayUSDMaterialType::DIFFUSE,
            .albedo = std::get<MRayUSDFallbackMaterial>(m).color
        };
    }
    // Little bit more complex, try to check roughness etc
    // to create
    pxr::UsdShadeMaterial mat(std::get<pxr::UsdPrim>(m));
    pxr::UsdShadeShader shader = mat.ComputeSurfaceSource();
    //
    // Warn if specular
    if(GetInput(shader.GetInput(tokens.useSpecularWorkflow), 0) == 1)
    {
        warnSpecularMode = true;
    }

    // Get Opacity mode
    auto opacityMode = GetInput(shader.GetInput(tokens.opacityMode), tokens.transparent);
    bool dielectric = (opacityMode == tokens.transparent);
    // Get props
    auto metallic   = GetTexturedAttribute(shader.GetInput(tokens.metallic), 0.0f);
    auto roughness  = GetTexturedAttribute(shader.GetInput(tokens.roughness), 0.5f);
    auto albedo     = GetTexturedAttribute(shader.GetInput(tokens.diffuseColor), pxr::GfVec3f(0.5f));
    auto normal     = GetTexturedAttribute(shader.GetInput(tokens.normal), pxr::GfVec3f(0));
    auto opacity    = GetTexturedAttribute(shader.GetInput(tokens.opacity), 1.0f);
    auto iorOrSpec  = GetTexturedAttribute(shader.GetInput(tokens.ior), 1.0f);

    using MTT = MRayUSDTextureTerminal;
    bool nonMetal = (!std::holds_alternative<MTT>(metallic) &&
                     std::get<float>(metallic) == 0.0f);
    //bool smooth = (!std::holds_alternative<MTT>(roughness) &&
    //               std::get<float>(roughness) == 0.0f);
    bool rough = (!std::holds_alternative<MTT>(roughness) &&
                   std::get<float>(roughness) == 1.0f);
    //
    MRayUSDMaterialType type;
    if(dielectric && nonMetal &&
       std::holds_alternative<float>(opacity) &&
       std::get<float>(opacity) < 1.0f)
    {
        warnDielectricMaterialSucks = true;
        type = MRayUSDMaterialType::PURE_REFRACT;
    }
    else if(rough && nonMetal)
        type = MRayUSDMaterialType::DIFFUSE;
    else
        type = MRayUSDMaterialType::SPECULAR_DIFFUSE_COMBO;

    // Convert ior to "specularity" if material is opaque
    if(type == MRayUSDMaterialType::SPECULAR_DIFFUSE_COMBO &&
       std::holds_alternative<float>(iorOrSpec))
    {
        static constexpr float MaxF0 = 0.08f;
        static constexpr float MaxF0Recip = 1.0f / MaxF0;
        float newIoR = std::get<float>(iorOrSpec);
        newIoR = (1.0f - newIoR) / (1.0f + newIoR);
        // Convert IoR to "specularity" parameter of the BRDF
        newIoR = newIoR * newIoR * MaxF0Recip;
        iorOrSpec = newIoR;
    }

    return MRayUSDMaterialProps
    {
        .type = type,
        .albedo = albedo,
        .normal = normal,
        .metallic = metallic,
        .roughness = roughness,
        .opacity = opacity,
        .iorOrSpec = iorOrSpec
    };
}

std::vector<MRayUSDMaterialProps>
MaterialConverter::ResolveMatProps(const MRayUSDMaterialMap& uniqueMaterials)
{
    std::vector<MRayUSDMaterialProps> matProps;
    matProps.reserve(uniqueMaterials.size());
    for(const auto& m : uniqueMaterials)
    {
        matProps.push_back(ResolveMatPropsSingle(m.second));
    }
    return matProps;
}

FlatSet<std::pair<pxr::UsdPrim, MRayUSDTexture>>
MaterialConverter::ResolveTextures(const std::vector<MRayUSDMaterialProps>& props)
{
    FlatSet<std::pair<pxr::UsdPrim, MRayUSDTexture>> textures;
    auto ReadAndEmplace = [this, &textures](const auto& t, bool isColor,
                                            bool isNormal = false)
    {
        using MTT = MRayUSDTextureTerminal;
        if(std::holds_alternative<MTT>(t))
        {
            const auto& texTerminal = std::get<MTT>(t);
            textures.emplace
            (
                texTerminal.texShaderNode,
                ReadTextureNode(texTerminal.texShaderNode,
                                texTerminal.channelRead,
                                isColor, isNormal)
            );
        }
    };
    //
    for(const auto& p : props)
    {
        ReadAndEmplace(p.albedo, true);
        ReadAndEmplace(p.normal, false, true);
        ReadAndEmplace(p.metallic, false);
        ReadAndEmplace(p.roughness, false);
        ReadAndEmplace(p.opacity, false);
        ReadAndEmplace(p.iorOrSpec, false);
    }
    return textures;
}

MRayError MaterialConverter::LoadTextures(std::map<pxr::UsdPrim, TextureId>& result,
                                          FlatSet<std::pair<pxr::UsdPrim, MRayUSDTexture>>&& tex,
                                          TracerI& tracer, ThreadPool& threadPool)
{
    auto flatTextures = std::move(tex).extract();
    uint32_t textureCount = static_cast<uint32_t>(flatTextures.size());
    std::vector<std::pair<pxr::UsdPrim, TextureId>> texIds(textureCount);

    const auto BarrierFunc = [&]() noexcept
    {
        try{ tracer.CommitTextures();}
        catch(const MRayError& e)
        {
            // Fatally crash here, barrier's sync
            // do not allowed to have exceptions
            MRAY_ERROR_LOG("[Tracer]: {}", e.GetError());
            std::exit(1);
        }
    };
    uint32_t threadCount = std::min(threadPool.ThreadCount(),
                                    textureCount);

    using Barrier = std::barrier<decltype(BarrierFunc)>;
    auto barrier = std::make_shared<Barrier>(threadCount, BarrierFunc);

    // Implicitly Load the ImageLoader via C++ API
    // Convert it to shared pointer, we will copy these to threads
    // (We wait the threads before the scope exit, but just to be safe.
    // later we may enable more asynchronicity)
    auto imgLoader = std::shared_ptr(CreateImageLoader(false));
    ErrorList errors;

    const auto THRD_ProcessTextures =
    [&, imgLoader](uint32_t start, uint32_t end) -> void
    {
        MRayError err = MRayError::OK;
        // Subset the data to per core
        std::span myTextureRange(flatTextures.begin() + start, end - start);
        std::span myIdRange(texIds.begin() + start, end - start);
        //
        std::vector<ImageFilePtr> localTexFiles;
        localTexFiles.reserve(end - start);
        // Find out the texture sizes
        // .....................................
        // TODO: check if the twice opening is a bottleneck?
        // We are opening here to determining size/format
        // and on the other iteration we actual memcpy it
        bool barrierPassed = false;
        try
        {
            for(size_t i = 0; i < end - start; i++)
            {
                const auto& myTex = myTextureRange[i];
                auto& myId = myIdRange[i];

                using enum ImageIOFlags::F;
                ImageIOFlags flags;
                flags[DISREGARD_COLOR_SPACE] =
                    (myTex.second.params.isColor == AttributeIsColor::IS_PURE_DATA);
                flags[LOAD_AS_SIGNED] = myTex.second.isNormal;
                // Always do channel expand (HW limitation)
                flags[TRY_3C_4C_CONVERSION] = true;

                auto imgFileE = imgLoader->OpenFile(myTex.second.absoluteFilePath,
                                                    myTex.second.imageSubChannel,
                                                    flags);
                if(imgFileE.has_error())
                {
                    barrier->arrive_and_drop();
                    errors.AddException(MRayError(imgFileE.error()));
                    return;
                }
                localTexFiles.emplace_back(std::move(imgFileE.value()));

                Expected<ImageHeader> headerE = localTexFiles.back()->ReadHeader();
                if(!headerE.has_value())
                {
                    barrier->arrive_and_drop();
                    errors.AddException(MRayError(headerE.error()));
                    return;
                }

                const auto& header = headerE.value();
                using enum AttributeIsColor;
                // Most of the params are directly coming from the USD
                // But set the pixel type and override colorspace if
                // USD's value does not exist.
                MRayTextureParameters params = myTex.second.params;
                params.readMode = header.readMode;
                //
                params.pixelType = header.pixelType;
                if(params.colorSpace == MRayColorSpaceEnum::MR_DEFAULT &&
                   header.colorSpace.second != MRayColorSpaceEnum::MR_DEFAULT)
                {
                    params.colorSpace = header.colorSpace.second;
                    params.gamma = header.colorSpace.first;
                }
                TextureId tId;
                tId = tracer.CreateTexture2D(Vector2ui(header.dimensions),
                                             header.mipCount,
                                             params);
                myId = std::make_pair(myTex.first, tId);
            }
            // Barrier code is invoked, and all textures are allocated
            barrier->arrive_and_wait();
            barrierPassed = true;

            for(size_t i = 0; i < end - start; i++)
            {
                const auto& myId = myIdRange[i];
                Expected<Image> imgE = localTexFiles[i]->ReadImage();
                if(!imgE.has_value())
                {
                    errors.AddException(MRayError(imgE.error()));
                    return;
                }
                auto& img = imgE.value();
                // Send data mip by mip
                for(uint32_t j = 0; j < img.header.mipCount; j++)
                    tracer.PushTextureData(myId.second, j,
                                           std::move(img.imgData[j].pixels));
            }
        }
        catch(MRayError& e)
        {
            if(!barrierPassed) barrier->arrive_and_drop();
            errors.AddException(MRayError(e));
            return;
        }
        catch(std::exception& e)
        {
            if(!barrierPassed) barrier->arrive_and_drop();
            errors.AddException(MRayError("Unknown Error ({})", std::string(e.what())));
            return;
        }
    };
    auto future = threadPool.SubmitBlocks(uint32_t(textureCount),
                                         THRD_ProcessTextures, threadCount);
    future.WaitAll();
    MRayError err = MRayError::OK;
    bool isFirst = true;
    for(const auto& threadErr : errors.exceptions)
    {
        if(threadErr && isFirst)
            err = threadErr;
        else if(threadErr)
            err.AppendInfo(threadErr.GetError());
    }
    if(err) return err;

    // Copy to map
    for(const auto& t : texIds)
    {
        [[maybe_unused]]
        const auto& [_, inserted] = result.emplace(t.first, t.second);
        assert(inserted);
    }

    return MRayError::OK;
}

std::map<pxr::UsdPrim, MRayUSDMatAlphaPack>
MaterialConverter::CreateMaterials(TracerI& tracer,
                                   const std::vector<pxr::UsdPrim>& flatMatNames,
                                   const std::vector<MRayUSDMaterialProps>& flatMatProps,
                                   const std::map<pxr::UsdPrim, TextureId>& texLookup)
{
    std::vector<MatKeyValPair> combinedMatKVPairs;
    combinedMatKVPairs.reserve(flatMatNames.size());
    assert(flatMatNames.size() == flatMatProps.size());
    for(size_t i = 0; i < flatMatNames.size(); i++)
        combinedMatKVPairs.emplace_back(&flatMatNames[i], &flatMatProps[i]);
    //
    auto TypePartition = [](const MatKeyValPair& l,
                            const MatKeyValPair& r)
    {
        return l.second->type < r.second->type;
    };

    std::sort(combinedMatKVPairs.begin(), combinedMatKVPairs.end(), TypePartition);
    std::vector<Vector2ul> ranges = Algo::PartitionRange(combinedMatKVPairs.begin(),
                                                         combinedMatKVPairs.end(),
                                                         TypePartition);
    std::vector<MRayUSDMatAlphaPack> flatMaterialIds(combinedMatKVPairs.size());
    for(const Vector2ul& range : ranges)
    {
        Span<const MatKeyValPair> localKVPairs(combinedMatKVPairs.begin() + std::ptrdiff_t(range[0]),
                                               range[1] - range[0]);
        Span<MRayUSDMatAlphaPack> idOuts(flatMaterialIds.begin() + std::ptrdiff_t(range[0]),
                                         range[1] - range[0]);
        switch(localKVPairs.front().second->type)
        {
            using enum MRayUSDMaterialType;
            case DIFFUSE:
                CreateDiffuseMaterials(idOuts, tracer, localKVPairs, texLookup);
                break;
            case SPECULAR_DIFFUSE_COMBO:
                CreateUnrealMaterials(idOuts, tracer, localKVPairs, texLookup);
                break;
            case PURE_REFLECT:
                CreateReflectMaterials(idOuts, tracer, localKVPairs, texLookup);
                break;
            case PURE_REFRACT:
                CreateRefractMaterials(idOuts, tracer, localKVPairs, texLookup);
                break;
            default: break;
        }
    }
    // Convert to map
    std::map<pxr::UsdPrim, MRayUSDMatAlphaPack> result;
    for(size_t i = 0; i < flatMaterialIds.size(); i++)
    {
        result.emplace(*combinedMatKVPairs[i].first,
                       flatMaterialIds[i]);
    }
    return result;
}

void PrintMaterials(const std::vector<MRayUSDMaterialProps>& matPropList,
                    const std::vector<pxr::UsdPrim>& flatKeys)
{
    auto MatTypeToString = [](MRayUSDMaterialType t)
    {
        auto loc = std::find_if(MRayMaterialTypes.cbegin(),
                                MRayMaterialTypes.cend(),
        [t](const auto l)
        {
            return t == l.first;
        });
        if(loc != MRayMaterialTypes.cend())
            return loc->second;
        else return "";
    };
    auto AttributeToString = []<class T>(const MaterialTerminalVariant<T>& t)
    {
        if(std::holds_alternative<MRayUSDTextureTerminal>(t))
        {
            const auto& tt = std::get<MRayUSDTextureTerminal>(t);
            return MRAY_FORMAT("{}| {}",
                                static_cast<uint32_t>(tt.channelRead),
                                tt.texShaderNode.GetPath().GetString());
        }
        if constexpr(std::is_same_v<T, pxr::GfVec3f>)
        {
            T v = std::get<T>(t);
            return MRAY_FORMAT("[{}, {}, {}]", v[0], v[1], v[2]);
        }
        else
        {
            return MRAY_FORMAT("{}", std::get<T>(t));
        }
    };

    assert(flatKeys.size() == matPropList.size());
    MRAY_LOG("---------------------------------------");
    for(size_t i = 0; i < flatKeys.size(); i++)
    {
        MRAY_LOG("Name-{}", flatKeys[i].GetPath().GetString());

        const auto& l = matPropList[i];
        MRAY_LOG("Type      : {}", MatTypeToString(l.type));
        MRAY_LOG("Albedo    : {}", AttributeToString(l.albedo));
        MRAY_LOG("Normal    : {}", AttributeToString(l.normal));
        MRAY_LOG("Metallic  : {}", AttributeToString(l.metallic));
        MRAY_LOG("Roughness : {}", AttributeToString(l.roughness));
        MRAY_LOG("Opacity   : {}", AttributeToString(l.opacity));
        MRAY_LOG("IoR/Spec  : {}", AttributeToString(l.iorOrSpec));
        MRAY_LOG("---------------------------------------");
    }
}

MRayError ProcessUniqueMaterials(std::map<pxr::UsdPrim, MRayUSDMatAlphaPack>& outMaterials,
                                 std::map<pxr::UsdPrim, TextureId>& uniqueTextureIds,
                                 TracerI& tracer, ThreadPool& threadPool,
                                 const MRayUSDMaterialMap& uniqueMaterials,
                                 const std::map<pxr::UsdPrim, MRayUSDTexture>& extraTextures)
{
    std::vector<pxr::UsdPrim> flatKeys;
    flatKeys.reserve(uniqueMaterials.size());
    for(const auto& [name, _] : uniqueMaterials)
        flatKeys.push_back(name);

    MaterialConverter converter;
    auto matPropList = converter.ResolveMatProps(uniqueMaterials);

    //PrintMaterials(matPropList, flatKeys);

    // Now lets find out the textures
    auto uniqueTextures = converter.ResolveTextures(matPropList);
    for(const auto& e : extraTextures)
        uniqueTextures.emplace(e);

    MRayError err = converter.LoadTextures(uniqueTextureIds,
                                           std::move(uniqueTextures),
                                           tracer, threadPool);
    if(err) return err;
    outMaterials = converter.CreateMaterials(tracer, flatKeys, matPropList,
                                             uniqueTextureIds);

    if(converter.warnDielectricMaterialSucks)
        MRAY_WARNING_LOG("[MRayUSD]: One or more UsdPreviewSurface material(s) are detected "
                         "as Dielectric material(s). MRay only supports perfectly specular "
                         "dielectric materials. Roughness values are ignored.");
    if(converter.warnMultipleTerminals)
        MRAY_WARNING_LOG("[MRayUSD]: Some materials' input values have multiple "
                         "sources. MRay does not support shading graphs. The first "
                         "terminal texture is used.");
    if(converter.warnNonDirectTexConnection)
        MRAY_WARNING_LOG("[MRayUSD]: Some materials' input values are not directly "
                         "connected to a texture node. MRay does not support "
                         "shading graphs. Intermediate nodes are ignored.");
    if(converter.warnSpecularMode)
        MRAY_WARNING_LOG("[MRayUSD]: Some materials' have \"useSpecularWorkflow\" value "
                         "set to 1. MRay supports only metallic workflows. These "
                         "materials are not correctly read and may not be visually pleasing");

    return MRayError::OK;
}
