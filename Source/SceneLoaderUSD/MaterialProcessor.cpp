#include "MaterialProcessor.h"

#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/shader.h>

static const std::array MRayMaterialTypes =
{
    std::pair(MRayUSDMaterialType::DIFFUSE,                 "Lambert"),
    std::pair(MRayUSDMaterialType::SPECULAR_DIFFUSE_COMBO,  "Unreal"),
    std::pair(MRayUSDMaterialType::PURE_REFLECT,            "Reflect"),
    std::pair(MRayUSDMaterialType::PURE_REFRACT,            "Refract"),
};

// Usd have some "reflection" functionality (via macros),
// crashes intellisense though. Just elbow grease should suffice.
struct MRayUsdShadeTokensT
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
    pxr::TfToken transparent            = pxr::TfToken("transparent");
    pxr::TfToken presence               = pxr::TfToken("presence");
    //
    pxr::TfToken UsdUVTexture           = pxr::TfToken("UsdUVTexture");
    pxr::TfToken file                   = pxr::TfToken("file");
    pxr::TfToken st                     = pxr::TfToken("st");
    pxr::TfToken wrapS                  = pxr::TfToken("wrapS");
    pxr::TfToken wrapT                  = pxr::TfToken("wrapT");
    pxr::TfToken fallback               = pxr::TfToken("fallback");
    // Thse should be supported, but not yet
    pxr::TfToken scale                  = pxr::TfToken("scale");
    pxr::TfToken bias                   = pxr::TfToken("bias");
    pxr::TfToken sourceColorSpace       = pxr::TfToken("sourceColorSpace");
    // Only float2 currently
    pxr::TfToken UsdPrimvarReader_float2    = pxr::TfToken("UsdPrimvarReader_float2");
    pxr::TfToken varname                    = pxr::TfToken("varname");
    // These will be ignored
    pxr::TfToken UsdTransform2d         = pxr::TfToken("UsdTransform2d");
    pxr::TfToken in                     = pxr::TfToken("in");
    pxr::TfToken rotation               = pxr::TfToken("rotation");
    pxr::TfToken translation            = pxr::TfToken("translation");
};

const auto& MRayUSDShadeTokens()
{
    static const MRayUsdShadeTokensT TOKENS;
    return TOKENS;
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
MaterialTerminalVariant<T>
MaterialConverter::PunchThroughGraphFindTexture(const auto& inputOutput, const T& defaultValue)
{
    auto attribList = pxr::UsdShadeUtils::GetValueProducingAttributes(inputOutput);
    if(attribList.size() >= 1) warnMultipleTerminals = true;
    //
    pxr::TfToken name;
    pxr::UsdPrim prim = attribList[0].GetPrim();
    if(prim.IsA<pxr::UsdShadeShader>() &&
       pxr::UsdShadeShader(prim).GetShaderId(&name) &&
       name == MRayUSDShadeTokens().UsdUVTexture)
    {
        // Ladies and gentlemen, we got him
        return prim;
    }
    else if constexpr(std::is_same_v<decltype(inputOutput), pxr::UsdShadeInput>)
    {
        warnNonDirectTexConnection = true;
        return PunchThroughGraphFindTexture(pxr::UsdShadeOutput(attribList[0]));
    }
    else if constexpr(std::is_same_v<decltype(inputOutput), pxr::UsdShadeOutput>)
    {
        warnNonDirectTexConnection = true;
        return PunchThroughGraphFindTexture(pxr::UsdShadeInput(attribList[0]));
    }
    return defaultValue;
}

template<class T>
MaterialTerminalVariant<T>
MaterialConverter::GetTexturedAttribute(const pxr::UsdShadeInput& input,
                                        const T& defaultValue)
{
    // Punch thorugh the graph and find the connected texture
    // Or directly return the the value
    if(!input.HasConnectedSource())
    {
        if(input)
        {
            T result; input.Get<T>(&result);
            return result;
        }
        else return defaultValue;
    }
    else return PunchThroughGraphFindTexture<T>(input, defaultValue);
}

MRayUSDMaterialProps MaterialConverter::ResolveSingle(const MRayUSDBoundMaterial& m)
{
    MRayUSDMaterialType matType = MRayUSDMaterialType::DIFFUSE;
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
    auto roughness  = GetTexturedAttribute(shader.GetInput(tokens.roughness), 0.0f);
    auto albedo     = GetTexturedAttribute(shader.GetInput(tokens.diffuseColor), pxr::GfVec3f(0.5f));
    auto normal     = GetTexturedAttribute(shader.GetInput(tokens.normal), pxr::GfVec3f(0));
    auto opacity    = GetTexturedAttribute(shader.GetInput(tokens.opacity), 1.0f);
    auto iorOrSpec  = GetTexturedAttribute(shader.GetInput(tokens.ior), 1.0f);

    bool nonMetal = (!std::holds_alternative<pxr::UsdPrim>(metallic) &&
                     std::get<float>(metallic) == 0.0f);
    bool smooth = (!std::holds_alternative<pxr::UsdPrim>(roughness) &&
                   std::get<float>(roughness) == 0.0f);
    bool rough = (!std::holds_alternative<pxr::UsdPrim>(roughness) &&
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
MaterialConverter::Resolve(const MRayUSDMaterialMap& uniqueMaterials)
{
    std::vector<MRayUSDMaterialProps> result;
    result.reserve(uniqueMaterials.size());
    for(const auto& m : uniqueMaterials)
    {
        result.push_back(ResolveSingle(m.second));
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
        if(std::holds_alternative<pxr::UsdPrim>(t))
            return MRAY_FORMAT("{}", std::get<pxr::UsdPrim>(t).GetPath().GetString());

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
                                 TracerI& tracer, BS::thread_pool& threadPool,
                                 const MRayUSDMaterialMap& uniqueMaterials)
{
    std::vector<pxr::UsdPrim> flatKeys;
    flatKeys.reserve(uniqueMaterials.size());
    for(const auto [name, _] : uniqueMaterials)
        flatKeys.push_back(name);

    MaterialConverter converter;
    auto matPropList = converter.Resolve(uniqueMaterials);

    PrintMaterials(matPropList, flatKeys);

    if(converter.warnDielectricMaterialSucks)
        MRAY_WARNING_LOG("A");
    if(converter.warnMultipleTerminals)
        MRAY_WARNING_LOG("B");
    if(converter.warnNonDirectTexConnection)
        MRAY_WARNING_LOG("C");
    if(converter.warnSpecularMode)
        MRAY_WARNING_LOG("D");

    return MRayError::OK;
}