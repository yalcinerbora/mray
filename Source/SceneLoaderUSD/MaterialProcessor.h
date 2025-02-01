#pragma once

#include "MRayUSDTypes.h"

#include "Core/Error.h"
#include "Core/TracerI.h"

#include <map>

#include <pxr/usd/usdShade/input.h>

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
    DIFFUSE,
    SPECULAR_DIFFUSE_COMBO,
    PURE_REFLECT,
    PURE_REFRACT
};

struct MRayUSDMatAlphaPack
{
    Optional<TextureId> alphaMap;
    MaterialId          materialId;
};

template<class T>
using MaterialTerminalVariant = std::variant<pxr::UsdPrim, T>;

// Hard-coded material definition
struct MRayUSDMaterialProps
{
    MRayUSDMaterialType type;
    MaterialTerminalVariant<pxr::GfVec3f> albedo;
    MaterialTerminalVariant<pxr::GfVec3f> normal;
    MaterialTerminalVariant<float> metallic;
    MaterialTerminalVariant<float> roughness;
    MaterialTerminalVariant<float> opacity;
    MaterialTerminalVariant<float> iorOrSpec;
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
    MRayUSDMaterialProps ResolveSingle(const MRayUSDBoundMaterial& m);

    public:
    bool warnMultipleTerminals = false;
    bool warnNonDirectTexConnection = false;
    bool warnSpecularMode = false;
    bool warnDielectricMaterialSucks = false;

    std::vector<MRayUSDMaterialProps>
    Resolve(const MRayUSDMaterialMap& uniqueMaterials);
};

MRayError ProcessUniqueMaterials(std::map<pxr::UsdPrim, MRayUSDMatAlphaPack>& outMaterials,
                                 TracerI& tracer, BS::thread_pool& threadPool,
                                 const MRayUSDMaterialMap& uniqueMaterials);