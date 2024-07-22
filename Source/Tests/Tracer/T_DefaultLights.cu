#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>

#include "Device/GPUSystem.h"

#include "Tracer/PrimitiveDefaultTriangle.h"
#include "Tracer/LightC.h"
#include "Tracer/LightsDefault.h"
#include "Tracer/MetaLight.h"

#include "Core/Log.h"

// Meta Light Test Types
using LightGroupSkysphereCoOcta = LightGroupSkysphere<CoOctaCoordConverter>;
using LightGroupSkysphereSpherical = LightGroupSkysphere<SphericalCoordConverter>;

using TContextVariant = Variant<TransformContextIdentity,
                                TransformContextSingle>;
using LightVariantIn = Variant
<
    typename LightGroupSkysphereCoOcta:: template Light<TransformContextSingle>,
    typename LightGroupSkysphereCoOcta:: template Light<TransformContextIdentity>,
    typename LightGroupPrim<PrimGroupTriangle>:: template Light<TransformContextSingle>,
    typename LightGroupPrim<PrimGroupTriangle>:: template Light<TransformContextIdentity>
>;

using MetaLightList = MetaLightArray<TContextVariant, LightVariantIn>;
using MetaLight = typename MetaLightList::MetaLight;
using MetaLightView = typename MetaLightList::MetaLightView<MetaHit>;


TEST(DefaultLights, DISABLED_Skysphere)
{
}

TEST(DefaultLights, DISABLED_MetaLight)
{
    GPUSystem system;
    // Generate some groups
    // PGs
    PrimGroupEmpty emptyPrimGroup(0u, system);
    PrimGroupTriangle triangleGroup(1u, system);
    // TGs
    TransformGroupIdentity identityTG(0u, system);
    TransformGroupSingle singleTG(1u, system);
    // LGs
    LightGroupPrim<PrimGroupTriangle> triangleLightGroup(0u, system, {}, triangleGroup);
    LightGroupSkysphereCoOcta skysphereCOLightGroup(0u, system, {}, emptyPrimGroup);

    // Add ids

    Span<const LightKey> lightKeys;
    Span<const PrimitiveKey> primKeys;
    Span<const TransformKey> transformKeys;

    MetaLightList lightList = MetaLightList(system);
    lightList.AddBatch(triangleLightGroup, identityTG,
                       primKeys, lightKeys, transformKeys,
                       Vector2ui(0, 1));
    lightList.AddBatch(triangleLightGroup, singleTG,
                       primKeys, lightKeys, transformKeys,
                       Vector2ui(0, 1));

    lightList.AddBatch(skysphereCOLightGroup, identityTG,
                       primKeys, lightKeys, transformKeys,
                       Vector2ui(0, 1));

    lightList.AddBatch(skysphereCOLightGroup, singleTG,
                       primKeys, lightKeys, transformKeys,
                       Vector2ui(0, 1));


}