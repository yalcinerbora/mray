#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>

#include "Device/GPUSystem.h"

#include "Tracer/PrimitiveDefaultTriangle.h"
#include "Tracer/LightC.h"
#include "Tracer/Lights.h"

#include "Core/Log.h"

// Meta Light Test Types
using LightGroupSkysphereCoOcto = LightGroupSkysphere<CoOctoCoordConverter>;
using LightGroupSkysphereSpherical = LightGroupSkysphere<SphericalCoordConverter>;

using TContextVariant = Variant<TransformContextIdentity,
                                TransformContextSingle>;
using LightVariantIn = Variant
<
    typename LightGroupSkysphereCoOcto:: template Light<TransformContextSingle>,
    typename LightGroupSkysphereCoOcto:: template Light<TransformContextIdentity>,
    typename LightGroupPrim<PrimGroupTriangle>:: template Light<TransformContextSingle>,
    typename LightGroupPrim<PrimGroupTriangle>:: template Light<TransformContextIdentity>
>;

using MetaLightList = MetaLightArray<TContextVariant, LightVariantIn>;
using MetaLight = typename MetaLightList::MetaLight;
using MetaLightView = typename MetaLightList::MetaLightView<Vector2>;


TEST(DefaultLights, DISABLED_Skysphere)
{
}

TEST(DefaultLights, MetaLight)
{
    GPUSystem system;
    // Generate some groups
    // PGs
    EmptyPrimGroup emptyPrimGroup(0u, system);
    PrimGroupTriangle triangleGroup(1u, system);
    // TGs
    TransformGroupIdentity identityTG(0u, system);
    TransformGroupSingle singleTG(1u, system);
    // LGs
    LightGroupPrim<PrimGroupTriangle> triangleLightGroup(0u, system, triangleGroup);
    LightGroupSkysphereCoOcto skysphereCOLightGroup(0u, system, emptyPrimGroup);

    // Add ids

    Span<const LightId> lightIds;
    Span<const PrimitiveId> primIds;
    Span<const TransformId> transformIds;

    MetaLightList lightList = MetaLightList(system);
    lightList.AddBatch(triangleLightGroup, identityTG,
                       primIds, lightIds, transformIds,
                       Vector2ui(0, 1));

    lightList.AddBatch(triangleLightGroup, singleTG,
                       primIds, lightIds, transformIds,
                       Vector2ui(0, 1));

    lightList.AddBatch(skysphereCOLightGroup, identityTG,
                       primIds, lightIds, transformIds,
                       Vector2ui(0, 1));

    lightList.AddBatch(skysphereCOLightGroup, singleTG,
                       primIds, lightIds, transformIds,
                       Vector2ui(0, 1));


}