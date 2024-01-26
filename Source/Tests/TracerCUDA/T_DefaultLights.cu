#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>

#include "Device/GPUSystem.h"

#include "Tracer/PrimitiveDefaultTriangle.h"
#include "Tracer/LightC.h"
#include "Tracer/Lights.h"

#include "Core/Log.h"

TEST(DefaultLights, DISABLED_Skysphere)
{
}

TEST(DefaultLights, MetaLight)
{
    using LightGroupSkysphereCoOcto = LightGroupSkysphere<CoOctoCoordConverter>;
    using LightGroupSkysphereSpherical = LightGroupSkysphere<SphericalCoordConverter>;

    LightId lId;
    lId = LightId(0x1100ABC0);

    MRAY_LOG("{}", lId);
    MRAY_LOG("{}", HexKeyT(lId));

    GPUSystem system;
    // PGs
    EmptyPrimGroup emptyPrimGroup(0u, system);
    PrimGroupTriangle triangleGroup(1u, system);
    // TGs
    TransformGroupIdentity identityTG(0u, system);
    TransformGroupSingle singleTG(1u, system);
    // LGs
    LightGroupPrim<PrimGroupTriangle> triangleLightGroup(0u, system, triangleGroup);
    LightGroupSkysphereCoOcto skysphereCOLightGroup(0u, system, emptyPrimGroup);

    using MetaLightList = MetaLightArray
    <
        typename LightGroupSkysphereCoOcto:: template Light<TransformContextSingle>,
        typename LightGroupSkysphereCoOcto:: template Light<TransformContextIdentity>,
        typename LightGroupPrim<PrimGroupTriangle>:: template Light<TransformContextSingle>,
        typename LightGroupPrim<PrimGroupTriangle>:: template Light<TransformContextIdentity>
    >;


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