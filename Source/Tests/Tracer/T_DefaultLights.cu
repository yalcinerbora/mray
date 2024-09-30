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

using MetaLightList = MetaLightArrayT
<
    Tuple<LightGroupNull, TransformGroupIdentity>,
    Tuple<LightGroupSkysphereCoOcta, TransformGroupIdentity>,
    Tuple<LightGroupSkysphereCoOcta, TransformGroupSingle>,
    Tuple<LightGroupPrim<PrimGroupTriangle>, TransformGroupIdentity>,
    Tuple<LightGroupPrim<PrimGroupTriangle>, TransformGroupSingle>,
    Tuple<LightGroupPrim<PrimGroupSkinnedTriangle>, TransformGroupMulti>
>;

using MetaLight = typename MetaLightList::MetaLight;

template<class SpectrumConverter>
using MetaLightView = typename MetaLightList::MetaLightView< SpectrumConverter>;

TEST(DefaultLights, DISABLED_Skysphere)
{
}

TEST(DefaultLights, DISABLED_MetaLight)
{
    GPUSystem system;
    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);
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
                       Vector2ui(0, 1), queue);
    lightList.AddBatch(triangleLightGroup, singleTG,
                       primKeys, lightKeys, transformKeys,
                       Vector2ui(0, 1), queue);

    lightList.AddBatch(skysphereCOLightGroup, identityTG,
                       primKeys, lightKeys, transformKeys,
                       Vector2ui(0, 1), queue);

    lightList.AddBatch(skysphereCOLightGroup, singleTG,
                       primKeys, lightKeys, transformKeys,
                       Vector2ui(0, 1), queue);
}