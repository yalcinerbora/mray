#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>

#include "Device/GPUSystem.h"

#include "Tracer/PrimitiveDefaultTriangle.h"
#include "Tracer/LightC.h"
#include "Tracer/LightsDefault.h"
#include "Tracer/MetaLight.h"
#include "Tracer/TextureView.hpp"

#include "Core/Log.h"

// Meta Light Test Types
using LightGroupSkysphereCoOcta = LightGroupSkysphere<CoOctaCoordConverter>;
using LightGroupSkysphereSpherical = LightGroupSkysphere<SphericalCoordConverter>;

using MetaLightList = MetaLightArrayT
<
    PackedTypes<LightGroupNull, TransformGroupIdentity>,
    PackedTypes<LightGroupSkysphereCoOcta, TransformGroupIdentity>,
    PackedTypes<LightGroupSkysphereCoOcta, TransformGroupSingle>,
    PackedTypes<LightGroupPrim<PrimGroupTriangle>, TransformGroupIdentity>,
    PackedTypes<LightGroupPrim<PrimGroupTriangle>, TransformGroupSingle>,
    PackedTypes<LightGroupPrim<PrimGroupSkinnedTriangle>, TransformGroupMulti>
>;

using MetaLight = typename MetaLightList::MetaLight;

using MetaLightArrayView = typename MetaLightList::View;

TransientData GenTriPos()
{
    TransientData data(std::in_place_type_t<Vector3>{}, 4);
    data.ReserveAll();
    auto span = data.AccessAs<Vector3>();

    span[0] = Vector3(0, 0, 0);
    span[1] = Vector3(1, 0, 0);
    span[2] = Vector3(0, 1, 0);
    span[3] = Vector3(1, 1, 0);
    return data;
}

TransientData GenTriNormal()
{
    TransientData data(std::in_place_type_t<Quaternion>{}, 4);
    data.ReserveAll();
    auto span = data.AccessAs<Quaternion>();

    span[0] = Quaternion::Identity();
    span[1] = Quaternion::Identity();
    span[2] = Quaternion::Identity();
    span[3] = Quaternion::Identity();
    return data;
}

TransientData GenTriUV()
{
    TransientData data(std::in_place_type_t<Vector2>{}, 4);
    data.ReserveAll();
    auto span = data.AccessAs<Vector2>();
    span[0] = Vector2(0, 0);
    span[1] = Vector2(0, 1);
    span[2] = Vector2(1, 0);
    span[3] = Vector2(1, 1);
    return data;
}

TransientData GenTriIndex()
{
    TransientData data(std::in_place_type_t<Vector3ui>{}, 2);
    data.ReserveAll();
    auto span = data.AccessAs<Vector3ui>();
    span[0] = Vector3ui(0, 1, 2);
    span[1] = Vector3ui(0, 2, 3);
    return data;
}

TransientData GenRadiance()
{
    TransientData data(std::in_place_type_t<Vector3>{}, 1);
    data.ReserveAll();
    auto span = data.AccessAs<Vector3>();
    span[0] = Vector3(10);
    return data;
}

TransientData GenMatrix()
{
    TransientData data(std::in_place_type_t<Matrix4x4>{}, 1);
    data.ReserveAll();
    auto span = data.AccessAs<Matrix4x4>();
    span[0] = Matrix4x4::Identity();
    return data;
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCReadLights(MRAY_GRID_CONSTANT const MetaLightArrayView lightArrayView,
                  MRAY_GRID_CONSTANT const Span<const RandomNumber> dRandomNumbers)
{
    Vector3 DistantPoint = Vector3(2, 3, 4);
    MetaHit Hit = MetaHit(Vector2(0.5, 0.3));
    Vector3 Dir = Vector3(2, 3, 4).Normalize();
    Ray TestRay = Ray(Dir, DistantPoint);

    using MLVIdentity = typename MetaLightArrayView::template MetaLightView<SpectrumConverterIdentity>;

    SpectrumConverterIdentity identitySc = {};

    KernelCallParams kp;
    uint32_t lightCount = lightArrayView.Size();
    for(uint32_t i = kp.GlobalId(); i < lightCount; i += kp.TotalSize())
    {
        RNGDispenser rng(dRandomNumbers, i, lightCount);

        MLVIdentity light = lightArrayView(identitySc, i);

        SampleT<Vector3> sampleVec = light.SampleSolidAngle(rng, DistantPoint);
        Float pdfSolidAngle = light.PdfSolidAngle(Hit, DistantPoint, Dir);
        uint32_t solidRNCount = light.SampleSolidAngleRNCount();
        //
        SampleT<Ray> sampleRay = light.SampleRay(rng);
        Float pdfRay = light.PdfRay(TestRay);
        uint32_t rayRNCount = light.SampleRayRNCount();
        //
        RayCone rayCone = RayCone{.aperture = 0, .width = 0};
        Spectrum radiance0 = light.EmitViaHit(Dir, Hit, rayCone);
        Spectrum radiance1 = light.EmitViaSurfacePoint(Dir, DistantPoint, rayCone);
        bool isPrimBacked = light.IsPrimitiveBackedLight();

        #ifdef MRAY_GPU_BACKEND_CPU
            MRAY_LOG("[{}]-----\n"
                     "    SolidAngle - ({}, {}, {}| {}), ({}), ({})\n"
                     "    Ray        - ([{}, {}, {}] [{}, {}, {}]| {}), ({}), ({}) - Ray\n"
                     "    Emit       - ({}, {}, {}), ({}, {}, {})\n"
                     "    PrimBacked - ({})\n"
                     "--------\n",
                     i,
                     //
                     sampleVec.value[0], sampleVec.value[1], sampleVec.value[2],
                     sampleVec.pdf, pdfSolidAngle, solidRNCount,
                     //
                     sampleRay.value.Dir()[0], sampleRay.value.Dir()[1], sampleRay.value.Dir()[2],
                     sampleRay.value.Pos()[0], sampleRay.value.Pos()[1], sampleRay.value.Pos()[2],
                     sampleRay.pdf, pdfRay, rayRNCount,
                     //
                     radiance0[0], radiance0[1], radiance0[2],
                     radiance1[0], radiance1[1], radiance1[2],
                     //
                     (isPrimBacked) ? "True" : "False");
        #else
            printf("[%u]-----\n"
                   "    SolidAngle - (%f, %f, %f| %f), (%f), (%u)\n"
                   "    Ray        - ([%f, %f, %f] [%f, %f, %f]| %f), (%f), (%u) - Ray\n"
                   "    Emit       - (%f, %f, %f), (%f, %f, %f)\n"
                   "    PrimBacked - (%s)\n"
                   "--------\n",
                   i,
                   //
                   sampleVec.value[0], sampleVec.value[1], sampleVec.value[2],
                   sampleVec.pdf, pdfSolidAngle, solidRNCount,
                   //
                   sampleRay.value.Dir()[0], sampleRay.value.Dir()[1], sampleRay.value.Dir()[2],
                   sampleRay.value.Pos()[0], sampleRay.value.Pos()[1], sampleRay.value.Pos()[2],
                   sampleRay.pdf, pdfRay, rayRNCount,
                   //
                   radiance0[0], radiance0[1], radiance0[2],
                   radiance1[0], radiance1[1], radiance1[2],
                   //
                   (isPrimBacked) ? "True" : "False");
        #endif
    }
}

TEST(DefaultLights, DISABLED_Skysphere)
{
}

TEST(DefaultLights, MetaLight)
{
    GPUSystem system;
    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);
    // Generate some groups in maps
    // PGs
    Map<PrimGroupId, PrimGroupPtr> primGroups;
    primGroups.emplace(PrimGroupId(0),
                       PrimGroupPtr(std::make_unique<PrimGroupEmpty>(0u, system)));
    primGroups.emplace(PrimGroupId(1),
                       PrimGroupPtr(std::make_unique<PrimGroupTriangle>(1u, system)));
    auto& emptyPrimGroup = *primGroups.at(PrimGroupId(0)).value().get().get();
    auto& triangleGroup = *primGroups.at(PrimGroupId(1)).value().get().get();
    // TGs
    Map<TransGroupId, TransformGroupPtr> transformGroups;
    transformGroups.emplace(TransGroupId(0),
                            TransformGroupPtr(std::make_unique<TransformGroupIdentity>(0u, system)));
    transformGroups.emplace(TransGroupId(1),
                            TransformGroupPtr(std::make_unique<TransformGroupSingle >(1u, system)));
    auto& identityTG = *transformGroups.at(TransGroupId(0)).value().get().get();
    auto& singleTG = *transformGroups.at(TransGroupId(1)).value().get().get();
    // LGs
    Map<LightGroupId, LightGroupPtr> lightGroups;
    lightGroups.emplace
    (
        LightGroupId(0),
        LightGroupPtr
        (
            std::make_unique<LightGroupPrim<PrimGroupTriangle>>
            (0u, system, TextureViewMap{}, TextureMap{}, triangleGroup)
        )
    );
    lightGroups.emplace
    (
        LightGroupId(1),
        LightGroupPtr
        (
            std::make_unique<LightGroupSkysphereCoOcta>
            (1u, system, TextureViewMap{}, TextureMap{}, emptyPrimGroup)
        )
    );
    auto& triangleLightGroup = *lightGroups.at(LightGroupId(0)).value().get().get();
    auto& skysphereCOLightGroup = *lightGroups.at(LightGroupId(1)).value().get().get();
    //
    //auto hEmptyPrimBatchIds = emptyPrimGroup.Reserve({{1, 1}});
    auto hTriBatchIds = triangleGroup.Reserve({{2, 4}});
    auto hIdentityTransIds = identityTG.Reserve({{1}});
    auto hSingleTransIds = singleTG.Reserve({{1}});
    auto hTriLightIds = triangleLightGroup.Reserve({{1}}, hTriBatchIds);
    auto hSkysphereLightIds = skysphereCOLightGroup.Reserve({{1}}, {});
    //
    emptyPrimGroup.CommitReservations();
    triangleGroup.CommitReservations();
    identityTG.CommitReservations();
    singleTG.CommitReservations();
    triangleLightGroup.CommitReservations();
    skysphereCOLightGroup.CommitReservations();
    // Add stuff
    triangleGroup.PushAttribute(hTriBatchIds[0], 0, GenTriPos(), queue);
    triangleGroup.PushAttribute(hTriBatchIds[0], 1, GenTriNormal(), queue);
    triangleGroup.PushAttribute(hTriBatchIds[0], 2, GenTriUV(), queue);
    triangleGroup.PushAttribute(hTriBatchIds[0], 3, GenTriIndex(), queue);
    //
    singleTG.PushAttribute(hIdentityTransIds[0], 0, GenMatrix(), queue);
    //
    triangleLightGroup.PushTexAttribute(hTriLightIds[0], hTriLightIds[0], 0,
                                        GenRadiance(), std::vector<Optional<TextureId>>(1),
                                        queue);
    skysphereCOLightGroup.PushTexAttribute(hTriLightIds[0], hTriLightIds[0], 0,
                                           GenRadiance(), std::vector<Optional<TextureId>>(1),
                                           queue);
    // Finalize
    emptyPrimGroup.Finalize(queue);
    triangleGroup.Finalize(queue);
    identityTG.Finalize(queue);
    singleTG.Finalize(queue);
    triangleGroup.Finalize(queue);
    skysphereCOLightGroup.Finalize(queue);

    //
    using LSPair = Pair<LightSurfaceId, LightSurfaceParams>;
    std::vector<LSPair> lSurfs =
    {
        {
            LightSurfaceId(0),
            LightSurfaceParams
            {
                .lightId = std::bit_cast<LightId>(hTriLightIds[0]),
                .transformId = std::bit_cast<TransformId>(hIdentityTransIds[0])
            },
        },
        {
            LightSurfaceId(1),
            LightSurfaceParams
            {
                .lightId = std::bit_cast<LightId>(hTriLightIds[0]),
                .transformId = std::bit_cast<TransformId>(hSingleTransIds[0])
            }
        },
        {
            LightSurfaceId(2),
            LightSurfaceParams
            {
                .lightId = std::bit_cast<LightId>(hSkysphereLightIds[0]),
                .transformId = std::bit_cast<TransformId>(hIdentityTransIds[0])
            }
        }
    };
    LightSurfaceParams boundarySurface =
    {
        .lightId = std::bit_cast<LightId>(hSkysphereLightIds[0]),
        .transformId = std::bit_cast<TransformId>(hSingleTransIds[0])
    };
    std::sort(lSurfs.begin(), lSurfs.end(), LightSurfaceLessThan);

    MetaLightListConstructionParams params
    {
        .lightGroups = lightGroups,
        .transformGroups = transformGroups,
        .lSurfList = Span<const LSPair>(lSurfs)
    };
    MetaLightList lightList = MetaLightList(system);
    lightList.Construct(params, boundarySurface, queue);

    Span<RandomNumber> dRandomNumbers;
    DeviceLocalMemory rnMem(*queue.Device());
    MemAlloc::AllocateMultiData(std::tie(dRandomNumbers), rnMem,
                                {1024});
    //
    MetaLightArrayView lightView = lightList.Array();
    queue.IssueWorkKernel<KCReadLights>
    (
        "KCReadLights",
        DeviceWorkIssueParams{.workCount = lightView.Size()},
        lightView,
        dRandomNumbers
    );
    queue.Barrier().Wait();
}

TEST(DefaultLights, PrimLight_Triangle)
{
    using Triangle  = typename PrimGroupTriangle::Primitive<>;
    using Light     = typename LightGroupPrim<PrimGroupTriangle>::template Light<>;
    using LightData = typename LightGroupPrim<PrimGroupTriangle>::DataSoA;
    using PrimData  = typename PrimGroupTriangle::DataSoA;

    TransformContextIdentity tContext;
    SpectrumConverterIdentity sContext;

    std::vector<Vector3> hPos =
    {
        Vector3(-0.5, -0.5, -2),
        Vector3( 0.5, -0.5, -2),
        Vector3( 0.5,  0.5, -2),
        Vector3(-0.5,  0.5, -2)
    };
    std::vector<Vector3ui> hIndex =
    {
        Vector3ui(0, 1, 2),
        Vector3ui(0, 2, 3)
    };
    PrimData pData =
    {
        .positions = Span<const Vector3>(hPos),
        .tbnRotations = Span<const Quaternion>(),
        .uvs = Span<const Vector2>(),
        .indexList = Span<const Vector3ui>(hIndex)
    };

    std::vector<uint32_t> hIsTwoSidedFlags = {0};
    std::vector<ParamVaryingData<2, Vector3>> hRadiances = {ParamVaryingData<2, Vector3>(Vector3(1))};
    LightData lData =
    {
        .dRadiances = Span<const ParamVaryingData<2, Vector3>>(hRadiances),
        .dIsTwoSidedFlags = Bitspan<const uint32_t>(Span<const uint32_t>(hIsTwoSidedFlags))
    };

    Triangle tri0(tContext, pData, PrimitiveKey::CombinedKey(0, 0));
    Triangle tri1(tContext, pData, PrimitiveKey::CombinedKey(0, 1));
    Light l0(sContext, tri0, lData, LightKey::CombinedKey(0, 0));
    Light l1(sContext, tri1, lData, LightKey::CombinedKey(0, 0));


    static constexpr uint32_t SAMPLE_COUNT = 4096;
    std::vector<RandomNumber> hRandomNumbers(SAMPLE_COUNT * 2 * 2);
    std::mt19937 rng;
    for(RandomNumber& rn : hRandomNumbers)
        rn = RandomNumber(rng());

    for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
    {
        RNGDispenser rng0(Span<const RandomNumber>(hRandomNumbers), i, 2);
        RNGDispenser rng1(Span<const RandomNumber>(hRandomNumbers), i + 1, 2);
        auto sample0 = l0.SampleSolidAngle(rng0, Vector3::Zero());
        auto sample1 = l1.SampleSolidAngle(rng1, Vector3::Zero());

        Ray r0 = Ray(sample0.value.Normalize(), Vector3::Zero());
        Ray r1 = Ray(sample1.value.Normalize(), Vector3::Zero());

        Float pdf0 = l0.PdfSolidAngle(tri0.Intersects(r0, false).value().hit,
                                      r0.Pos(), r0.Dir());
        Float pdf1 = l1.PdfSolidAngle(tri1.Intersects(r1, false).value().hit,
                                      r1.Pos(), r1.Dir());

        EXPECT_FLOAT_EQ(pdf0, sample0.pdf);
        EXPECT_FLOAT_EQ(pdf1, sample1.pdf);
    }

}