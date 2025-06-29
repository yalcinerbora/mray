#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>

#include "Tracer/PrimitiveDefaultTriangle.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

#include "Core/Quaternion.h"

namespace HelloTriangle
{

static constexpr std::array<Vector3ui, 1> indices =
{
    Vector3ui(0, 1, 2)
};

static constexpr std::array<Vector3, 3> positions =
{
    Vector3(0, 0.5, 0),
    Vector3(-0.5, -0.5, 0),
    Vector3(0.5, -0.5, 0)
};

static constexpr std::array<Vector3, 3> normals =
{
    Vector3(0, 0, 1),
    Vector3(0, 0, 1),
    Vector3(0, 0, 1)
};

using namespace TransformGen;
static const std::array<Quaternion, 3> normalsQuat =
{
    ToSpaceQuat(Vector3::XAxis(),
                Vector3::YAxis(),
                normals[0]),
    ToSpaceQuat(Vector3::XAxis(),
                Vector3::YAxis(),
                normals[1]),
    ToSpaceQuat(Vector3::XAxis(),
                Vector3::YAxis(),
                normals[2])
};

static constexpr std::array<Vector2, 3> uvs =
{
    Vector2(0.5, 1),
    Vector2(0, 0),
    Vector2(1, 0)
};

static constexpr uint32_t IndexCount = static_cast<uint32_t>(indices.size());
static constexpr uint32_t VertexCount = static_cast<uint32_t>(positions.size());

static_assert(indices.size() * 3 == positions.size());
static_assert(positions.size() == normals.size());
static_assert(normals.size() == uvs.size());

}

MRAY_KERNEL void KCAccessFirstTriangle(Span<uint8_t> dIsValid,
                                       Span<PrimitiveKey> primKeys,
                                       // Constants
                                       const typename PrimGroupTriangle::DataSoA soa,
                                       const std::array<Vector3ui, 1> indices,
                                       const std::array<Vector3, 3> positions,
                                       const std::array<Quaternion, 3> normalsQuat,
                                       const std::array<Vector2, 3> uvs)
{
    static constexpr uint32_t TRI_VERTEX_COUNT = Shape::Triangle::TRI_VERTEX_COUNT;
    assert(dIsValid.size() == primKeys.size());

    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId();
        i < static_cast<uint32_t>(primKeys.size());
        i += kp.TotalSize())
    {
        using Int = PrimitiveKey::Type;
        Int index = primKeys[i].FetchIndexPortion();

        bool isValid = true;
        // Index offsets are internally handled by the primitive
        Vector3ui indicesReg = soa.indexList[index];
        isValid &= (soa.indexList[index] == (indices[0] + i * TRI_VERTEX_COUNT));

        Vector3 position0 = soa.positions[indicesReg[0]];
        Vector3 position1 = soa.positions[indicesReg[1]];
        Vector3 position2 = soa.positions[indicesReg[2]];
        isValid &= (position0 == positions[0]);
        isValid &= (position1 == positions[1]);
        isValid &= (position2 == positions[2]);

        Quaternion tbn0 = soa.tbnRotations[indicesReg[0]];
        Quaternion tbn1 = soa.tbnRotations[indicesReg[1]];
        Quaternion tbn2 = soa.tbnRotations[indicesReg[2]];
        isValid &= (tbn0 == normalsQuat[0]);
        isValid &= (tbn1 == normalsQuat[1]);
        isValid &= (tbn2 == normalsQuat[2]);

        Vector2 uv0 = soa.uvs[indicesReg[0]];
        Vector2 uv1 = soa.uvs[indicesReg[1]];
        Vector2 uv2 = soa.uvs[indicesReg[2]];
        isValid &= (uv0 == uvs[0]);
        isValid &= (uv1 == uvs[1]);
        isValid &= (uv2 == uvs[2]);

        dIsValid[i] = isValid ? uint8_t(0) : uint8_t(1);
    }
}


TEST(DefaultTriangle, TypeCheck)
{
    GPUSystem s;

    PrimGroupTriangle triGroup(0, s);
    PrimAttributeInfoList list = triGroup.AttributeInfo();

    using namespace std::literals;
    EXPECT_EQ(std::get<PrimAttributeInfo::LOGIC_INDEX>(list[0]),
              PrimitiveAttributeLogic::POSITION);
    EXPECT_EQ(std::get<PrimAttributeInfo::LAYOUT_INDEX>(list[0]).Name(),
              MRayDataEnum::MR_VECTOR_3);
    EXPECT_EQ(std::get<PrimAttributeInfo::OPTIONALITY_INDEX>(list[0]),
              AttributeOptionality::MR_MANDATORY);
    EXPECT_EQ(std::get<PrimAttributeInfo::IS_ARRAY_INDEX>(list[0]),
              AttributeIsArray::IS_SCALAR);

    EXPECT_EQ(std::get<PrimAttributeInfo::LOGIC_INDEX>(list[1]),
              PrimitiveAttributeLogic::NORMAL);
    EXPECT_EQ(std::get<PrimAttributeInfo::LAYOUT_INDEX>(list[1]).Name(),
              MRayDataEnum::MR_QUATERNION);
    EXPECT_EQ(std::get<PrimAttributeInfo::OPTIONALITY_INDEX>(list[1]),
              AttributeOptionality::MR_MANDATORY);
    EXPECT_EQ(std::get<PrimAttributeInfo::IS_ARRAY_INDEX>(list[1]),
              AttributeIsArray::IS_SCALAR);

    EXPECT_EQ(std::get<PrimAttributeInfo::LOGIC_INDEX>(list[2]),
              PrimitiveAttributeLogic::UV0);
    EXPECT_EQ(std::get<PrimAttributeInfo::LAYOUT_INDEX>(list[2]).Name(),
              MRayDataEnum::MR_VECTOR_2);
    EXPECT_EQ(std::get<PrimAttributeInfo::OPTIONALITY_INDEX>(list[2]),
              AttributeOptionality::MR_MANDATORY);
    EXPECT_EQ(std::get<PrimAttributeInfo::IS_ARRAY_INDEX>(list[2]),
              AttributeIsArray::IS_SCALAR);

    EXPECT_EQ(std::get<PrimAttributeInfo::LOGIC_INDEX>(list[3]),
              PrimitiveAttributeLogic::INDEX);
    EXPECT_EQ(std::get<PrimAttributeInfo::LAYOUT_INDEX>(list[3]).Name(),
              MRayDataEnum::MR_VECTOR_3UI);
    EXPECT_EQ(std::get<PrimAttributeInfo::OPTIONALITY_INDEX>(list[3]),
              AttributeOptionality::MR_MANDATORY);
    EXPECT_EQ(std::get<PrimAttributeInfo::IS_ARRAY_INDEX>(list[3]),
              AttributeIsArray::IS_SCALAR);
}

TEST(DefaultTriangle, Load)
{
    GPUSystem system;
    using namespace HelloTriangle;

    std::vector<AttributeCountList> primCounts;
    primCounts.push_back({IndexCount, VertexCount});

    PrimGroupTriangle triGroup(0, system);
    std::vector<PrimBatchKey> batch = triGroup.Reserve(primCounts);
    EXPECT_EQ(batch.size(), 1);
    EXPECT_EQ(batch.front(), PrimBatchKey::CombinedKey(0, 0));

    triGroup.CommitReservations();
    PrimAttributeInfoList primIdList = triGroup.AttributeInfo();

    TransientData inputIndex(std::in_place_type<Vector3ui>, IndexCount);
    inputIndex.Push<Vector3ui>(indices);
    assert(inputIndex.IsFull());
    TransientData inputPos(std::in_place_type<Vector3>, VertexCount);
    inputPos.Push<Vector3>(positions);
    assert(inputPos.IsFull());
    TransientData inputNormal(std::in_place_type<Quaternion>, VertexCount);
    inputNormal.Push<Quaternion>(normalsQuat);
    assert(inputNormal.IsFull());
    TransientData inputUV(std::in_place_type<Vector2>, VertexCount);
    inputUV.Push<Vector2>(uvs);
    assert(inputUV.IsFull());

    // Fire and forget style load
    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);
    triGroup.PushAttribute(batch.front(), 3, std::move(inputIndex), queue);
    triGroup.PushAttribute(batch.front(), 0, std::move(inputPos), queue);
    triGroup.PushAttribute(batch.front(), 1, std::move(inputNormal), queue);
    triGroup.PushAttribute(batch.front(), 2, std::move(inputUV), queue);

    // TODO: This is not a good design? User have to sync everything
    // maybe return a ticket (barrier to the user)
    system.SyncAll();

    using DataSoA = typename PrimGroupTriangle::DataSoA;
    DataSoA triGroupSoA = triGroup.SoA();

    // Dont use bool here, it is packed
    Span<uint8_t> dIsValid;
    Span<PrimitiveKey> dPrimKeys;
    HostLocalMemory mem(system);
    MemAlloc::AllocateMultiData(std::tie(dIsValid, dPrimKeys), mem, {1, 1});

    // Eliminate false positives
    queue.MemsetAsync(dIsValid, 0x00);
    // dPrimKeys[0] must be zero, just memset here
    queue.MemsetAsync(dPrimKeys, 0x00);

    using namespace std::literals;
    queue.IssueWorkKernel<KCAccessFirstTriangle>
    (
        "GTest CheckDefaultTri"sv,
        DeviceWorkIssueParams{.workCount = 1},
        //
        dIsValid,
        dPrimKeys,
        triGroupSoA,
        indices,
        positions,
        normalsQuat,
        uvs
    );
    // Wait the data (we will directly access it)
    queue.Barrier().Wait();
    EXPECT_EQ(dIsValid[0], 1);
}

TEST(DefaultTriangle, LoadMulti)
{
    using namespace HelloTriangle;
    using IdInt = typename PrimBatchKey::Type;
    static constexpr size_t TRI_COUNT = 3;
    static constexpr uint32_t PRIM_GROUP_ID = 0;

    GPUSystem system;

    std::vector<AttributeCountList> primCounts(TRI_COUNT, {IndexCount, VertexCount});

    PrimGroupTriangle triGroup(PRIM_GROUP_ID, system);
    std::vector<PrimBatchKey> batches = triGroup.Reserve(primCounts);
    EXPECT_EQ(batches.size(), TRI_COUNT);

    IdInt counter = 0;
    for(const auto& batch : batches)
        EXPECT_EQ(batch, PrimBatchKey::CombinedKey(PRIM_GROUP_ID, counter++));

    triGroup.CommitReservations();
    PrimAttributeInfoList primIdList = triGroup.AttributeInfo();

    uint32_t queueIndex = 0;
    for(const auto& batch : batches)
    {
        // Convert input to transient data
        TransientData inputIndex(std::in_place_type<Vector3ui>, IndexCount);
        inputIndex.Push<Vector3ui>(indices);
        assert(inputIndex.IsFull());
        TransientData inputPos(std::in_place_type<Vector3>, VertexCount);
        inputPos.Push<Vector3>(positions);
        assert(inputPos.IsFull());
        TransientData inputNormal(std::in_place_type<Quaternion>, VertexCount);
        inputNormal.Push<Quaternion>(normalsQuat);
        assert(inputNormal.IsFull());
        TransientData inputUV(std::in_place_type<Vector2>, VertexCount);
        inputUV.Push<Vector2>(uvs);
        assert(inputUV.IsFull());

        // Fire and forget style load
        const GPUQueue& queue = system.BestDevice().GetComputeQueue(queueIndex);
        triGroup.PushAttribute(batch, 3, std::move(inputIndex), queue);
        triGroup.PushAttribute(batch, 0, std::move(inputPos), queue);
        triGroup.PushAttribute(batch, 1, std::move(inputNormal), queue);
        triGroup.PushAttribute(batch, 2, std::move(inputUV), queue);

        queueIndex++;
        queueIndex %= TotalQueuePerDevice();
    }
    // TODO: This is not a good design? User have to sync everything
    // maybe return a ticket (barrier to the user)
    system.SyncAll();
    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);
    // Finalize the indices
    triGroup.Finalize(queue);

    using DataSoA = typename PrimGroupTriangle::DataSoA;
    DataSoA triGroupSoA = triGroup.SoA();

    // Dont use bool here, it is packed
    Span<uint8_t> dIsValid;
    Span<PrimitiveKey> dPrimKeys;
    HostLocalMemory mem(system);
    MemAlloc::AllocateMultiData(std::tie(dIsValid, dPrimKeys), mem,
                                {TRI_COUNT, TRI_COUNT});

    // Eliminate false positives
    queue.MemsetAsync(dIsValid, 0x00);
    // Primitive ids, are linear in memory
    // MRay guarantees it
    counter = 0;
    std::vector<PrimitiveKey> hPrimKeys(TRI_COUNT);
    for(auto& key : hPrimKeys)
    {
        key = PrimitiveKey::CombinedKey(PRIM_GROUP_ID, counter++);
    }
    Span<const PrimitiveKey> hPrimKeySpan(hPrimKeys.cbegin(),
                                          hPrimKeys.cend());
    queue.MemcpyAsync(dPrimKeys, hPrimKeySpan);

    using namespace std::literals;
    queue.IssueWorkKernel<KCAccessFirstTriangle>
    (
        "GTest CheckDefaultTri"sv,
        DeviceWorkIssueParams{.workCount = TRI_COUNT},
        //
        dIsValid,
        dPrimKeys,
        triGroupSoA,
        indices,
        positions,
        normalsQuat,
        uvs
    );
    // Wait the data (we will directly access it)
    queue.Barrier().Wait();

    for(size_t i = 0; i < TRI_COUNT; i++)
        EXPECT_EQ(dIsValid[i], 1);
}