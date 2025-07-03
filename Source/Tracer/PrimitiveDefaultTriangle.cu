#include "PrimitiveDefaultTriangle.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUMemory.h"
#include "Device/GPUAlgGeneric.h"

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCAdjustIndices(// I-O
                     MRAY_GRID_CONSTANT const Span<Vector3ui> dIndicesInOut,
                     // Input
                     MRAY_GRID_CONSTANT const Span<const Vector4ui> dVertexIndexRanges)
{
    KernelCallParams kp;
    uint32_t totalRanges = static_cast<uint32_t>(dVertexIndexRanges.size());

    // TODO: Check if this is undefined behaviour,
    // We technically do an out of bounds access over the "std::array<uint32_t, 3>"
    // Sanity check, if there is a padding (it should not but just to be sure)
    static_assert(sizeof(std::array<Vector3ui, 2>) == sizeof(Vector3ui) * 2);
    Span<uint32_t> dVertexIndices = Span<uint32_t>(dIndicesInOut.data()->AsArray().data(),
                                                   dIndicesInOut.size() * Vector3ui::Dims);

    // Block-stride Loop
    for(uint32_t blockId = kp.blockId; blockId < totalRanges;
        blockId += kp.gridSize)
    {
        using namespace TracerConstants;
        uint32_t localTid = kp.threadId;

        MRAY_SHARED_MEMORY Vector4ui sVertexIndexRange;
        // CPU Does not have proper shared memory
        // and parallel loads, so let t0 to load it to
        // thread_local memory at start
        #ifdef MRAY_GPU_BACKEND_CPU
            if(localTid == 0)
                sVertexIndexRange = dVertexIndexRanges[blockId];
        #else
            assert(kp.blockSize >= Vector4ui::Dims);
            if(localTid < Vector4ui::Dims)
                sVertexIndexRange[localTid] = dVertexIndexRanges[blockId][localTid];
        #endif
        BlockSynchronize();

        Vector2ui indexRange = Vector2ui(sVertexIndexRange[2],
                                         sVertexIndexRange[3]);
        uint32_t primIndexCount = indexRange[1] - indexRange[0];
        uint32_t vICount = primIndexCount * Vector3ui::Dims;
        uint32_t vIStart = indexRange[0] * Vector3ui::Dims;
        uint32_t vStart = sVertexIndexRange[0];

        // Hand unroll some here (slightly improved performance)
        auto& dVI = dVertexIndices;
        for(uint32_t j = localTid; j < vICount;)
        {
                            dVI[vIStart + j] += vStart; j += kp.blockSize;
            if(j < vICount) dVI[vIStart + j] += vStart; j += kp.blockSize;
            if(j < vICount) dVI[vIStart + j] += vStart; j += kp.blockSize;
            if(j < vICount) dVI[vIStart + j] += vStart; j += kp.blockSize;
        }
        // Before writing new shared memory values wait all threads to end
        BlockSynchronize();
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCApplyTransformsTriangle(// I-O
                               MRAY_GRID_CONSTANT const Span<Vector3> dPositionsInOut,
                               MRAY_GRID_CONSTANT const Span<Quaternion> dNormalsInOut,
                               // Input
                               MRAY_GRID_CONSTANT const Span<const Matrix4x4> dBatchTransforms,
                               MRAY_GRID_CONSTANT const Span<const Matrix4x4> dBatchInvTransforms,
                               MRAY_GRID_CONSTANT const Span<const Vector2ul> dVertexRanges,
                               MRAY_GRID_CONSTANT const uint32_t blockPerBatch)
{
    KernelCallParams kp;
    uint32_t totalBatches = static_cast<uint32_t>(dVertexRanges.size());
    uint32_t blockCount = totalBatches * blockPerBatch;

    // Block-stride Loop
    for(uint32_t blockId = kp.blockId; blockId < blockCount; blockId += kp.gridSize)
    {
        uint32_t batchI = blockId / blockPerBatch;
        uint32_t localBatchI = blockId % blockPerBatch;

        MRAY_SHARED_MEMORY Matrix4x4 sBatchTransform;
        MRAY_SHARED_MEMORY Matrix4x4 sBatchInvTransform;
        MRAY_SHARED_MEMORY Vector2ul sVertexRanges;
        #ifdef MRAY_GPU_BACKEND_CPU
            if(kp.threadId == 0)
            {
                sBatchTransform = dBatchTransforms[batchI];
                sBatchInvTransform = dBatchInvTransforms[batchI];
                sVertexRanges = dVertexRanges[batchI];
            }
        #else
            assert(kp.blockSize >= 34);
            // Load matrices / ranges
            if(kp.threadId < 16)
            {
                uint32_t i = kp.threadId;
                sBatchTransform[i] = dBatchTransforms[batchI][i];

            }
            else if(kp.threadId >= 16 && kp.threadId < 32)
            {
                uint32_t i = kp.threadId - 16;
                sBatchInvTransform[i] = dBatchInvTransforms[batchI][i];
            }
            else if(kp.threadId >= 32 && kp.threadId < 34)
            {
                uint32_t i = kp.threadId - 32;
                sVertexRanges[i] = dVertexRanges[batchI][i];
            }
        #endif
        BlockSynchronize();

        // Loop over each vertex for this tex
        uint32_t vertexCount = uint32_t(sVertexRanges[1] - sVertexRanges[0]);
        uint32_t vertexStart = localBatchI * kp.blockSize + kp.threadId;
        uint32_t vertexIncrement = blockPerBatch * kp.blockSize;
        for(uint32_t i = vertexStart; i < vertexCount; i += vertexIncrement)
        {
            // Convert the position (easy)
            uint64_t vI = sVertexRanges[0] + i;
            Vector4 pos = Vector4(dPositionsInOut[vI], 1);
            dPositionsInOut[vI] = Vector3(sBatchTransform * pos);
            // Convert the normal (hard)
            // Create the tbn vectors from quaternion
            // multiply with the matrices
            // then convert back to the tbn quaternion
            Quaternion tbn = dNormalsInOut[vI];
            Vector3 t = tbn.ApplyInvRotation(Vector3::XAxis());
            Vector3 b = tbn.ApplyInvRotation(Vector3::YAxis());
            Vector3 n = tbn.ApplyInvRotation(Vector3::ZAxis());
            // We need to multiply these with normal matrix
            t = Vector3(sBatchInvTransform.LeftMultiply(Vector4(t, 0))).Normalize();
            b = Vector3(sBatchInvTransform.LeftMultiply(Vector4(b, 0))).Normalize();
            n = Vector3(sBatchInvTransform.LeftMultiply(Vector4(n, 0))).Normalize();
            auto[tN, bN] = Graphics::GSOrthonormalize(t, b, n);
            //
            Quaternion tbnOut = TransformGen::ToSpaceQuat(tN, bN, n);
            dNormalsInOut[vI] = tbnOut;
        }
        // Before writing new shared memory values wait all threads to end
        BlockSynchronize();
    }
}

PrimGroupTriangle::PrimGroupTriangle(uint32_t primGroupId,
                                     const GPUSystem& sys)
    : GenericGroupPrimitive(primGroupId, sys,
                            DefaultTriangleDetail::DeviceMemAllocationGranularity,
                            DefaultTriangleDetail::DeviceMemReservationSize)
{}

void PrimGroupTriangle::CommitReservations()
{
    GenericCommit(std::tie(dPositions, dTBNRotations, dUVs, dIndexList),
                  {1, 1, 1, 0});

    soa.positions = ToConstSpan(dPositions);
    soa.tbnRotations = ToConstSpan(dTBNRotations);
    soa.uvs = ToConstSpan(dUVs);
    soa.indexList = ToConstSpan(dIndexList);
}

PrimAttributeInfoList PrimGroupTriangle::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum PrimitiveAttributeLogic;
    using enum AttributeOptionality;
    using enum AttributeIsArray;
    // Here we mark them as "IS_SCALAR", because primitive group is group of primitives
    // and not primitive batches
    static const PrimAttributeInfoList LogicList =
    {
        PrimAttributeInfo(POSITION, MRayDataType<MR_VECTOR_3>(),    IS_SCALAR, MR_MANDATORY),
        PrimAttributeInfo(NORMAL,   MRayDataType<MR_QUATERNION>(),  IS_SCALAR, MR_MANDATORY),
        PrimAttributeInfo(UV0,      MRayDataType<MR_VECTOR_2>(),    IS_SCALAR, MR_MANDATORY),
        PrimAttributeInfo(INDEX,    MRayDataType<MR_VECTOR_3UI>(),  IS_SCALAR, MR_MANDATORY)
    };
    return LogicList;
}

void PrimGroupTriangle::PushAttribute(PrimBatchKey batchKey,
                                      uint32_t attributeIndex,
                                      TransientData data,
                                      const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>& d)
    {
        GenericPushData(d, batchKey.FetchIndexPortion(),
                        attributeIndex,
                        std::move(data), queue);
    };

    switch(attributeIndex)
    {
        case POSITION_ATTRIB_INDEX: PushData(dPositions);     break;
        case NORMAL_ATTRIB_INDEX:   PushData(dTBNRotations);  break;
        case UV_ATTRIB_INDEX:       PushData(dUVs);           break;
        case INDICES_ATTRIB_INDEX:  PushData(dIndexList);     break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void PrimGroupTriangle::PushAttribute(PrimBatchKey batchKey,
                                      uint32_t attributeIndex,
                                      const Vector2ui& subRange,
                                      TransientData data,
                                      const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>& d)
    {
        GenericPushData(d, batchKey.FetchIndexPortion(),
                        attributeIndex, subRange,
                        std::move(data), queue);
    };

    switch(attributeIndex)
    {
        case POSITION_ATTRIB_INDEX: PushData(dPositions);     break;
        case NORMAL_ATTRIB_INDEX:   PushData(dTBNRotations);  break;
        case UV_ATTRIB_INDEX:       PushData(dUVs);           break;
        case INDICES_ATTRIB_INDEX:  PushData(dIndexList);     break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void PrimGroupTriangle::PushAttribute(PrimBatchKey idStart, PrimBatchKey idEnd,
                                      uint32_t attributeIndex,
                                      TransientData data,
                                      const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>& d)
    {
        Vector<2, IdInt> idRange(idStart.FetchIndexPortion(),
                                 idEnd.FetchIndexPortion());
        GenericPushData(d, idRange, attributeIndex,
                        std::move(data), queue);
    };

    switch(attributeIndex)
    {
        case POSITION_ATTRIB_INDEX: PushData(dPositions);     break;
        case NORMAL_ATTRIB_INDEX:   PushData(dTBNRotations);  break;
        case UV_ATTRIB_INDEX:       PushData(dUVs);           break;
        case INDICES_ATTRIB_INDEX:  PushData(dIndexList);     break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void PrimGroupTriangle::Finalize(const GPUQueue& queue)
{
    size_t batchCount = this->itemRanges.size();
    std::vector<Vector4ui> hVertexIndexRanges;
    hVertexIndexRanges.reserve(batchCount);

    DeviceLocalMemory tempMem(*queue.Device());
    Span<Vector4ui> dVertexIndexRanges;
    MemAlloc::AllocateMultiData(std::tie(dVertexIndexRanges),
                                tempMem,
                                {batchCount});



    for(const auto& kv : this->itemRanges)
    {
        const AttributeRanges& ranges = kv.second;
        hVertexIndexRanges.emplace_back(Vector4ui(ranges[POSITION_ATTRIB_INDEX][0],
                                                  ranges[POSITION_ATTRIB_INDEX][1],
                                                  ranges[INDICES_ATTRIB_INDEX][0],
                                                  ranges[INDICES_ATTRIB_INDEX][1]));
    }
    queue.MemcpyAsync(dVertexIndexRanges,
                      Span<const Vector4ui>(hVertexIndexRanges.cbegin(),
                                            hVertexIndexRanges.end()));

    uint32_t blockCount = queue.RecommendedBlockCountDevice
    (
        reinterpret_cast<const void*>(&KCAdjustIndices),
        StaticThreadPerBlock1D(),
        0
    );
    using namespace std::string_view_literals;
    queue.IssueBlockKernel<KCAdjustIndices>
    (
        "KCAdjustIndices"sv,
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = StaticThreadPerBlock1D()
        },
        // I-O
        dIndexList,
        dVertexIndexRanges
    );
    queue.Barrier().Wait();
}

void PrimGroupTriangle::ApplyTransformations(const std::vector<PrimBatchKey>& primBatches,
                                             const std::vector<Matrix4x4>& batchTransformations,
                                             const GPUQueue& queue)
{
    if(primBatches.empty()) return;

    for(PrimBatchKey bk : primBatches)
    {
        if(bk.FetchBatchPortion() != this->groupId)
            throw MRayError("{:s}:{:d}: While doing \"ApplyTransformations\", "
                            "there are PrimBatchIds which does not belong this group",
                            TypeName(), this->groupId);
    }

    assert(primBatches.size() == batchTransformations.size());
    size_t batchCount = primBatches.size();

    Span<Vector2ul> dVertexRanges;
    Span<Matrix4x4> dTransformations;
    Span<Matrix4x4> dInvTransformations;
    DeviceLocalMemory tempMem(*queue.Device());
    MemAlloc::AllocateMultiData(std::tie(dVertexRanges,
                                         dTransformations,
                                         dInvTransformations),
                                tempMem,
                                {batchCount, batchCount, batchCount});

    // Issue transformation inverting before finding batch ranges
    queue.MemcpyAsync(dTransformations,
                      ToConstSpan(Span(batchTransformations.data(), batchCount)));
    DeviceAlgorithms::Transform(dInvTransformations, ToConstSpan(dTransformations),
                                queue, KCInvertTransforms());

    // While GPU inverts transformation create batch range buffer
    std::vector<Vector2ul> hVertexRanges;
    hVertexRanges.reserve(batchCount);
    for(PrimBatchKey batchKey : primBatches)
    {
        auto rangeOpt = this->itemRanges.at(batchKey.FetchIndexPortion());
        assert(rangeOpt.has_value());
        const AttributeRanges& ranges = rangeOpt.value().get();
        assert(ranges[POSITION_ATTRIB_INDEX] == ranges[NORMAL_ATTRIB_INDEX]);
        hVertexRanges.push_back(ranges[POSITION_ATTRIB_INDEX]);
    }
    queue.MemcpyAsync(dVertexRanges,
                      Span<const Vector2ul>(hVertexRanges.data(), batchCount));

    uint32_t blockCount = queue.RecommendedBlockCountDevice
    (
        reinterpret_cast<const void*>(&KCApplyTransformsTriangle),
        StaticThreadPerBlock1D(),
        0
    );
    using namespace std::string_view_literals;
    static constexpr uint32_t BLOCK_PER_BATCH = 128;
    queue.IssueBlockKernel<KCApplyTransformsTriangle>
    (
        "KCApplyTransformationsTriangle"sv,
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = StaticThreadPerBlock1D()
        },
        //
        dPositions,
        dTBNRotations,
        ToConstSpan(dTransformations),
        ToConstSpan(dInvTransformations),
        ToConstSpan(dVertexRanges),
        BLOCK_PER_BATCH
    );
    // Wait for completion, before scope exit
    // due to temporary memory allocation
    queue.Barrier().Wait();
}

Vector2ui PrimGroupTriangle::BatchRange(PrimBatchKey key) const
{
    auto range = FindRange(static_cast<CommonKey>(key))[INDICES_ATTRIB_INDEX];
    return Vector2ui(range);
}

size_t PrimGroupTriangle::TotalPrimCount() const
{
    return this->TotalPrimCountImpl(0);
}

typename PrimGroupTriangle::DataSoA PrimGroupTriangle::SoA() const
{
    return soa;
}

Span<const Vector3ui> PrimGroupTriangle::GetIndexSpan() const
{
    return ToConstSpan(dIndexList);
}

Span<const Vector3> PrimGroupTriangle::GetVertexPositionSpan() const
{
    return ToConstSpan(dPositions);
}

PrimGroupSkinnedTriangle::PrimGroupSkinnedTriangle(uint32_t primGroupId,
                                                   const GPUSystem& sys)
    : GenericGroupPrimitive(primGroupId, sys,
                            DefaultTriangleDetail::DeviceMemAllocationGranularity,
                            DefaultTriangleDetail::DeviceMemReservationSize)
{}

void PrimGroupSkinnedTriangle::CommitReservations()
{
    GenericCommit(std::tie(dPositions, dTBNRotations, dUVs,
                           dIndexList, dSkinWeights, dSkinIndices),
                  {1, 1, 1, 1, 1, 0});

    soa.positions = ToConstSpan(dPositions);
    soa.tbnRotations = ToConstSpan(dTBNRotations);
    soa.uvs = ToConstSpan(dUVs);
    soa.indexList = ToConstSpan(dIndexList);
    soa.skinWeights = ToConstSpan(dSkinWeights);
    soa.skinIndices = ToConstSpan(dSkinIndices);
}

PrimAttributeInfoList PrimGroupSkinnedTriangle::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum PrimitiveAttributeLogic;
    using enum AttributeOptionality;
    using enum AttributeIsArray;
    static const PrimAttributeInfoList LogicList =
    {
        PrimAttributeInfo(POSITION,     MRayDataType<MR_VECTOR_3>(),    IS_SCALAR, MR_MANDATORY),
        PrimAttributeInfo(NORMAL,       MRayDataType<MR_QUATERNION>(),  IS_SCALAR, MR_MANDATORY),
        PrimAttributeInfo(UV0,          MRayDataType<MR_VECTOR_2>(),    IS_SCALAR, MR_MANDATORY),
        PrimAttributeInfo(WEIGHT,       MRayDataType<MR_UNORM_4x8>(),   IS_SCALAR, MR_MANDATORY),
        PrimAttributeInfo(WEIGHT_INDEX, MRayDataType<MR_VECTOR_4UC>(),  IS_SCALAR, MR_MANDATORY),
        PrimAttributeInfo(INDEX,        MRayDataType<MR_VECTOR_3UI>(),  IS_SCALAR, MR_MANDATORY)
    };
    return LogicList;
}

void PrimGroupSkinnedTriangle::PushAttribute(PrimBatchKey batchKey, uint32_t attributeIndex,
                                             TransientData data, const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>& d)
    {
        GenericPushData(d, batchKey.FetchIndexPortion(),
                        attributeIndex,
                        std::move(data), queue);
    };

    switch(attributeIndex)
    {
        case POSITION_ATTRIB_INDEX: PushData(dPositions);     break;
        case NORMAL_ATTRIB_INDEX:   PushData(dTBNRotations);  break;
        case UV_ATTRIB_INDEX:       PushData(dUVs);           break;
        case SKIN_W_ATTRIB_INDEX:   PushData(dSkinWeights);   break;
        case SKIN_I_ATTRIB_INDEX:   PushData(dSkinIndices);   break;
        case INDICES_ATTRIB_INDEX:  PushData(dIndexList);     break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void PrimGroupSkinnedTriangle::PushAttribute(PrimBatchKey batchKey,
                                             uint32_t attributeIndex,
                                             const Vector2ui& subRange,
                                             TransientData data,
                                             const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>& d)
    {
        GenericPushData(d, batchKey.FetchIndexPortion(), attributeIndex,
                        subRange, std::move(data), queue);
    };

    switch(attributeIndex)
    {
        case POSITION_ATTRIB_INDEX: PushData(dPositions);     break;
        case NORMAL_ATTRIB_INDEX:   PushData(dTBNRotations);  break;
        case UV_ATTRIB_INDEX:       PushData(dUVs);           break;
        case SKIN_W_ATTRIB_INDEX:   PushData(dSkinWeights);   break;
        case SKIN_I_ATTRIB_INDEX:   PushData(dSkinIndices);   break;
        case INDICES_ATTRIB_INDEX:  PushData(dIndexList);     break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void PrimGroupSkinnedTriangle::PushAttribute(PrimBatchKey idStart, PrimBatchKey idEnd,
                                             uint32_t attributeIndex,
                                             TransientData data,
                                             const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>& d)
    {
        Vector<2, IdInt> idRange(idStart.FetchIndexPortion(),
                                 idEnd.FetchIndexPortion());
        GenericPushData(d, idRange, attributeIndex,
                        std::move(data), queue);
    };

    switch(attributeIndex)
    {
        case POSITION_ATTRIB_INDEX: PushData(dPositions);       break;
        case NORMAL_ATTRIB_INDEX:   PushData(dTBNRotations);    break;
        case UV_ATTRIB_INDEX:       PushData(dUVs);             break;
        case SKIN_W_ATTRIB_INDEX:   PushData(dSkinWeights);     break;
        case SKIN_I_ATTRIB_INDEX:   PushData(dSkinIndices);     break;
        case INDICES_ATTRIB_INDEX:  PushData(dIndexList);       break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void PrimGroupSkinnedTriangle::Finalize(const GPUQueue& queue)
{
        size_t batchCount = this->itemRanges.size();
    std::vector<Vector4ui> hVertexIndexRanges;
    hVertexIndexRanges.reserve(batchCount);

    DeviceLocalMemory tempMem(*queue.Device());
    Span<Vector4ui> dVertexIndexRanges;
    MemAlloc::AllocateMultiData(std::tie(dVertexIndexRanges),
                                tempMem,
                                {batchCount});

    for(const auto& kv : this->itemRanges)
    {
        const AttributeRanges& ranges = kv.second;
        hVertexIndexRanges.emplace_back(Vector4ui(ranges[POSITION_ATTRIB_INDEX][0],
                                                  ranges[POSITION_ATTRIB_INDEX][1],
                                                  ranges[INDICES_ATTRIB_INDEX][0],
                                                  ranges[INDICES_ATTRIB_INDEX][1]));
    }
    queue.MemcpyAsync(dVertexIndexRanges,
                      Span<const Vector4ui>(hVertexIndexRanges.cbegin(),
                                            hVertexIndexRanges.end()));

    uint32_t blockCount = queue.RecommendedBlockCountDevice
    (
        reinterpret_cast<const void*>(&KCAdjustIndices),
        StaticThreadPerBlock1D(),
        0
    );
    using namespace std::string_view_literals;
    queue.IssueBlockKernel<KCAdjustIndices>
    (
        "KCAdjustIndices"sv,
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = StaticThreadPerBlock1D()
        },
        // I-O
        dIndexList,
        dVertexIndexRanges
    );
    queue.Barrier().Wait();
}

inline
Vector2ui PrimGroupSkinnedTriangle::BatchRange(PrimBatchKey key) const
{
    auto range = FindRange(static_cast<CommonKey>(key))[INDICES_ATTRIB_INDEX];
    return Vector2ui(range);
}

size_t PrimGroupSkinnedTriangle::TotalPrimCount() const
{
    return this->TotalPrimCountImpl(0);
}

typename PrimGroupSkinnedTriangle::DataSoA PrimGroupSkinnedTriangle::SoA() const
{
    return soa;
}

Span<const Vector3ui> PrimGroupSkinnedTriangle::GetIndexSpan() const
{
    return ToConstSpan(dIndexList);
}

Span<const Vector3> PrimGroupSkinnedTriangle::GetVertexPositionSpan() const
{
    return ToConstSpan(dPositions);
}