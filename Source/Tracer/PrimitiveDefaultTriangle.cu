#include "PrimitiveDefaultTriangle.h"
#include "Device/GPUSystem.hpp"

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
        if(localTid < Vector4ui::Dims)
            sVertexIndexRange[localTid] = dVertexIndexRanges[blockId][localTid];
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

PrimGroupTriangle::PrimGroupTriangle(uint32_t primGroupId,
                                     const GPUSystem& sys)
    : GenericGroupPrimitive(primGroupId, sys,
                            DefaultTriangleDetail::DeviceMemAllocationGranularity,
                            DefaultTriangleDetail::DeviceMemReservationSize)
{}

void PrimGroupTriangle::CommitReservations()
{
    std::array<size_t, AttributeCount> countLookup = {1, 1, 1, 0};
    auto [p, n, uv, i] = this->GenericCommit<Vector3, Quaternion,
                                             Vector2, Vector3ui>(countLookup);

    dPositions = p;
    dTBNRotations = n;
    dUVs = uv;
    dIndexList = i;

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
        PrimAttributeInfo(NORMAL,   MRayDataType<MR_QUATERNION>(),  IS_SCALAR, MR_OPTIONAL),
        PrimAttributeInfo(UV0,      MRayDataType<MR_VECTOR_2>(),    IS_SCALAR, MR_OPTIONAL),
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

    uint32_t blockCount = queue.SMCount() *
        GPUQueue::RecommendedBlockCountPerSM(&KCAdjustIndices,
                                             StaticThreadPerBlock1D(),
                                             0);

    using namespace std::string_view_literals;
    queue.IssueExactKernel<KCAdjustIndices>
    (
        "KCAdjustIndices"sv,
        KernelExactIssueParams
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

Vector2ui PrimGroupTriangle::BatchRange(PrimBatchKey key) const
{
    auto range = FindRange(static_cast<CommonKey>(key))[INDICES_ATTRIB_INDEX];
    return Vector2ui(range);
}

typename PrimGroupTriangle::DataSoA PrimGroupTriangle::SoA() const
{
    return soa;
}

PrimGroupSkinnedTriangle::PrimGroupSkinnedTriangle(uint32_t primGroupId,
                                                   const GPUSystem& sys)
    : GenericGroupPrimitive(primGroupId, sys,
                            DefaultTriangleDetail::DeviceMemAllocationGranularity,
                            DefaultTriangleDetail::DeviceMemReservationSize)
{}

void PrimGroupSkinnedTriangle::CommitReservations()
{
    std::array<size_t, AttributeCount> countLookup = {1, 1, 1,
                                                      1, 1, 0};
    auto [p, n, uv, sw, si, i]
        = GenericCommit<Vector3, Quaternion,
                        Vector2, UNorm4x8,
                        Vector4uc, Vector3ui>(countLookup);

    dPositions = p;
    dTBNRotations = n;
    dUVs = uv;
    dIndexList = i;
    dSkinWeights = sw;
    dSkinIndices = si;

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
        PrimAttributeInfo(NORMAL,       MRayDataType<MR_QUATERNION>(),  IS_SCALAR, MR_OPTIONAL),
        PrimAttributeInfo(UV0,          MRayDataType<MR_VECTOR_2>(),    IS_SCALAR, MR_OPTIONAL),
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

    uint32_t blockCount = queue.SMCount() *
        GPUQueue::RecommendedBlockCountPerSM(&KCAdjustIndices,
                                             StaticThreadPerBlock1D(),
                                             0);
    using namespace std::string_view_literals;
    queue.IssueExactKernel<KCAdjustIndices>
    (
        "KCAdjustIndices"sv,
        KernelExactIssueParams
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

typename PrimGroupSkinnedTriangle::DataSoA PrimGroupSkinnedTriangle::SoA() const
{
    return soa;
}