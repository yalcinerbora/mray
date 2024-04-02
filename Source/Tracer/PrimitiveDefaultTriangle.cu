#include "PrimitiveDefaultTriangle.h"
#include "Device/GPUAlgorithms.h"

template<class I>
struct KCAdjustIndices
{
    private:
    uint32_t attributeOffset;
    public:
    MRAY_HYBRID     KCAdjustIndices(uint32_t attributeOffset);
    MRAY_HYBRID I   operator()(const I& t) const;
};

template<class I>
MRAY_HYBRID MRAY_CGPU_INLINE
KCAdjustIndices<I>::KCAdjustIndices(uint32_t attributeOffset)
    : attributeOffset(attributeOffset)
{}

template<class I>
MRAY_HYBRID MRAY_CGPU_INLINE
I KCAdjustIndices<I>::operator()(const I& t) const
{
    return t + attributeOffset;
}

std::string_view PrimGroupTriangle::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(P)Triangle"sv;
    return name;
}

PrimGroupTriangle::PrimGroupTriangle(uint32_t primGroupId,
                                     const GPUSystem& sys)
    : GenericGroupT(primGroupId, sys,
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
        case 0  : PushData(dPositions);     break;  // Position
        case 1  : PushData(dTBNRotations);  break;  // Normal
        case 2  : PushData(dUVs);           break;  // UVs
        case 3  : PushData(dIndexList);     break;  // Indices
        default :
        {
            MRAY_ERROR_LOG("{:s}: Unknown Attribute Index {:d}",
                           TypeName(), attributeIndex);
            return;
        }
    }

    if(attributeIndex != 3) return;

    IdInt batch = batchKey.FetchIndexPortion();
    auto attributeStart = static_cast<uint32_t>(this->itemRanges.at(batch)[0][0]);
    auto range = this->itemRanges.at(batch)[3];
    size_t count = range[1] - range[0];
    Span<Vector3ui> batchSpan = dIndexList.subspan(range[0], count);

    DeviceAlgorithms::InPlaceTransform(batchSpan, queue,
                                        KCAdjustIndices<Vector3ui>(attributeStart));
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
        case 0  : PushData(dPositions);     break;  // Position
        case 1  : PushData(dTBNRotations);  break;  // Normal
        case 2  : PushData(dUVs);           break;  // UVs
        case 3  : PushData(dIndexList);     break;  // Indices
        default :
        {
            MRAY_ERROR_LOG("{:s}: Unknown Attribute Index {:d}",
                           TypeName(), attributeIndex);
            return;
        }
    }

    if(attributeIndex != 3) return;

    IdInt batch = batchKey.FetchIndexPortion();
    auto attributeStart = static_cast<uint32_t>(this->itemRanges.at(batch)[0][0]);
    auto range = this->itemRanges.at(batch)[3];
    auto innerRange = Vector2ui(range[0] + subRange[0], subRange[1]);
    size_t count = innerRange[1] - innerRange[0];
    Span<Vector3ui> batchSpan = dIndexList.subspan(innerRange[0], count);

    DeviceAlgorithms::InPlaceTransform(batchSpan, queue,
                                       KCAdjustIndices<Vector3ui>(attributeStart));
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
        case 0  : PushData(dPositions);     break;  // Position
        case 1  : PushData(dTBNRotations);  break;  // Normal
        case 2  : PushData(dUVs);           break;  // UVs
        case 3  : PushData(dIndexList);     break;  // Indices
        default:
        {
            MRAY_ERROR_LOG("{:s}: Unknown Attribute Index {:d}",
                           TypeName(), attributeIndex);
            return;
        }
    }

    if(attributeIndex != 3) return;
    // Now here we need to do for loop
    for(auto i = idStart.FetchIndexPortion();
        i < idEnd.FetchIndexPortion(); i++)
    {
        auto attributeStart = static_cast<uint32_t>(this->itemRanges.at(i)[0][0]);
        auto range = this->itemRanges.at(i)[3];
        size_t count = range[1] - range[0];
        Span<Vector3ui> batchSpan = dIndexList.subspan(range[0], count);

        DeviceAlgorithms::InPlaceTransform(batchSpan, queue,
                                           KCAdjustIndices<Vector3ui>(attributeStart));
    }

}

typename PrimGroupTriangle::DataSoA PrimGroupTriangle::SoA() const
{
    return soa;
}

std::string_view PrimGroupSkinnedTriangle::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(P)TriangleSkinned"sv;
    return name;
}

PrimGroupSkinnedTriangle::PrimGroupSkinnedTriangle(uint32_t primGroupId,
                                                   const GPUSystem& sys)
    : GenericGroupT(primGroupId, sys,
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
        case 0  : PushData(dPositions);     break;  // Position
        case 1  : PushData(dTBNRotations);  break;  // Normal
        case 2  : PushData(dUVs);           break;  // UVs
        case 3  : PushData(dSkinWeights);   break;  // Weights
        case 4  : PushData(dSkinIndices);   break;  // Weights
        case 5  : PushData(dIndexList);     break;  // Indices
        default :
        {
            MRAY_ERROR_LOG("{:s}: Unknown Attribute Index {:d}",
                           TypeName(), attributeIndex);
            return;
        }
    }

    if(attributeIndex != 3) return;

    IdInt batch = batchKey.FetchIndexPortion();
    auto attributeStart = static_cast<uint32_t>(this->itemRanges.at(batch)[0][0]);
    auto range = this->itemRanges.at(batch)[3];
    size_t count = range[1] - range[0];
    Span<Vector3ui> batchSpan = dIndexList.subspan(range[0], count);

    DeviceAlgorithms::InPlaceTransform(batchSpan, queue,
                                       KCAdjustIndices<Vector3ui>(attributeStart));
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
        case 0  : PushData(dPositions);     break;  // Position
        case 1  : PushData(dTBNRotations);  break;  // Normal
        case 2  : PushData(dUVs);           break;  // UVs
        case 3  : PushData(dSkinWeights);   break;  // Weights
        case 4  : PushData(dSkinIndices);   break;  // WeightIndices
        case 5  : PushData(dIndexList);     break;  // Indices
        default :
        {
            MRAY_ERROR_LOG("{:s}: Unknown Attribute Index {:d}",
                           TypeName(), attributeIndex);
            return;
        }
    }

    if(attributeIndex != 3) return;

    IdInt batch = batchKey.FetchIndexPortion();
    auto attributeStart = static_cast<uint32_t>(this->itemRanges.at(batch)[0][0]);
    auto range = this->itemRanges.at(batch)[3];
    auto innerRange = Vector2ui(range[0] + subRange[0], subRange[1]);
    size_t count = innerRange[1] - innerRange[0];
    Span<Vector3ui> batchSpan = dIndexList.subspan(innerRange[0], count);

    DeviceAlgorithms::InPlaceTransform(batchSpan, queue,
                                       KCAdjustIndices<Vector3ui>(attributeStart));
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
        case 0: PushData(dPositions);       break;  // Position
        case 1: PushData(dTBNRotations);    break;  // Normal
        case 2: PushData(dUVs);             break;  // UVs
        case 3: PushData(dSkinWeights);     break;  // Weights
        case 4: PushData(dSkinIndices);     break;  // WeightIndices
        case 5: PushData(dIndexList);       break;  // Indices
        default :
        {
            MRAY_ERROR_LOG("{:s}: Unknown Attribute Index {:d}",
                           TypeName(), attributeIndex);
            return;
        }
    }

    if(attributeIndex != 3) return;
    // Now here we need to do for loop
    for(auto i = idStart.FetchIndexPortion();
        i < idEnd.FetchIndexPortion(); i++)
    {
        auto attributeStart = static_cast<uint32_t>(this->itemRanges.at(i)[0][0]);
        auto range = this->itemRanges.at(i)[3];
        size_t count = range[1] - range[0];
        Span<Vector3ui> batchSpan = dIndexList.subspan(range[0], count);

        DeviceAlgorithms::InPlaceTransform(batchSpan, queue,
                                           KCAdjustIndices<Vector3ui>(attributeStart));
    }
}

typename PrimGroupSkinnedTriangle::DataSoA PrimGroupSkinnedTriangle::SoA() const
{
    return soa;
}