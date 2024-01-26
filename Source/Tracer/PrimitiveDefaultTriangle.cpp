#include "PrimitiveDefaultTriangle.h"

std::string_view PrimGroupTriangle::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(P)DefaultTriangle"sv;
    return name;
}

PrimGroupTriangle::PrimGroupTriangle(uint32_t primGroupId,
                         const GPUSystem& sys)
    : PrimitiveGroupT(primGroupId, sys)
{}

void PrimGroupTriangle::CommitReservations()
{
    std::array<bool, AttributeCount> isAttribute = {false, false, false, true};
    auto [p, n, uv, i] = GenericCommit<Vector3, Quaternion,
                                       Vector2, Vector3ui>(isAttribute);

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
    static const std::array<PrimAttributeInfo, AttributeCount> LogicList =
    {
        PrimAttributeInfo(PrimAttributeConverter::ToString(POSITION), MRayDataType<MR_VECTOR_3>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(NORMAL),   MRayDataType<MR_QUATERNION>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(UV0),      MRayDataType<MR_VECTOR_2>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(INDEX),    MRayDataType<MR_VECTOR_3UI>()),
    };
    return std::vector(LogicList.cbegin(), LogicList.cend());
}

void PrimGroupTriangle::PushAttribute(PrimBatchId batchId, uint32_t attributeIndex,
                                      std::vector<Byte> data)
{
    auto PushData = [&]<class T>(const Span<T>&d, bool isPerPrimitive)
    {
        GenericPushData(batchId, d, data, isPerPrimitive);
    };

    switch(attributeIndex)
    {
        case 0  : PushData(dPositions, false);      break;  // Position
        case 1  : PushData(dTBNRotations, false);   break;  // Normal
        case 2  : PushData(dUVs, false);            break;  // UVs
        case 3  : PushData(dIndexList, true);       break;  // Indices
        default : MRAY_WARNING_LOG("{:s}: Unknown Attribute Index {:d}",
                                   TypeName(), attributeIndex);
    }
}

void PrimGroupTriangle::PushAttribute(PrimBatchId batchId,
                                      uint32_t attributeIndex,
                                      const Vector2ui& subBatchRange,
                                      std::vector<Byte> data)
{
    auto PushData = [&]<class T>(const Span<T>&d, bool isPerPrimitive)
    {
        GenericPushData(batchId, d, subBatchRange, data, isPerPrimitive);
    };

    switch(attributeIndex)
    {
        case 0  : PushData(dPositions, false);      break;  // Position
        case 1  : PushData(dTBNRotations, false);   break;  // Normal
        case 2  : PushData(dUVs, false);            break;  // UVs
        case 3  : PushData(dIndexList, true);       break;  // Indices
        default : MRAY_WARNING_LOG("{:s}: Unknown Attribute Index {:d}",
                                   TypeName(), attributeIndex);
    }
}

typename PrimGroupTriangle::DataSoA PrimGroupTriangle::SoA() const
{
    return soa;
}

std::string_view PrimGroupSkinnedTriangle::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(P)DefaultTriangleSkinned"sv;
    return name;
}

PrimGroupSkinnedTriangle::PrimGroupSkinnedTriangle(uint32_t primGroupId,
                                                   const GPUSystem& sys)
    : PrimitiveGroupT(primGroupId, sys)
{}

void PrimGroupSkinnedTriangle::CommitReservations()
{
    std::array<bool, AttributeCount> isAttribute = {false, false, false,
                                                    false, false, true};
    auto [p, n, uv, sw, si, i]
        = GenericCommit<Vector3, Quaternion,
                        Vector2, UNorm4x8,
                        Vector4uc, Vector3ui>(isAttribute);

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
    static const std::array<PrimAttributeInfo, AttributeCount> LogicList =
    {
        PrimAttributeInfo(PrimAttributeConverter::ToString(POSITION),     MRayDataType<MR_VECTOR_3>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(NORMAL),       MRayDataType<MR_QUATERNION>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(UV0),          MRayDataType<MR_VECTOR_2>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(WEIGHT),       MRayDataType<MR_UNORM_4x8>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(WEIGHT_INDEX), MRayDataType<MR_VECTOR_4UC>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(INDEX),        MRayDataType<MR_VECTOR_3UI>())
    };
    return std::vector(LogicList.cbegin(), LogicList.cend());
}

void PrimGroupSkinnedTriangle::PushAttribute(PrimBatchId batchId, uint32_t attributeIndex,
                                             std::vector<Byte> data)
{
    auto PushData = [&]<class T>(const Span<T>&d, bool isPerPrimitive)
    {
        GenericPushData(batchId, d, data, isPerPrimitive);
    };

    switch(attributeIndex)
    {
        case 0  : PushData(dPositions, false);      break;  // Position
        case 1  : PushData(dTBNRotations, false);   break;  // Normal
        case 2  : PushData(dUVs, false);            break;  // UVs
        case 3  : PushData(dSkinWeights, false);    break;  // Weights
        case 4  : PushData(dSkinIndices, false);    break;  // Weights
        case 5  : PushData(dIndexList, true);       break;  // Indices
        default : MRAY_WARNING_LOG("{:s}: Unknown Attribute Index {:d}",
                                   TypeName(), attributeIndex);
    }
}

void PrimGroupSkinnedTriangle::PushAttribute(PrimBatchId batchId, uint32_t attributeIndex,
                                             const Vector2ui& subBatchRange,
                                             std::vector<Byte> data)
{
    auto PushData = [&]<class T>(const Span<T>&d, bool isPerPrimitive)
    {
        GenericPushData(batchId, d, subBatchRange, data, isPerPrimitive);
    };

    switch(attributeIndex)
    {
        case 0  : PushData(dPositions, false);      break;  // Position
        case 1  : PushData(dTBNRotations, false);   break;  // Normal
        case 2  : PushData(dUVs, false);            break;  // UVs
        case 3  : PushData(dSkinWeights, false);    break;  // Weights
        case 4  : PushData(dSkinIndices, false);    break;  // WeightIndices
        case 5  : PushData(dIndexList, true);       break;  // Indices
        default : MRAY_WARNING_LOG("{:s}: Unknown Attribute Index {:d}",
                                   TypeName(), attributeIndex);
    }
}

typename PrimGroupSkinnedTriangle::DataSoA PrimGroupSkinnedTriangle::SoA() const
{
    return soa;
}