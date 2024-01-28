#include "PrimitiveDefaultTriangle.h"

std::string_view PrimGroupTriangle::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(P)DefaultTriangle"sv;
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
    using enum AttributeOptionality;
    static const std::array<PrimAttributeInfo, AttributeCount> LogicList =
    {
        PrimAttributeInfo(PrimAttributeConverter::ToString(POSITION), MR_MANDATORY, MRayDataType<MR_VECTOR_3>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(NORMAL),   MR_OPTIONAL,  MRayDataType<MR_QUATERNION>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(UV0),      MR_OPTIONAL,  MRayDataType<MR_VECTOR_2>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(INDEX),    MR_MANDATORY, MRayDataType<MR_VECTOR_3UI>()),
    };
    return std::vector(LogicList.cbegin(), LogicList.cend());
}

void PrimGroupTriangle::PushAttribute(PrimBatchKey batchKey,
                                      uint32_t attributeIndex,
                                      MRayInput data)
{
    auto PushData = [&]<class T>(const Span<T>&d, bool isPerPrimitive)
    {
        GenericPushData(batchKey, d, std::move(data), isPerPrimitive);
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

void PrimGroupTriangle::PushAttribute(PrimBatchKey batchKey,
                                      const Vector2ui& subRange,
                                      uint32_t attributeIndex,
                                      MRayInput data)
{
    auto PushData = [&]<class T>(const Span<T>&d, bool isPerPrimitive)
    {
        GenericPushData(batchKey, subRange, d,
                        std::move(data), isPerPrimitive);
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

void PrimGroupTriangle::PushAttribute(const Vector<2, PrimBatchKey::Type>& idRange,
                                      uint32_t attributeIndex,
                                      MRayInput data)
{
    auto PushData = [&]<class T>(const Span<T>&d, bool isPerPrimitive)
    {
        GenericPushData(idRange, d, std::move(data), false, isPerPrimitive);
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
    : GenericGroupT(primGroupId, sys,
                    DefaultTriangleDetail::DeviceMemAllocationGranularity,
                    DefaultTriangleDetail::DeviceMemReservationSize)
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
    using enum AttributeOptionality;
    static const std::array<PrimAttributeInfo, AttributeCount> LogicList =
    {
        PrimAttributeInfo(PrimAttributeConverter::ToString(POSITION),
                          MR_MANDATORY, MRayDataType<MR_VECTOR_3>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(NORMAL),
                          MR_OPTIONAL, MRayDataType<MR_QUATERNION>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(UV0),
                          MR_OPTIONAL, MRayDataType<MR_VECTOR_2>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(WEIGHT),
                          MR_MANDATORY, MRayDataType<MR_UNORM_4x8>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(WEIGHT_INDEX),
                          MR_MANDATORY, MRayDataType<MR_VECTOR_4UC>()),
        PrimAttributeInfo(PrimAttributeConverter::ToString(INDEX),
                          MR_MANDATORY, MRayDataType<MR_VECTOR_3UI>())
    };
    return std::vector(LogicList.cbegin(), LogicList.cend());
}

void PrimGroupSkinnedTriangle::PushAttribute(PrimBatchKey batchKey, uint32_t attributeIndex,
                                             MRayInput data)
{
    auto PushData = [&]<class T>(const Span<T>&d, bool isPerPrimitive)
    {
        GenericPushData(batchKey, d,
                        std::move(data), isPerPrimitive);
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

void PrimGroupSkinnedTriangle::PushAttribute(PrimBatchKey batchKey,
                                             const Vector2ui& subRange,
                                             uint32_t attributeIndex,
                                             MRayInput data)
{
    auto PushData = [&]<class T>(const Span<T>&d, bool isPerPrimitive)
    {
        GenericPushData(batchKey, subRange, d,
                        std::move(data), isPerPrimitive);
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

void PrimGroupSkinnedTriangle::PushAttribute(const Vector<2, PrimBatchKey::Type>& idRange,
                                             uint32_t attributeIndex,
                                             MRayInput data)
{
    auto PushData = [&]<class T>(const Span<T>&d, bool isPerPrimitive)
    {
        GenericPushData(idRange, d, std::move(data), false, isPerPrimitive);
    };

    switch(attributeIndex)
    {
        case 0: PushData(dPositions, false);      break;  // Position
        case 1: PushData(dTBNRotations, false);   break;  // Normal
        case 2: PushData(dUVs, false);            break;  // UVs
        case 3: PushData(dSkinWeights, false);    break;  // Weights
        case 4: PushData(dSkinIndices, false);    break;  // WeightIndices
        case 5: PushData(dIndexList, true);       break;  // Indices
        default: MRAY_WARNING_LOG("{:s}: Unknown Attribute Index {:d}",
                                  TypeName(), attributeIndex);
    }
}

typename PrimGroupSkinnedTriangle::DataSoA PrimGroupSkinnedTriangle::SoA() const
{
    return soa;
}