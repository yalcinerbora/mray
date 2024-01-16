#include "PrimitiveDefaultTriangle.h"

std::string_view PrimGroupTriangle::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(P)DefaultTriangle"sv;
    return name;
}

void PrimGroupTriangle::CommitReservations()
{
    std::array<bool, AttributeCount> isAttribute = {true, true, true, false};
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

uint32_t PrimGroupTriangle::GetAttributeCount() const
{
    return AttributeCount;
}

PrimAttributeInfo PrimGroupTriangle::GetAttributeInfo(uint32_t attributeIndex) const
{
    assert(attributeIndex < AttributeCount);
    using enum MRayDataEnum;
    using enum PrimitiveAttributeLogic;
    static const std::array<PrimAttributeInfo, AttributeCount> LogicList =
    {
        PrimAttributeInfo(POSITION, MRayDataType<MR_VECTOR_3>()),
        PrimAttributeInfo(INDEX,    MRayDataType<MR_VECTOR_3UI>()),
        PrimAttributeInfo(NORMAL,   MRayDataType<MR_QUATERNION>()),
        PrimAttributeInfo(UV0,      MRayDataType<MR_VECTOR_2>()),
    };
    return LogicList[attributeIndex];
}

void PrimGroupTriangle::PushAttributeData(PrimBatchId, uint32_t,
                                          std::vector<Byte>)
{
    //assert(attributeIndex < AttributeCount);
}

void PrimGroupTriangle::PushAttributeData(PrimBatchId, uint32_t,
                                          Vector2ui, std::vector<Byte>)
{
    //assert(attributeIndex < AttributeCount);
}


std::string_view PrimGroupSkinnedTriangle::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(P)DefaultTriangleSkinned"sv;
    return name;
}

void PrimGroupSkinnedTriangle::CommitReservations()
{
    std::array<bool, AttributeCount> isAttribute = {true, true, true,
                                                    true, true, false};
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

uint32_t PrimGroupSkinnedTriangle::GetAttributeCount() const
{
    return AttributeCount;
}

PrimAttributeInfo PrimGroupSkinnedTriangle::GetAttributeInfo(uint32_t attributeIndex) const
{
    assert(attributeIndex < AttributeCount);
    using enum MRayDataEnum;
    using enum PrimitiveAttributeLogic;
    static const std::array<PrimAttributeInfo, AttributeCount> LogicList =
    {
        PrimAttributeInfo(POSITION,     MRayDataType<MR_VECTOR_3>()),
        PrimAttributeInfo(INDEX,        MRayDataType<MR_VECTOR_3UI>()),
        PrimAttributeInfo(NORMAL,       MRayDataType<MR_QUATERNION>()),
        PrimAttributeInfo(UV0,          MRayDataType<MR_VECTOR_2>()),
        PrimAttributeInfo(WEIGHT,       MRayDataType<MR_UNORM_4x8>()),
        PrimAttributeInfo(WEIGHT_INDEX, MRayDataType<MR_VECTOR_4UC>()),
    };
    return LogicList[attributeIndex];
}

void PrimGroupSkinnedTriangle::PushAttributeData(PrimBatchId, uint32_t,
                                                 std::vector<Byte>)
{

}

void PrimGroupSkinnedTriangle::PushAttributeData(PrimBatchId, uint32_t,
                                                 Vector2ui, std::vector<Byte>)
{
    //assert(attributeIndex < AttributeCount);
}