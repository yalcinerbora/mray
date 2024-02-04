#include "MeshLoaderJson.h"

MeshFileJson::MeshFileJson(const nlohmann::json& jsonObject)
    : jsonNode(jsonObject)
{}

AABB3 MeshFileJson::AABB(uint32_t) const
{
    //return std::nullopt;

    //result[j] = NegativeAABB3;
    //for(size_t i = 0; i < (posList.size() / 3); i++)
    //{
    //    result[j].UnionSelf(Triangle::BoundingBox(posList[i * 3 + 0],
    //                                              posList[i * 3 + 1],
    //                                              posList[i * 3 + 2]));
    //}
    return AABB3::Zero();
}

uint32_t MeshFileJson::MeshPrimitiveCount(uint32_t) const
{
    return 0;
}

uint32_t MeshFileJson::MeshAttributeCount(uint32_t) const
{
    return 0;
}

bool MeshFileJson::HasAttribute(PrimitiveAttributeLogic, uint32_t) const
{
    return false;
}

MRayInput MeshFileJson::GetAttribute(PrimitiveAttributeLogic, uint32_t) const
{
    return MRayInput(std::in_place_type_t<Byte>{}, 0);
}

MRayDataTypeRT MeshFileJson::AttributeLayout(PrimitiveAttributeLogic, uint32_t) const
{
    return MRayDataTypeRT(MRayDataType<MRayDataEnum::MR_CHAR>{});
}

std::string MeshFileJson::Name() const
{
    // https://github.com/nlohmann/json/discussions/3508
    // Currently json pointer is an internal state object
    // Return empty string
    return std::string("");
}