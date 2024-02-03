#include "MeshLoaderMRayJson.h"


MeshFileMRayJson::MeshFileMRayJson(const nlohmann::json& jsonObject)
    : jsonNode(jsonObject)
{}

AABB3 MeshFileMRayJson::AABB(uint32_t) const
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

uint32_t MeshFileMRayJson::MeshPrimitiveCount(uint32_t) const
{
    return 0;
}

uint32_t MeshFileMRayJson::MeshAttributeCount(uint32_t) const
{
    return 0;
}

bool MeshFileMRayJson::HasAttribute(PrimitiveAttributeLogic, uint32_t) const
{
    return false;
}

MRayInput MeshFileMRayJson::GetAttribute(PrimitiveAttributeLogic, uint32_t) const
{
    return MRayInput(std::in_place_type_t<Byte>{}, 0);
}

MRayDataTypeRT MeshFileMRayJson::AttributeLayout(PrimitiveAttributeLogic, uint32_t) const
{
    return MRayDataTypeRT(MRayDataType<MRayDataEnum::MR_CHAR>{});
}

std::string MeshFileMRayJson::Name() const
{
    // https://github.com/nlohmann/json/discussions/3508
    // Currently json pointer is an internal state object
    // Return empty string
    return std::string("");
}

std::unique_ptr<MeshFileI> MeshLoaderMRayJson::OpenJson(const nlohmann::json& jsonObject)
{
    return std::unique_ptr<MeshFileI>(new MeshFileMRayJson(jsonObject));
}


std::unique_ptr<MeshFileI> MeshLoaderMRayJson::OpenFile(std::string&)
{
    throw MRayError("MRay json Mesh Loader does not support files!");
}