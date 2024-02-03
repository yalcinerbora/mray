#pragma once

#include <nlohmann/json.hpp>
// Bad design change this later
#include "MeshLoader/EntryPoint.h"

class MeshFileMRayJson : public MeshFileI
{
    private:
    nlohmann::json  jsonNode;

    public:
                    MeshFileMRayJson(const nlohmann::json& jsonObject);

    AABB3           AABB(uint32_t innerId = 0) const override;
    uint32_t        MeshPrimitiveCount(uint32_t innerId = 0) const override;
    uint32_t        MeshAttributeCount(uint32_t innerId = 0) const override;
    std::string     Name() const override;

    // Entire Data Fetch
    bool            HasAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
    MRayInput       GetAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
    MRayDataTypeRT  AttributeLayout(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;

};

class MeshLoaderMRayJson : public MeshLoaderI
{
    public:
    std::unique_ptr<MeshFileI>      OpenJson(const nlohmann::json& jsonObject) override;
    private:
    // Design leak... change this later
    std::unique_ptr<MeshFileI>      OpenFile(std::string& filePath) override;
};
