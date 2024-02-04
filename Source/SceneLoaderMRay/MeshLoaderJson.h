#pragma once

#include <nlohmann/json.hpp>
#include "MeshLoader/EntryPoint.h"

class MeshFileJson : public MeshFileI
{
    private:
    nlohmann::json  jsonNode;

    public:
                    MeshFileJson(const nlohmann::json& jsonObject);

    AABB3           AABB(uint32_t innerId = 0) const override;
    uint32_t        MeshPrimitiveCount(uint32_t innerId = 0) const override;
    uint32_t        MeshAttributeCount(uint32_t innerId = 0) const override;
    std::string     Name() const override;

    // Entire Data Fetch
    bool            HasAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
    MRayInput       GetAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
    MRayDataTypeRT  AttributeLayout(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;

};

