#pragma once

#include <string_view>
#include "MeshLoader/EntryPoint.h"
#include "TransientPool/TransientPool.h"

class JsonNode;

namespace JsonMeshNames
{
    using namespace std;
    using namespace literals;

    static constexpr string_view NODE_POSITION  = "position"sv;
    static constexpr string_view NODE_NORMAL    = "normal"sv;
    static constexpr string_view NODE_UV        = "uv"sv;
    static constexpr string_view NODE_INDEX     = "index"sv;
    static constexpr string_view NODE_RADIUS    = "radius"sv;
}

class JsonTriangle : public MeshFileI
{
    private:
    uint32_t                id;
    TransientData           positions;
    TransientData           indices;
    Optional<TransientData> normals;
    Optional<TransientData> uvs;
    Optional<TransientData> tangents;
    Optional<TransientData> bitangents;

    public:
                    JsonTriangle(const JsonNode&, bool isIndexed);

    AABB3           AABB(uint32_t innerId = 0) const override;
    uint32_t        MeshPrimitiveCount(uint32_t innerId = 0) const override;
    uint32_t        MeshAttributeCount(uint32_t innerId = 0) const override;
    std::string     Name() const override;

    // Entire Data Fetch
    bool            HasAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
    TransientData   GetAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
    MRayDataTypeRT  AttributeLayout(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
};

class JsonSphere : public MeshFileI
{
    private:
    uint32_t        id;
    Vector3         position;
    Float           radius;

    public:
                    JsonSphere(const JsonNode&);

    AABB3           AABB(uint32_t innerId = 0) const override;
    uint32_t        MeshPrimitiveCount(uint32_t innerId = 0) const override;
    uint32_t        MeshAttributeCount(uint32_t innerId = 0) const override;
    std::string     Name() const override;

    // Entire Data Fetch
    bool            HasAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
    TransientData   GetAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
    MRayDataTypeRT  AttributeLayout(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
};

