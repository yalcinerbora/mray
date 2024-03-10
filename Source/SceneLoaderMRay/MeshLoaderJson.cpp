#include "MeshLoaderJson.h"
#include "JsonNode.h"

JsonTriangle::JsonTriangle(const JsonNode& jn, bool isIndexed)
    : id(jn.Id())
    , positions(jn.AccessDataArray<Vector3>(JsonMeshNames::NODE_POSITION))
    , indices((isIndexed)
                ? jn.AccessDataArray<Vector3ui>(JsonMeshNames::NODE_INDEX)
                : TransientData(std::in_place_type_t<Vector3ui>(),
                                positions.AccessAs<uint32_t>().size()))
    , normals(jn.AccessOptionalDataArray<Vector3>(JsonMeshNames::NODE_NORMAL))
    , uvs(jn.AccessOptionalDataArray<Vector2>(JsonMeshNames::NODE_UV))
{
    // Generate indices if not indexed
    if(!isIndexed)
    {
        auto indexSpan = indices.AccessAs<Vector3ui>();
        Vector3ui initialIndex = Vector3ui(0, 1, 2);
        for(auto& index : indexSpan)
        {
            index = initialIndex;
            initialIndex = initialIndex + 3;
        }
    }
}

AABB3 JsonTriangle::AABB(uint32_t) const
{
    auto indexSpan = indices.AccessAs<Vector3ui>();
    auto positionSpan = positions.AccessAs<Vector3>();

    AABB3 result = AABB3::Negative();
    std::for_each(indexSpan.begin(), indexSpan.end(),
                  [&result, positionSpan](const Vector3ui& index)
    {
        result.UnionSelf(AABB3(positionSpan[index[0]],
                               positionSpan[index[0]]));
        result.UnionSelf(AABB3(positionSpan[index[1]],
                               positionSpan[index[1]]));
        result.UnionSelf(AABB3(positionSpan[index[2]],
                               positionSpan[index[2]]));
    });
    return result;
}

uint32_t JsonTriangle::MeshPrimitiveCount(uint32_t) const
{
    return static_cast<uint32_t>(indices.AccessAs<Vector3ui>().size());
}

uint32_t JsonTriangle::MeshAttributeCount(uint32_t) const
{
    return static_cast<uint32_t>(positions.AccessAs<Vector3>().size());
}

bool JsonTriangle::HasAttribute(PrimitiveAttributeLogic attribLogic, uint32_t) const
{
    using enum PrimitiveAttributeLogic;
    switch(attribLogic)
    {
        case POSITION:  return true;
        case NORMAL:    return normals.has_value();
        case UV0:       return uvs.has_value();
        default:        return false;
    }
}

TransientData JsonTriangle::GetAttribute(PrimitiveAttributeLogic attribLogic, uint32_t) const
{
    auto ExplicitCopy = []<class T>(const TransientData& t) ->  TransientData
    {
        auto inSpan = t.AccessAs<T>();
        TransientData result(std::in_place_type_t<T>(), inSpan.size());
        auto outSpan = result.AccessAs<T>();
        std::copy(inSpan.begin(), inSpan.end(), outSpan.begin());
        return std::move(result);
    };

    using enum PrimitiveAttributeLogic;
    switch(attribLogic)
    {
        case POSITION:  return ExplicitCopy.operator()<Vector3>(positions);
        case NORMAL:    return ExplicitCopy.operator()<Vector3>(normals.value());
        case UV0:       return ExplicitCopy.operator()<Vector2>(uvs.value());
        case INDEX:     return ExplicitCopy.operator()<Vector3ui>(indices);
        default:        throw MRayError("Unknown attribute logic!");
    }
}

MRayDataTypeRT JsonTriangle::AttributeLayout(PrimitiveAttributeLogic attribLogic, uint32_t) const
{
    using enum MRayDataEnum;
    if(attribLogic == PrimitiveAttributeLogic::POSITION ||
       attribLogic == PrimitiveAttributeLogic::NORMAL)
        return MRayDataTypeRT(MRayDataType<MR_VECTOR_3>());
    else if(attribLogic == PrimitiveAttributeLogic::UV0)
        return MRayDataTypeRT(MRayDataType<MR_VECTOR_2>());
    else if(attribLogic == PrimitiveAttributeLogic::INDEX)
        return MRayDataTypeRT(MRayDataType<MR_VECTOR_3UI>());
    else
        throw MRayError("Unknown attribute logic!");
}

std::string JsonTriangle::Name() const
{
    return std::string("Json Triangle") + std::to_string(id);
}

//==============================================//
//      Sphere                                  //
//==============================================//

JsonSphere::JsonSphere(const JsonNode& jn)
    : id(jn.Id())
    , position(jn.AccessData<Vector3>(JsonMeshNames::NODE_POSITION))
    , radius(jn.AccessData<Float>(JsonMeshNames::NODE_RADIUS))
{}

AABB3 JsonSphere::AABB(uint32_t) const
{
    return AABB3(position - radius, position + radius);
}

uint32_t JsonSphere::MeshPrimitiveCount(uint32_t) const
{
    return 1;
}

uint32_t JsonSphere::MeshAttributeCount(uint32_t) const
{
    return 1;
}

bool JsonSphere::HasAttribute(PrimitiveAttributeLogic attribLogic, uint32_t) const
{
    using enum PrimitiveAttributeLogic;
    switch(attribLogic)
    {
        case POSITION:  return true;
        case RADIUS:    return true;
        default:        return false;
    }
}

TransientData JsonSphere::GetAttribute(PrimitiveAttributeLogic attribLogic, uint32_t) const
{
    using namespace JsonMeshNames;
    if(attribLogic == PrimitiveAttributeLogic::POSITION)
    {
        TransientData result(std::in_place_type_t<Vector3>{}, 1);
        result.Push(Span<const Vector3>(&position, 1));
        return std::move(result);
    }
    else if(attribLogic == PrimitiveAttributeLogic::RADIUS)
    {
        TransientData result(std::in_place_type_t<Float>{}, 1);
        result.Push(Span<const Float>(&radius, 1));
        return std::move(result);
    }
    else throw MRayError("Unknown attribute logic!");
}

MRayDataTypeRT JsonSphere::AttributeLayout(PrimitiveAttributeLogic attribLogic, uint32_t) const
{
    using enum MRayDataEnum;
    using enum PrimitiveAttributeLogic;
    switch(attribLogic)
    {
        case POSITION:  return MRayDataTypeRT(MRayDataType<MR_VECTOR_3>());
        case RADIUS:    return MRayDataTypeRT(MRayDataType<MR_DEFAULT_FLT>());
        default:        throw MRayError("Unknown attribute logic!");
    }
}

std::string JsonSphere::Name() const
{
    return std::string("Json Sphere") + std::to_string(id);
}