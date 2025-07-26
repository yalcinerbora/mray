#include "MeshLoaderJson.h"
#include "JsonNode.h"
#include "Core/ShapeFunctions.h"

// TODO: This is bad design reiterate over this
JsonTriangle::JsonTriangle(const JsonNode& jn, bool isIndexed)
    : id(jn.Id())
    , positions(jn.AccessDataArray<Vector3>(JsonMeshNames::NODE_POSITION))
    , indices((isIndexed)
                ? jn.AccessDataArray<Vector3ui>(JsonMeshNames::NODE_INDEX)
                : TransientData(std::in_place_type_t<Vector3ui>(),
                                positions.AccessAs<Vector3>().size() /
                                Shape::Triangle::TRI_VERTEX_COUNT))
    , normals(jn.AccessOptionalDataArray<Vector3>(JsonMeshNames::NODE_NORMAL))
    , uvs(jn.AccessOptionalDataArray<Vector2>(JsonMeshNames::NODE_UV))
{
    // Generate indices if not indexed
    if(!isIndexed)
    {
        size_t pc = (positions.AccessAs<Vector3>().size() /
                     Shape::Triangle::TRI_VERTEX_COUNT);
        uint32_t primCount = static_cast<uint32_t>(pc);
        Vector3ui initialIndex = Vector3ui(0, 1, 2);
        for(uint32_t i = 0; i < primCount; i++)
        {
            indices.Push(Span<const Vector3ui>(&initialIndex, 1));
            initialIndex = initialIndex + 3;
        }
    }
    size_t attribCount = MeshAttributeCount();
    size_t primCount = MeshPrimitiveCount();
    Span<Vector3ui> iSpan = indices.AccessAs<Vector3ui>();
    Span<Vector3> pSpan = positions.AccessAs<Vector3>();

    // We do not have normals, calculate
    if(!normals.has_value())
    {
        normals = TransientData(std::in_place_type_t<Vector3>(), attribCount);
        TransientData& data = normals.value();
        // Zero the data
        for(size_t i = 0; i < attribCount; i++)
        {
            Vector3 zero = Vector3::Zero();
            data.Push(Span<const Vector3>(&zero, 1));
        }
        Span<Vector3> nSpan = data.AccessAs<Vector3>();
        // Calculate normals
        for(size_t i = 0; i < primCount; i++)
        {
            Vector3ui index = iSpan[i];

            using namespace Shape::Triangle;
            std::array<Vector3, TRI_VERTEX_COUNT> p =
            {
                pSpan[index[0]],
                pSpan[index[1]],
                pSpan[index[2]]
            };
            Vector3 normal = Normal(p);

            // Add the normals these may be shared
            nSpan[index[0]] += normal;
            nSpan[index[1]] += normal;
            nSpan[index[2]] += normal;
        }
        // And normalize it
        for(auto& n : nSpan) n.NormalizeSelf();
    }
    Span<Vector3> nSpan = normals.value().AccessAs<Vector3>();

    // We have to create tangents (due to quaternion thing)
    tangents = TransientData(std::in_place_type_t<Vector3>(), attribCount);
    bitangents = TransientData(std::in_place_type_t<Vector3>(), attribCount);
    // Zero the data
    for(size_t i = 0; i < attribCount; i++)
    {
        Vector3 zero = Vector3::Zero();
        tangents.value().Push(Span<const Vector3>(&zero, 1));
        bitangents.value().Push(Span<const Vector3>(&zero, 1));
    }
    Span<Vector3> tSpan = tangents.value().AccessAs<Vector3>();
    Span<Vector3> bSpan = bitangents.value().AccessAs<Vector3>();

    // Utilize uvs (align the tangent to uv vectors)
    if(uvs.has_value())
    {
        Span<Vector2> uvSpan = uvs.value().AccessAs<Vector2>();
        for(size_t i = 0; i < primCount; i++)
        {
            Vector3ui index = iSpan[i];
            Vector3 n0 = nSpan[index[0]];
            Vector3 n1 = nSpan[index[1]];
            Vector3 n2 = nSpan[index[2]];

            Vector3 p0 = pSpan[index[0]];
            Vector3 p1 = pSpan[index[1]];
            Vector3 p2 = pSpan[index[2]];

            Vector2 uv0 = uvSpan[index[0]];
            Vector2 uv1 = uvSpan[index[1]];
            Vector2 uv2 = uvSpan[index[2]];

            // Generate the tangents for this triangle orientation
            using namespace Shape::Triangle;
            Vector3 t0 = CalculateTangent(n0, {p0, p1, p2},
                                          {uv0, uv1, uv2});
            Vector3 t1 = CalculateTangent(n1, {p1, p2, p0},
                                          {uv1, uv2, uv0});
            Vector3 t2 = CalculateTangent(n2, {p2, p0, p1},
                                          {uv2, uv0, uv1});
            using Math::IsFinite;
            using Graphics::OrthogonalVector;
            if(!IsFinite(t0)) t0 = OrthogonalVector(n0);
            if(!IsFinite(t1)) t1 = OrthogonalVector(n1);
            if(!IsFinite(t2)) t2 = OrthogonalVector(n2);

            // Add the normals these may be shared
            tSpan[index[0]] += t0;
            tSpan[index[1]] += t1;
            tSpan[index[2]] += t2;

        }
        // Normalize and calculate bitangent
        for(size_t i = 0; i < attribCount; i++)
        {
            tSpan[i] = Math::Normalize(tSpan[i]);
            bSpan[i] = Math::Cross(nSpan[i], tSpan[i]);
        }
    }
    // Just put random orthogonal spaces
    else
    {
        for(size_t i = 0; i < attribCount; i++)
        {
            Vector3 n = normals.value().AccessAs<Vector3>()[i];
            Vector3 t = Graphics::OrthogonalVector(n);
            Vector3 b = Math::Cross(n, t);
            tSpan[i] = t;
            bSpan[i] = b;
        }
    }


    assert(positions.IsFull());
    assert(indices.IsFull());
    assert(normals && normals->IsFull());
    assert(tangents && tangents->IsFull());
    assert(bitangents && bitangents->IsFull());
}

AABB3 JsonTriangle::AABB() const
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

uint32_t JsonTriangle::MeshPrimitiveCount() const
{
    return static_cast<uint32_t>(indices.AccessAs<Vector3ui>().size());
}

uint32_t JsonTriangle::MeshAttributeCount() const
{
    return static_cast<uint32_t>(positions.AccessAs<Vector3>().size());
}

std::string JsonTriangle::Name() const
{
    return std::string("Json Triangle") + std::to_string(id);
}

uint32_t JsonTriangle::InnerIndex() const
{
    return id;
}

bool JsonTriangle::HasAttribute(PrimitiveAttributeLogic attribLogic) const
{
    using enum PrimitiveAttributeLogic;
    switch(attribLogic)
    {
        case POSITION:
        case INDEX:
        case TANGENT:
        case BITANGENT:
        case NORMAL:
        case UV0:
            return true;
        default:
            return false;
    }
}

TransientData JsonTriangle::GetAttribute(PrimitiveAttributeLogic attribLogic) const
{
    // TODO: This is bad design reiterate over this
    auto ExplicitCopy = []<class T>(const TransientData& t) ->  TransientData
    {
        auto inSpan = t.AccessAs<T>();
        TransientData result(std::in_place_type_t<T>(), inSpan.size());
        result.Push(inSpan);
        return result;
    };

    using enum PrimitiveAttributeLogic;
    switch(attribLogic)
    {
        case POSITION:  return ExplicitCopy.operator()<Vector3>(positions);
        case NORMAL:    return ExplicitCopy.operator()<Vector3>(normals.value());
        case TANGENT:   return ExplicitCopy.operator()<Vector3>(tangents.value());
        case BITANGENT: return ExplicitCopy.operator()<Vector3>(bitangents.value());
        case UV0:       return ExplicitCopy.operator()<Vector2>(uvs.value());
        case INDEX:     return ExplicitCopy.operator()<Vector3ui>(indices);
        default:        throw MRayError("Unknown attribute logic!");
    }
}

MRayDataTypeRT JsonTriangle::AttributeLayout(PrimitiveAttributeLogic attribLogic) const
{
    using enum MRayDataEnum;
    if(attribLogic == PrimitiveAttributeLogic::POSITION ||
       attribLogic == PrimitiveAttributeLogic::NORMAL ||
       attribLogic == PrimitiveAttributeLogic::TANGENT ||
       attribLogic == PrimitiveAttributeLogic::BITANGENT)
        return MRayDataTypeRT(MRayDataType<MR_VECTOR_3>());
    else if(attribLogic == PrimitiveAttributeLogic::UV0)
        return MRayDataTypeRT(MRayDataType<MR_VECTOR_2>());
    else if(attribLogic == PrimitiveAttributeLogic::INDEX)
        return MRayDataTypeRT(MRayDataType<MR_VECTOR_3UI>());
    else
        throw MRayError("Unknown attribute logic!");
}

//==============================================//
//      Sphere                                  //
//==============================================//

JsonSphere::JsonSphere(const JsonNode& jn)
    : id(jn.Id())
    , position(jn.AccessData<Vector3>(JsonMeshNames::NODE_CENTER))
    , radius(jn.AccessData<Float>(JsonMeshNames::NODE_RADIUS))
{}

AABB3 JsonSphere::AABB() const
{
    return AABB3(position - radius, position + radius);
}

uint32_t JsonSphere::MeshPrimitiveCount() const
{
    return 1;
}

uint32_t JsonSphere::MeshAttributeCount() const
{
    return 1;
}

std::string JsonSphere::Name() const
{
    return std::string("Json Sphere") + std::to_string(id);
}


uint32_t JsonSphere::InnerIndex() const
{
    return id;
}

bool JsonSphere::HasAttribute(PrimitiveAttributeLogic attribLogic) const
{
    using enum PrimitiveAttributeLogic;
    switch(attribLogic)
    {
        case POSITION:  return true;
        case RADIUS:    return true;
        default:        return false;
    }
}

TransientData JsonSphere::GetAttribute(PrimitiveAttributeLogic attribLogic) const
{
    using namespace JsonMeshNames;
    if(attribLogic == PrimitiveAttributeLogic::POSITION)
    {
        TransientData result(std::in_place_type_t<Vector3>{}, 1);
        result.Push(Span<const Vector3>(&position, 1));
        assert(result.IsFull());
        return result;
    }
    else if(attribLogic == PrimitiveAttributeLogic::RADIUS)
    {
        TransientData result(std::in_place_type_t<Float>{}, 1);
        result.Push(Span<const Float>(&radius, 1));
        assert(result.IsFull());
        return result;
    }
    else throw MRayError("Unknown attribute logic!");
}

MRayDataTypeRT JsonSphere::AttributeLayout(PrimitiveAttributeLogic attribLogic) const
{
    using enum MRayDataEnum;
    using enum PrimitiveAttributeLogic;
    switch(attribLogic)
    {
        case POSITION:  return MRayDataTypeRT(MRayDataType<MR_VECTOR_3>());
        case RADIUS:    return MRayDataTypeRT(MRayDataType<MR_FLOAT>());
        default:        throw MRayError("Unknown attribute logic!");
    }
}