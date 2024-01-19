#include "Transforms.h"

std::string_view TransformGroupIdentity::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(T)Identity"sv;
    return name;
}

TransformGroupIdentity::TransformGroupIdentity(uint32_t groupId,
                                               const GPUSystem& s)
    : BaseType(groupId, s)
{}

void TransformGroupIdentity::Commit()
{
    GenericCommit<>();
}

void TransformGroupIdentity::PushAttribute(Vector2ui, uint32_t,
                                           std::vector<Byte>)
{}

TransformGroupIdentity::AttribInfoList TransformGroupIdentity::AttributeInfo() const
{
    return AttribInfoList();
}

std::string_view TransformGroupSingle::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(T)Identity"sv;
    return name;
}

TransformGroupSingle::TransformGroupSingle(uint32_t groupId,
                     const GPUSystem& s)
    : BaseType(groupId, s)
{}

void TransformGroupSingle::Commit()
{
    auto [t, it] = GenericCommit<Matrix4x4, Matrix4x4>();
    transforms = t;
    invTransforms = it;

    soa.transforms = ToConstSpan(transforms);
    soa.invTransforms = ToConstSpan(invTransforms);

    // Check if the invTransforms are not set

    //??
}

void TransformGroupSingle::PushAttribute(Vector2ui idRange,
                                         uint32_t attributeIndex,
                                         std::vector<Byte> data)
{
    switch(attributeIndex)
    {
        case 0: GenericPushData(idRange, transforms, std::move(data)); break;
        case 1: GenericPushData(idRange, invTransforms, std::move(data)); break;
        default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
                                             TypeName(), attributeIndex));
    }
}

TransformGroupSingle::AttribInfoList TransformGroupSingle::AttributeInfo() const
{
    using enum MRayDataEnum;
    static const std::array<PrimAttributeInfo, 2> LogicList =
    {
        TransAttributeInfo("Transform",     MRayDataType<MR_MATRIX_4x4>()),
        TransAttributeInfo("InvTransform",  MRayDataType<MR_MATRIX_4x4>())
    };
    return std::vector(LogicList.cbegin(), LogicList.cend());
}

std::string_view TransformGroupMulti::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(T)Multi"sv;
    return name;
}

TransformGroupMulti::TransformGroupMulti(uint32_t groupId,
                                         const GPUSystem& s)
    : BaseType(groupId, s)
{}

void TransformGroupMulti::Commit()
{

}

void TransformGroupMulti::PushAttribute(Vector2ui idRange,
                                        uint32_t attributeIndex,
                                        std::vector<Byte> data)
{

}

TransformGroupMulti::AttribInfoList TransformGroupMulti::AttributeInfo() const
{
    return AttribInfoList();
}