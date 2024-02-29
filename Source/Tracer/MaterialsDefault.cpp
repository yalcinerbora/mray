#include "MaterialsDefault.h"

std::string_view MatGroupLambert::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(M)Lambert"sv;
    return name;
}

MatGroupLambert::MatGroupLambert(uint32_t groupId, const GPUSystem& s)
    : GenericGroupT(groupId, s)
{}

void MatGroupLambert::CommitReservations()
{
    auto [a, nm, mIds] = GenericCommit<ParamVaryingData<2, Spectrum>,
                                 Optional<TextureView<2, Vector3>>,
                                 MediumKey>({true, true, true});
    dAlbedo = a;
    dNormalMaps = nm;
    dMediumIds = mIds;

    soa.dAlbedo = ToConstSpan(dAlbedo);
    soa.dNormalMaps = ToConstSpan(dNormalMaps);
    soa.dMediumIds = ToConstSpan(dMediumIds);
}

typename MatGroupLambert::AttribInfoList MatGroupLambert::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeOptionality;
    using enum AttributeTexturable;
    using enum AttributeIsArray;
    static const std::array<MatAttributeInfo, 2> LogicList =
    {
        MatAttributeInfo("Albedo", MRayDataType<MR_VECTOR_3>(), IS_SCALAR, MR_MANDATORY, MR_TEXTURE_OR_CONSTANT),
        MatAttributeInfo("NormalMap", MRayDataType<MR_VECTOR_3>(), IS_SCALAR, MR_OPTIONAL, MR_TEXTURE_ONLY)
    };
    return std::vector(LogicList.cbegin(), LogicList.cend());
}
void MatGroupLambert::PushAttribute(MaterialKey,
                                    uint32_t attributeIndex,
                                    MRayInput,
                                    const GPUQueue&)
{
    //switch(attributeIndex)
    //{
    //    case 0: GenericPushData(id, dAlbedo, std::move(data), true); break;
    //    case 1: GenericPushData(id, dNormalMaps, std::move(data), true); break;
    //    default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
    //                                         TypeName(), attributeIndex));
    //}
    throw MRayError("{:s}: Unkown AttributeIndex {:d}",
                    TypeName(), attributeIndex);
}

void MatGroupLambert::PushAttribute(MaterialKey,
                                    const Vector2ui&,
                                    uint32_t attributeIndex,
                                    MRayInput,
                                    const GPUQueue&)
{
    //switch(attributeIndex)
    //{
    //    case 0: GenericPushData(id, subRange, dAlbedo, std::move(data), true); break;
    //    case 1: GenericPushData(id, subRange, dNormalMaps, std::move(data), true); break;
    //    default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
    //                                         TypeName(), attributeIndex));
    //}
    throw MRayError("{:s}: Unkown AttributeIndex {:d}",
                    TypeName(), attributeIndex);
}

void MatGroupLambert::PushAttribute(const Vector<2, MaterialKey::Type>&,
                                    uint32_t attributeIndex,
                                    MRayInput,
                                    const GPUQueue&)
{
    //switch(attributeIndex)
    //{
    //    case 0: GenericPushData(idRange, dAlbedo, std::move(data), true, true); break;
    //    case 1: GenericPushData(idRange, dNormalMaps, std::move(data), true, true); break;
    //    default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
    //                                         TypeName(), attributeIndex));
    //}
    throw MRayError("{:s}: Unkown AttributeIndex {:d}",
                    TypeName(), attributeIndex);
}

typename MatGroupLambert::DataSoA MatGroupLambert::SoA() const
{
    return soa;
}