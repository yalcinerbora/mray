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
    static const std::array<MatAttributeInfo, 2> LogicList =
    {
        MatAttributeInfo("Albedo", MR_MANDATORY, MR_TEXTURE_OR_CONSTANT,  MRayDataType<MR_VECTOR_3>()),
        MatAttributeInfo("NormalMap", MR_OPTIONAL, MR_TEXTURE_ONLY, MRayDataType<MR_VECTOR_3>())
    };
    return std::vector(LogicList.cbegin(), LogicList.cend());
}
void MatGroupLambert::PushAttribute(MaterialKey id,
                                    uint32_t attributeIndex,
                                    MRayInput data)
{
    //switch(attributeIndex)
    //{
    //    case 0: GenericPushData(id, dAlbedo, std::move(data), true); break;
    //    case 1: GenericPushData(id, dNormalMaps, std::move(data), true); break;
    //    default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
    //                                         TypeName(), attributeIndex));
    //}
    throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
                                TypeName(), attributeIndex));
}

void MatGroupLambert::PushAttribute(MaterialKey id,
                                    const Vector2ui& subRange,
                                    uint32_t attributeIndex,
                                    MRayInput data)
{
    //switch(attributeIndex)
    //{
    //    case 0: GenericPushData(id, subRange, dAlbedo, std::move(data), true); break;
    //    case 1: GenericPushData(id, subRange, dNormalMaps, std::move(data), true); break;
    //    default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
    //                                         TypeName(), attributeIndex));
    //}
    throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
                                TypeName(), attributeIndex));
}

void MatGroupLambert::PushAttribute(const Vector<2, MaterialKey::Type>& idRange,
                                    uint32_t attributeIndex,
                                    MRayInput data)
{
    //switch(attributeIndex)
    //{
    //    case 0: GenericPushData(idRange, dAlbedo, std::move(data), true, true); break;
    //    case 1: GenericPushData(idRange, dNormalMaps, std::move(data), true, true); break;
    //    default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
    //                                         TypeName(), attributeIndex));
    //}
    throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
                                TypeName(), attributeIndex));
}

typename MatGroupLambert::DataSoA MatGroupLambert::SoA() const
{
    return soa;
}