#include "MaterialsDefault.h"

std::string_view MatGroupLambert::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(Mt)Lambert"sv;
    return name;
}

void MatGroupLambert::HandleMediums(const MediumPairList& mediumList)
{
}

MatGroupLambert::MatGroupLambert(uint32_t groupId,
                                 const TextureView2DMap& map,
                                 const GPUSystem& s)
    : GenericMaterialGroup(groupId, s, map)
{}

void MatGroupLambert::CommitReservations()
{
    auto [a, nm, mIds] = GenericCommit<ParamVaryingData<2, Vector3>,
                                       Optional<TextureView<2, Vector3>>,
                                       MediumKey>({0, 0, 0});
    dAlbedo = a;
    dNormalMaps = nm;
    dMediumIds = mIds;

    soa.dAlbedo = ToConstSpan(dAlbedo);
    soa.dNormalMaps = ToConstSpan(dNormalMaps);
    soa.dMediumIds = ToConstSpan(dMediumIds);
}

MatAttributeInfoList MatGroupLambert::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeOptionality;
    using enum AttributeTexturable;
    using enum AttributeIsArray;
    using enum AttributeIsColor;
    static const MatAttributeInfoList LogicList =
    {
        MatAttributeInfo("albedo", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                         MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_COLOR),
        MatAttributeInfo("normalMap", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                         MR_OPTIONAL, MR_TEXTURE_ONLY, IS_PURE_DATA)
    };
    return LogicList;
}
void MatGroupLambert::PushAttribute(MaterialKey,
                                    uint32_t attributeIndex,
                                    TransientData,
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
                                    uint32_t attributeIndex,
                                    const Vector2ui&,
                                    TransientData,
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

void MatGroupLambert::PushAttribute(MaterialKey, MaterialKey,
                                    uint32_t attributeIndex,
                                    TransientData,
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


void MatGroupLambert::PushTex2DAttribute(MaterialKey idStart, MaterialKey idEnd,
                                         uint32_t attributeIndex,
                                         TransientData data,
                                         std::vector<Optional<TextureId>> texIds,
                                         const GPUQueue& queue)
{
    if(attributeIndex == 0)
    {
        GenericPushTex2DAttribute<Vector3>(dAlbedo,
                                           //
                                           idStart, idEnd,
                                           attributeIndex,
                                           std::move(data),
                                           std::move(texIds),
                                           queue);
    }
    else MRAY_ERROR_LOG("{:s}: Attribute {:d} is not \"ParamVarying\", wrong "
                        "function is called", TypeName(), attributeIndex);
}

void MatGroupLambert::PushTex2DAttribute(MaterialKey idStart, MaterialKey idEnd,
                                         uint32_t attributeIndex,
                                         std::vector<Optional<TextureId>> texIds,
                                         const GPUQueue& queue)
{
    if(attributeIndex == 1)
    {
        GenericPushTex2DAttribute<Vector3>(dNormalMaps,
                                           //
                                           idStart, idEnd,
                                           attributeIndex,
                                           std::move(texIds),
                                           queue);
    }
    else MRAY_ERROR_LOG("{:s}: Attribute {:d} is not \"Optional Texture\", wrong "
                        "function is called", TypeName(), attributeIndex);
}

void MatGroupLambert::PushTex2DAttribute(MaterialKey, MaterialKey,
                                         uint32_t,
                                         std::vector<TextureId>,
                                         const GPUQueue&)
{
    MRAY_ERROR_LOG("{:s} do not have and mandatory textures!", TypeName());
}

typename MatGroupLambert::DataSoA MatGroupLambert::SoA() const
{
    return soa;
}