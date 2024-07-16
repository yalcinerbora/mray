#include "MaterialsDefault.h"
#include "Core/TypeNameGenerators.h"
#include "Core/Error.hpp"

//===============================//
//       Lambert Material        //
//===============================//
std::string_view MatGroupLambert::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Lambert"sv;
    return MaterialTypeName<Name>;
}

void MatGroupLambert::HandleMediums(const MediumKeyPairList& mediumList)
{
}

MatGroupLambert::MatGroupLambert(uint32_t groupId,
                                 const GPUSystem& s,
                                 const TextureViewMap& map)
    : GenericGroupMaterial<MatGroupLambert>(groupId, s, map)
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
    throw MRayError("{:s}: Attribute {:d} is \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

void MatGroupLambert::PushAttribute(MaterialKey,
                                    uint32_t attributeIndex,
                                    const Vector2ui&,
                                    TransientData,
                                    const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

void MatGroupLambert::PushAttribute(MaterialKey, MaterialKey,
                                    uint32_t attributeIndex,
                                    TransientData,
                                    const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}


void MatGroupLambert::PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                       uint32_t attributeIndex,
                                       TransientData data,
                                       std::vector<Optional<TextureId>> texIds,
                                       const GPUQueue& queue)
{
    if(attributeIndex == 0)
    {
        GenericPushTexAttribute<2, Vector3>(dAlbedo,
                                            //
                                            idStart, idEnd,
                                            attributeIndex,
                                            std::move(data),
                                            std::move(texIds),
                                            queue);
    }
    else throw MRayError("{:s}: Attribute {:d} is not \"ParamVarying\", wrong "
                         "function is called", TypeName(), attributeIndex);
}

void MatGroupLambert::PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                       uint32_t attributeIndex,
                                       std::vector<Optional<TextureId>> texIds,
                                       const GPUQueue& queue)
{
    if(attributeIndex == 1)
    {
        GenericPushTexAttribute<2, Vector3>(dNormalMaps,
                                            //
                                            idStart, idEnd,
                                            attributeIndex,
                                            std::move(texIds),
                                            queue);
    }
    else throw MRayError("{:s}: Attribute {:d} is not \"Optional Texture\", wrong "
                         "function is called", TypeName(), attributeIndex);
}

void MatGroupLambert::PushTexAttribute(MaterialKey, MaterialKey,
                                       uint32_t,
                                       std::vector<TextureId>,
                                       const GPUQueue&)
{
    throw MRayError("{:s} do not have any mandatory textures!", TypeName());
}

typename MatGroupLambert::DataSoA MatGroupLambert::SoA() const
{
    return soa;
}

//===============================//
//       Reflect Material        //
//===============================//
std::string_view MatGroupReflect::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Reflect"sv;
    return MaterialTypeName<Name>;
}

void MatGroupReflect::HandleMediums(const MediumKeyPairList& mediumList)
{
}

MatGroupReflect::MatGroupReflect(uint32_t groupId,
                                 const GPUSystem& s,
                                 const TextureViewMap& map)
    : GenericGroupMaterial<MatGroupReflect>(groupId, s, map)
{}

void MatGroupReflect::CommitReservations()
{
    isCommitted = true;
}

MatAttributeInfoList MatGroupReflect::AttributeInfo() const
{
    return MatAttributeInfoList{};
}

void MatGroupReflect::PushAttribute(MaterialKey,
                                    uint32_t,
                                    TransientData,
                                    const GPUQueue&)
{}

void MatGroupReflect::PushAttribute(MaterialKey,
                                    uint32_t, const Vector2ui&,
                                    TransientData, const GPUQueue&)
{}

void MatGroupReflect::PushAttribute(MaterialKey, MaterialKey,
                                    uint32_t, TransientData,
                                    const GPUQueue&)
{}


void MatGroupReflect::PushTexAttribute(MaterialKey, MaterialKey,
                                       uint32_t, TransientData,
                                       std::vector<Optional<TextureId>>,
                                       const GPUQueue&)
{}

void MatGroupReflect::PushTexAttribute(MaterialKey, MaterialKey,
                                       uint32_t,
                                       std::vector<Optional<TextureId>>,
                                       const GPUQueue&)
{}

void MatGroupReflect::PushTexAttribute(MaterialKey, MaterialKey,
                                       uint32_t,
                                       std::vector<TextureId>,
                                       const GPUQueue&)
{}

typename MatGroupReflect::DataSoA MatGroupReflect::SoA() const
{
    return soa;
}

//===============================//
//       Refract Material        //
//===============================//
std::string_view MatGroupRefract::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Refract"sv;
    return MaterialTypeName<Name>;
}

void MatGroupRefract::HandleMediums(const MediumKeyPairList& mediumList)
{
}

MatGroupRefract::MatGroupRefract(uint32_t groupId,
                                 const GPUSystem& s,
                                 const TextureViewMap& map)
    : GenericGroupMaterial<MatGroupRefract>(groupId, s, map)
{}

void MatGroupRefract::CommitReservations()
{
    isCommitted = true;
}

MatAttributeInfoList MatGroupRefract::AttributeInfo() const
{
    return MatAttributeInfoList{};
}

void MatGroupRefract::PushAttribute(MaterialKey,
                                    uint32_t,
                                    TransientData,
                                    const GPUQueue&)
{}

void MatGroupRefract::PushAttribute(MaterialKey,
                                    uint32_t, const Vector2ui&,
                                    TransientData, const GPUQueue&)
{}

void MatGroupRefract::PushAttribute(MaterialKey, MaterialKey,
                                    uint32_t, TransientData,
                                    const GPUQueue&)
{}


void MatGroupRefract::PushTexAttribute(MaterialKey, MaterialKey,
                                       uint32_t, TransientData,
                                       std::vector<Optional<TextureId>>,
                                       const GPUQueue&)
{}

void MatGroupRefract::PushTexAttribute(MaterialKey, MaterialKey,
                                       uint32_t,
                                       std::vector<Optional<TextureId>>,
                                       const GPUQueue&)
{}

void MatGroupRefract::PushTexAttribute(MaterialKey, MaterialKey,
                                       uint32_t,
                                       std::vector<TextureId>,
                                       const GPUQueue&)
{}

typename MatGroupRefract::DataSoA MatGroupRefract::SoA() const
{
    return soa;
}

//===============================//
//        Unreal Material        //
//===============================//
std::string_view MatGroupUnreal::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Unreal"sv;
    return MaterialTypeName<Name>;
}

void MatGroupUnreal::HandleMediums(const MediumKeyPairList& mediumList)
{
}

MatGroupUnreal::MatGroupUnreal(uint32_t groupId,
                               const GPUSystem& s,
                               const TextureViewMap& map)
    : GenericGroupMaterial<MatGroupUnreal>(groupId, s, map)
{}

void MatGroupUnreal::CommitReservations()
{
    isCommitted = true;
}

MatAttributeInfoList MatGroupUnreal::AttributeInfo() const
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
        MatAttributeInfo("metallic", MRayDataType<MR_FLOAT>(), IS_SCALAR,
                         MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_PURE_DATA),
        MatAttributeInfo("specular", MRayDataType<MR_FLOAT>(), IS_SCALAR,
                         MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_PURE_DATA),
        MatAttributeInfo("roughness", MRayDataType<MR_FLOAT>(), IS_SCALAR,
                         MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_PURE_DATA),
        MatAttributeInfo("normalMap", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                         MR_OPTIONAL, MR_TEXTURE_ONLY, IS_PURE_DATA)
    };
    return LogicList;
}

void MatGroupUnreal::PushAttribute(MaterialKey,
                                   uint32_t attributeIndex,
                                   TransientData,
                                   const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

void MatGroupUnreal::PushAttribute(MaterialKey,
                                   uint32_t attributeIndex,
                                   const Vector2ui&,
                                   TransientData, const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

void MatGroupUnreal::PushAttribute(MaterialKey, MaterialKey,
                                   uint32_t attributeIndex,
                                   TransientData,
                                   const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}


void MatGroupUnreal::PushTexAttribute(MaterialKey, MaterialKey,
                                      uint32_t, TransientData,
                                      std::vector<Optional<TextureId>>,
                                      const GPUQueue&)
{

}

void MatGroupUnreal::PushTexAttribute(MaterialKey, MaterialKey,
                                      uint32_t,
                                      std::vector<Optional<TextureId>>,
                                      const GPUQueue&)
{}

void MatGroupUnreal::PushTexAttribute(MaterialKey, MaterialKey,
                                      uint32_t,
                                      std::vector<TextureId>,
                                      const GPUQueue&)
{}

typename MatGroupUnreal::DataSoA MatGroupUnreal::SoA() const
{
    return soa;
}