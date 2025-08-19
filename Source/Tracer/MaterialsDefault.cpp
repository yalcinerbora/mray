#include "MaterialsDefault.h"
#include "Texture.h"
#include "GenericGroup.hpp"

#include "Core/TypeNameGenerators.h"

#ifdef MRAY_GPU_BACKEND_CPU
    #include "Device/GPUSystem.hpp"
#endif

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

MatGroupLambert::MatGroupLambert(uint32_t groupId,
                                 const GPUSystem& s,
                                 const TextureViewMap& map,
                                 const TextureMap&)
    : GenericGroupMaterial<MatGroupLambert>(groupId, s, map)
{}

void MatGroupLambert::CommitReservations()
{
    GenericCommit(Tie(dAlbedo, dNormalMaps, dMediumKeys),
                  {0, 0, -1});

    soa.dAlbedo = ToConstSpan(dAlbedo);
    soa.dNormalMaps = ToConstSpan(dNormalMaps);
    soa.dMediumKeys = ToConstSpan(dMediumKeys);
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
        MatAttributeInfo("albedo", MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR,
                         MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_COLOR),
        MatAttributeInfo("normalMap", MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR,
                         MR_OPTIONAL, MR_TEXTURE_ONLY, IS_PURE_DATA)
    };
    return LogicList;
}

void MatGroupLambert::PushAttribute(MaterialKey,
                                    uint32_t attributeIndex,
                                    TransientData,
                                    const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

void MatGroupLambert::PushAttribute(MaterialKey,
                                    uint32_t attributeIndex,
                                    const Vector2ui&,
                                    TransientData,
                                    const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

void MatGroupLambert::PushAttribute(MaterialKey, MaterialKey,
                                    uint32_t attributeIndex,
                                    TransientData,
                                    const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"ConstantOnly\", wrong "
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

void MatGroupLambert::Finalize(const GPUQueue& q)
{
    q.MemcpyAsync(dMediumKeys, Span<const MediumKeyPair>(allMediums));
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

MatGroupReflect::MatGroupReflect(uint32_t groupId,
                                 const GPUSystem& s,
                                 const TextureViewMap& map,
                                 const TextureMap&)
    : GenericGroupMaterial<MatGroupReflect>(groupId, s, map)
{}

void MatGroupReflect::CommitReservations()
{
    GenericCommit(Tie(dMediumKeys), {-1});
    soa.dMediumKeys = ToConstSpan(dMediumKeys);
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

void MatGroupReflect::Finalize(const GPUQueue& q)
{
    q.MemcpyAsync(dMediumKeys, Span<const MediumKeyPair>(allMediums));
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

MatGroupRefract::MatGroupRefract(uint32_t groupId,
                                 const GPUSystem& s,
                                 const TextureViewMap& map,
                                 const TextureMap&)
    : GenericGroupMaterial<MatGroupRefract>(groupId, s, map)
{}

void MatGroupRefract::CommitReservations()
{
    GenericCommit(Tie(dMediumKeys,
                      dFrontCauchyCoeffs,
                      dBackCauchyCoeffs), {-1, 0, 0});

    soa.dBackCauchyCoeffs = ToConstSpan(dBackCauchyCoeffs);
    soa.dFrontCauchyCoeffs = ToConstSpan(dFrontCauchyCoeffs);
    soa.dMediumKeys = ToConstSpan(dMediumKeys);
}

MatAttributeInfoList MatGroupRefract::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeOptionality;
    using enum AttributeTexturable;
    using enum AttributeIsArray;
    using enum AttributeIsColor;
    static const MatAttributeInfoList LogicList =
    {
        MatAttributeInfo("cauchyBack", MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR,
                         MR_MANDATORY, MR_CONSTANT_ONLY, IS_PURE_DATA),
        MatAttributeInfo("cauchyFront", MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR,
                         MR_MANDATORY, MR_CONSTANT_ONLY, IS_PURE_DATA)
    };
    return LogicList;
}

void MatGroupRefract::PushAttribute(MaterialKey matKey,
                                    uint32_t attributeIndex,
                                    TransientData data,
                                    const GPUQueue& queue)
{
    auto GenericLoad = [&]<class T>(const Span<T>& d)
    {
        GenericPushData(d, matKey.FetchIndexPortion(),
                        attributeIndex, std::move(data),
                        queue);
    };

    switch(attributeIndex)
    {
        case 0: GenericLoad(dBackCauchyCoeffs); break;
        case 1: GenericLoad(dFrontCauchyCoeffs); break;
        default: throw MRayError("{:s}: Unkown attribute index {:d}",
                                 TypeName(), attributeIndex);
    }
}

void MatGroupRefract::PushAttribute(MaterialKey matKey,
                                    uint32_t attributeIndex,
                                    const Vector2ui& subRange,
                                    TransientData data, const GPUQueue& queue)
{
    auto GenericLoad = [&]<class T>(const Span<T>& d)
    {
        GenericPushData(d, matKey.FetchIndexPortion(),
                        attributeIndex, subRange,
                        std::move(data), queue);
    };

    switch(attributeIndex)
    {
        case 0: GenericLoad(dBackCauchyCoeffs); break;
        case 1: GenericLoad(dFrontCauchyCoeffs); break;
        default: throw MRayError("{:s}: Unkown attribute index {:d}",
                                 TypeName(), attributeIndex);
    }
}

void MatGroupRefract::PushAttribute(MaterialKey idStart, MaterialKey idEnd,
                                    uint32_t attributeIndex, TransientData data,
                                    const GPUQueue& queue)
{
    auto GenericLoad = [&]<class T>(const Span<T>& d)
    {
        using Vec = Vector<2, CommonKey>;
        GenericPushData(d, Vec(idStart.FetchIndexPortion(),
                               idEnd.FetchIndexPortion()),
                        attributeIndex, std::move(data),
                        queue);
    };
    switch(attributeIndex)
    {
        case 0: GenericLoad(dBackCauchyCoeffs); break;
        case 1: GenericLoad(dFrontCauchyCoeffs); break;
        default: throw MRayError("{:s}: Unkown attribute index {:d}",
                                 TypeName(), attributeIndex);
    }
}

void MatGroupRefract::PushTexAttribute(MaterialKey, MaterialKey,
                                       uint32_t attributeIndex, TransientData,
                                       std::vector<Optional<TextureId>>,
                                       const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"ParamVarying\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

void MatGroupRefract::PushTexAttribute(MaterialKey, MaterialKey,
                                       uint32_t attributeIndex,
                                       std::vector<Optional<TextureId>>,
                                       const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"Optional Texture\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

void MatGroupRefract::PushTexAttribute(MaterialKey, MaterialKey,
                                       uint32_t attributeIndex,
                                       std::vector<TextureId>,
                                       const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"Mandatory Texture\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

typename MatGroupRefract::DataSoA MatGroupRefract::SoA() const
{
    return soa;
}

void MatGroupRefract::Finalize(const GPUQueue& q)
{
    q.MemcpyAsync(dMediumKeys, Span<const MediumKeyPair>(allMediums));
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

MatGroupUnreal::MatGroupUnreal(uint32_t groupId,
                               const GPUSystem& s,
                               const TextureViewMap& map,
                               const TextureMap&)
    : GenericGroupMaterial<MatGroupUnreal>(groupId, s, map)
{}

void MatGroupUnreal::CommitReservations()
{
    GenericCommit(Tie(dAlbedo, dNormalMaps,
                      dRoughness, dSpecular, dMetallic,
                      dMediumKeys),
                  {0, 0, 0, 0, 0, -1});

    soa.dAlbedo = ToConstSpan(dAlbedo);
    soa.dNormalMaps = ToConstSpan(dNormalMaps);
    soa.dRoughness = ToConstSpan(dRoughness);
    soa.dSpecular = ToConstSpan(dSpecular);
    soa.dMetallic = ToConstSpan(dMetallic);
    soa.dMediumKeys = ToConstSpan(dMediumKeys);
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
        MatAttributeInfo("albedo", MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR,
                         MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_COLOR),
        MatAttributeInfo("normalMap", MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR,
                         MR_OPTIONAL, MR_TEXTURE_ONLY, IS_PURE_DATA),
        MatAttributeInfo("roughness", MRayDataTypeRT(MR_FLOAT), IS_SCALAR,
                         MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_PURE_DATA),
        MatAttributeInfo("specular", MRayDataTypeRT(MR_FLOAT), IS_SCALAR,
                         MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_PURE_DATA),
        MatAttributeInfo("metallic", MRayDataTypeRT(MR_FLOAT), IS_SCALAR,
                         MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_PURE_DATA)
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


void MatGroupUnreal::PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                      uint32_t attributeIndex, TransientData data,
                                      std::vector<Optional<TextureId>> texIds,
                                      const GPUQueue& queue)
{
    auto GenericLoad = [&]<class T>(Span<ParamVaryingData<2, T>> t)
    {
        GenericPushTexAttribute<2, T>(t, idStart, idEnd,
                                      attributeIndex, std::move(data),
                                      std::move(texIds), queue);
    };

    switch(attributeIndex)
    {
        case ALBEDO_INDEX:      GenericLoad(dAlbedo); break;
        case ROUGHNESS_INDEX:   GenericLoad(dRoughness); break;
        case SPECULAR_INDEX:    GenericLoad(dSpecular); break;
        case METALLIC_INDEX:    GenericLoad(dMetallic); break;
        default: throw MRayError("{:s}: Attribute {:d} is not \"ParamVarying\", wrong "
                                 "function is called", TypeName(), attributeIndex);
    }
}

void MatGroupUnreal::PushTexAttribute(MaterialKey idStart, MaterialKey idEnd,
                                      uint32_t attributeIndex,
                                      std::vector<Optional<TextureId>> texIds,
                                      const GPUQueue& queue)
{
    if(attributeIndex == NORMAL_MAP_INDEX)
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

void MatGroupUnreal::PushTexAttribute(MaterialKey, MaterialKey,
                                      uint32_t,
                                      std::vector<TextureId>,
                                      const GPUQueue&)
{
    throw MRayError("{:s} do not have any mandatory textures!", TypeName());
}

typename MatGroupUnreal::DataSoA MatGroupUnreal::SoA() const
{
    return soa;
}

void MatGroupUnreal::Finalize(const GPUQueue& q)
{
    q.MemcpyAsync(dMediumKeys, Span<const MediumKeyPair>(allMediums));
}