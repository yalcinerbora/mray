#include "SceneLoaderMRay.h"

#include "Core/TracerI.h"
#include "Core/Log.h"
#include "Core/Timer.h"
#include "Core/NormTypes.h"
#include "Core/Filesystem.h"

#include "MeshLoader/EntryPoint.h"
#include "MeshLoaderJson.h"

#include "ImageLoader/EntryPoint.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string_view>
#include <barrier>
#include <atomic>
#include <istream>

#include "JsonNode.h"

struct TexturedAttributeData
{
    TransientData                       data;
    std::vector<Optional<TextureId>>    textures;
};


std::string AddPrimitivePrefix(std::string_view primType)
{
    return (std::string(TracerConstants::PRIM_PREFIX) +
            std::string(primType));
}

std::string AddAddLightPrefix(std::string_view lightType)
{
    return (std::string(TracerConstants::LIGHT_PREFIX) +
            std::string(lightType));
}

std::string AddTransformPrefix(std::string_view transformType)
{
    return (std::string(TracerConstants::TRANSFORM_PREFIX) +
            std::string(transformType));
}

std::string AddMaterialPrefix(std::string_view matType)
{
    return (std::string(TracerConstants::MAT_PREFIX) +
            std::string(matType));
}

std::string AddCameraPrefix(std::string_view camType)
{
    return (std::string(TracerConstants::CAM_PREFIX) +
            std::string(camType));
}

std::string AddMediumPrefix(std::string_view medType)
{
    return (std::string(TracerConstants::MEDIUM_PREFIX) +
            std::string(medType));
}

std::string CreatePrimBackedLightType(std::string_view primType)
{
    using namespace std::literals;
    auto result = ("Primitive"s +
                   std::string(TracerConstants::PRIM_PREFIX) +
                   std::string(primType));
    return result;
}

template<class AttributeInfoList>
AttributeCountList GenericFindAttributeCounts(std::vector<AttributeCountList>& attributeCounts,
                                              const AttributeInfoList& list,
                                              Span<const JsonNode> nodes)
{
    AttributeCountList totalCounts(StaticVecSize(list.size()));

    for(const JsonNode& node : nodes)
    {
        AttributeCountList nodeCountList;
        uint32_t attribIndex = 0;
        for(const auto& l : list)
        {
            using enum TransAttributeInfo::E;
            std::string_view name = std::get<LOGIC_INDEX>(l);
            AttributeOptionality optional = std::get<OPTIONALITY_INDEX>(l);
            AttributeIsArray isArray = std::get<IS_ARRAY_INDEX>(l);

            size_t entityAttribCount = 0;
            if(isArray == AttributeIsArray::IS_ARRAY &&
               optional != AttributeOptionality::MR_MANDATORY)
            {
                // Check optional attribute is present
                entityAttribCount = node.CheckOptionalDataArraySize(name);
            }
            else if(isArray == AttributeIsArray::IS_ARRAY &&
                    optional == AttributeOptionality::MR_MANDATORY)
            {
                entityAttribCount = node.CheckDataArraySize(name);
            }
            else if(isArray != AttributeIsArray::IS_ARRAY &&
                    optional != AttributeOptionality::MR_MANDATORY)
            {
                entityAttribCount = node.CheckOptionalData(name) ? 1 : 0;
            }
            else if(isArray != AttributeIsArray::IS_ARRAY &&
                    optional == AttributeOptionality::MR_MANDATORY)
            {
                entityAttribCount = 1;
            }
            nodeCountList.push_back(entityAttribCount);
            totalCounts[attribIndex] += entityAttribCount;
            attribIndex++;
        }
        attributeCounts.push_back(std::move(nodeCountList));
    }
    return totalCounts;
}

std::vector<TransientData> GenericAttributeLoad(const AttributeCountList& totalCounts,
                                                const GenericAttributeInfoList& list,
                                                Span<const JsonNode> nodes)
{
    std::vector<TransientData> result;
    result.reserve(list.size());

    for(size_t i = 0; i < totalCounts.size(); i++)
    {
        std::visit([&](auto&& dataType)
        {
            using T = std::remove_cvref_t<decltype(dataType)>::Type;
            result.emplace_back(std::in_place_type_t<T>{}, totalCounts[i]);
        },
        std::get<GenericAttributeInfo::LAYOUT_INDEX>(list[i]));
    }

    // Now data is set we can load
    for(const JsonNode& node : nodes)
    {
        uint32_t i = 0;
        for(const auto& l : list)
        {
            using enum GenericAttributeInfo::E;
            std::string_view name = std::get<LOGIC_INDEX>(l);
            AttributeOptionality optional = std::get<OPTIONALITY_INDEX>(l);
            AttributeIsArray isArray = std::get<IS_ARRAY_INDEX>(l);

            std::visit([&](auto&& dataType)
            {
                using T = std::remove_cvref_t<decltype(dataType)>::Type;
                if(isArray == AttributeIsArray::IS_ARRAY &&
                   optional != AttributeOptionality::MR_MANDATORY)
                {
                    Optional<TransientData> data = node.AccessOptionalDataArray<T>(name);
                    if(!data.has_value()) return;
                    result[i].Push(ToSpan<const T>(data.value()));
                }
                else if(isArray == AttributeIsArray::IS_ARRAY &&
                        optional == AttributeOptionality::MR_MANDATORY)
                {
                    TransientData data = node.AccessDataArray<T>(name);
                    result[i].Push(ToSpan<const T>(data));
                }
                else if(isArray != AttributeIsArray::IS_ARRAY &&
                        optional != AttributeOptionality::MR_MANDATORY)
                {
                    Optional<T> data = node.AccessOptionalData<T>(name);
                    if(!data.has_value()) return;
                    result[i].Push(Span<const T>(&data.value(), 1));
                }
                else if(isArray != AttributeIsArray::IS_ARRAY &&
                        optional == AttributeOptionality::MR_MANDATORY)
                {
                    T data = node.AccessData<T>(name);
                    result[i].Push(Span<const T>(&data, 1));
                }
            },
            std::get<LAYOUT_INDEX>(l));
            i++;
        }
    }
    return result;
}


std::vector<TexturedAttributeData> TexturableAttributeLoad(const AttributeCountList& totalCounts,
                                                           const TexturedAttributeInfoList& list,
                                                           Span<const JsonNode> nodes,
                                                           const typename TracerIdPack::TextureIdMappings& texMappings)
{
    std::vector<TexturedAttributeData> result;
    result.reserve(list.size());

    for(size_t i = 0; i < totalCounts.size(); i++)
    {
        std::visit([&](auto&& dataType)
        {
            using T = std::remove_cvref_t<decltype(dataType)>::Type;
            auto initData = TexturedAttributeData
            {
                .data = TransientData(std::in_place_type_t<T>{}, totalCounts[i]),
                .textures = std::vector<Optional<TextureId>>()
            };
            result.push_back(std::move(initData));
        },
        std::get<TexturedAttributeInfo::LAYOUT_INDEX>(list[i]));
    }

    // Now data is set we can load
    for(const JsonNode& node : nodes)
    {
        uint32_t i = 0;
        for(const auto& l : list)
        {
            using enum TexturedAttributeInfo::E;
            std::string_view name = std::get<LOGIC_INDEX>(l);
            AttributeOptionality optional = std::get<OPTIONALITY_INDEX>(l);
            AttributeIsArray isArray = std::get<IS_ARRAY_INDEX>(l);
            AttributeTexturable texturability = std::get<TEXTURABLE_INDEX>(l);

            // Base checks
            if(isArray == AttributeIsArray::IS_ARRAY)
            {
                throw MRayError("Attribute \"{}\" can not be texturable and array "
                                "at the same time! Json: {}",
                                name, nlohmann::to_string(node.RawNode()));
            }
            if(texturability == AttributeTexturable::MR_TEXTURE_OR_CONSTANT &&
               optional == AttributeOptionality::MR_OPTIONAL)
            {
                throw MRayError("Attribute \"{}\" can not be \"texture or constant\" "
                                "*and* \"optional\" at the same time! Json: {}",
                                name, nlohmann::to_string(node.RawNode()));
            }

            // Only texture no type needed
            if(texturability == AttributeTexturable::MR_TEXTURE_ONLY)
            {
                if(optional == AttributeOptionality::MR_MANDATORY)
                {
                    NodeTexStruct texStruct = node.AccessTexture(name);
                    result[i].textures.push_back(texMappings.at(texStruct));
                }
                else if(optional == AttributeOptionality::MR_OPTIONAL)
                {
                    Optional<NodeTexStruct> texStruct = node.AccessOptionalTexture(name);
                    TextureId id = (texStruct.has_value())
                                        ? texMappings.at(texStruct.value())
                                        : TracerConstants::InvalidTexture;
                    result[i].textures.push_back(id);
                }
            }

            // Same as GenericAttributeInfo
            // TODO: Share functionality,  this is copy pase code
            if(texturability == AttributeTexturable::MR_CONSTANT_ONLY)
            {
                std::visit([&](auto&& dataType)
                {
                    using T = std::remove_cvref_t<decltype(dataType)>::Type;
                    if(isArray == AttributeIsArray::IS_ARRAY &&
                       optional == AttributeOptionality::MR_OPTIONAL)
                    {
                        Optional<TransientData> data = node.AccessOptionalDataArray<T>(name);
                        if(!data.has_value()) return;
                        result[i].data.Push(ToSpan<const T>(data.value()));
                    }
                    else if(isArray == AttributeIsArray::IS_ARRAY &&
                            optional == AttributeOptionality::MR_MANDATORY)
                    {
                        TransientData data = node.AccessDataArray<T>(name);
                        result[i].data.Push(ToSpan<const T>(data));
                    }
                    else if(isArray == AttributeIsArray::IS_SCALAR &&
                            optional == AttributeOptionality::MR_OPTIONAL)
                    {
                        Optional<T> data = node.AccessOptionalData<T>(name);
                        if(!data.has_value()) return;
                        result[i].data.Push(Span<const T>(&data.value(), 1));
                    }
                    else if(isArray == AttributeIsArray::IS_SCALAR &&
                            optional == AttributeOptionality::MR_MANDATORY)
                    {
                        T data = node.AccessData<T>(name);
                        result[i].data.Push(Span<const T>(&data, 1));
                    }
                },
                std::get<LAYOUT_INDEX>(l));
            }
            //  Now the hairy part
            if(texturability == AttributeTexturable::MR_TEXTURE_OR_CONSTANT)
            {
                std::visit([&](auto&& dataType)
                {
                    using T = std::remove_cvref_t<decltype(dataType)>::Type;
                    Variant<NodeTexStruct, T> texturable = node.AccessTexturableData<T>(name);
                    if(std::holds_alternative<NodeTexStruct>(texturable))
                    {
                        TextureId id = texMappings.at(std::get<NodeTexStruct>(texturable));
                        result[i].textures.emplace_back(id);
                        T phony;
                        result[i].data.Push(Span<const T>(&phony, 1));
                    }
                    else
                    {
                        result[i].textures.emplace_back(std::nullopt);
                        result[i].data.Push(Span<const T>(&std::get<T>(texturable), 1));
                    }
                },
                std::get<LAYOUT_INDEX>(l));
            }
            i++;
        }
    }
    return result;
}

void LoadPrimitive(TracerI& tracer,
                   PrimGroupId groupId,
                   PrimBatchId batchId,
                   uint32_t meshInternalIndex,
                   const MeshFileI* meshFile)
{
    using enum MRayDataEnum;
    using enum PrimitiveAttributeLogic;
    const auto& attributeList = tracer.AttributeInfo(groupId);
    // Meshes are quite different from other type groups,
    // Types should somewhat make sense since there are many file types
    // (.obj, .fbx, .usd even).
    // MRay enumerates these in "PrimitiveAttributeLogic". I did not want this as
    // string since strings have unbounded logic. This hinders
    // to extend the functionality somewhat but in the end all of the primitives
    // (discrete or not) has some form of tangents, positions, uvs etc.
    // (Probably one exception is the Bezier curve's control points but you can
    // put it as tangent maybe?)
    //
    // This is all fun and games untill this point.
    //
    // Some primitive groups may mandate some attributes, others may not.
    // Some files may have that attribute some may not.
    // In between this matrix of dependencies we can generate these from other attributes
    // (if available). All these things complicates the implementation.
    //
    // On top of all these complications, our default primitive requires normals as
    // quaternions for space efficiency (to tangent space transformation is held as a quat)
    //
    // The solution here is to rely on assimp's tangent/bitangent generation capabilities
    // and use it. On the other hand, for in json triangle primitives compute it on the
    // class.
    for(uint32_t attribIndex = 0; attribIndex < attributeList.size();
        attribIndex++)
    {
        const auto& attribute = attributeList[attribIndex];
        PrimitiveAttributeLogic attribLogic = std::get<PrimAttributeInfo::LOGIC_INDEX>(attribute);
        AttributeOptionality optionality = std::get<PrimAttributeInfo::OPTIONALITY_INDEX>(attribute);
        MRayDataTypeRT groupsLayout = std::get<PrimAttributeInfo::LAYOUT_INDEX>(attribute);
        MRayDataTypeRT filesLayout = meshFile->AttributeLayout(attribLogic);

        // Is this data available?
        if(!meshFile->HasAttribute(attribLogic) &&
           optionality == AttributeOptionality::MR_MANDATORY)
        {
            throw MRayError("Mesh File{:s}:[{:d}] do not have \"{}\" "
                            "which is mandatory for {}",
                            meshFile->Name(), meshInternalIndex,
                            PrimAttributeStringifier::ToString(attribLogic),
                            tracer.TypeName(groupId));
        }
        // Data is available...
        // "Normal" attribute's special case, if mesh has tangents
        // Convert normal/tangent to quaternion, store it as normal
        // normals are defined as to tangent space transformations
        // (shading tangent space that is)
        if(attribLogic == NORMAL && groupsLayout.Name() == MR_QUATERNION &&
           meshFile->HasAttribute(TANGENT) && meshFile->HasAttribute(BITANGENT) &&
           meshFile->AttributeLayout(TANGENT).Name() == MR_VECTOR_3 &&
           meshFile->AttributeLayout(BITANGENT).Name() == MR_VECTOR_3 &&
           meshFile->AttributeLayout(NORMAL).Name() == MR_VECTOR_3)
        {
            size_t normalCount = meshFile->MeshAttributeCount();
            TransientData quats(std::in_place_type_t<Quaternion>{}, normalCount);
            // Utilize TBN matrix directly
            TransientData t = meshFile->GetAttribute(TANGENT);
            TransientData b = meshFile->GetAttribute(BITANGENT);
            TransientData n = meshFile->GetAttribute(attribLogic);

            Span<const Vector3> tangents = t.AccessAs<const Vector3>();
            Span<const Vector3> bitangents = b.AccessAs<const Vector3>();
            Span<const Vector3> normals = n.AccessAs<const Vector3>();

            for(size_t i = 0; i < normalCount; i++)
            {
                Quaternion q = TransformGen::ToSpaceQuat(tangents[i],
                                                         bitangents[i],
                                                         normals[i]);
                quats.Push(Span<const Quaternion>(&q, 1));
            }
            tracer.PushPrimAttribute(groupId, batchId, attribIndex,
                                     std::move(quats));
        }
        // All Good, load and send
        else if(groupsLayout.Name() == filesLayout.Name())
        {
            tracer.PushPrimAttribute(groupId, batchId, attribIndex,
                                     meshFile->GetAttribute(attribLogic));

        }
        // Data's layout does not match with the primitive group
        else
        {
            // We require exact match currently
            throw MRayError("Mesh File {:s}:[{:d}]'s data layout of \"{}\" "
                            "(has type {:s}) does not match the {}'s data layout "
                            "(which is {:s})",
                            meshFile->Name(), meshInternalIndex,
                            PrimAttributeStringifier::ToString(attribLogic),
                            MRayDataTypeStringifier::ToString(filesLayout.Name()),
                            tracer.TypeName(groupId),
                            MRayDataTypeStringifier::ToString(groupsLayout.Name()));
        }
    }
}

std::vector<TransientData> TransformAttributeLoad(const AttributeCountList& totalCounts,
                                                  const GenericAttributeInfoList& list,
                                                  Span<const JsonNode> nodes)
{
    using namespace std::literals;
    static constexpr auto SINGLE_TRANSFORM_TYPE = "Single"sv;
    static constexpr auto MULTI_TRANSFORM_TYPE = "Multi"sv;
    static constexpr auto LAYOUT = "layout"sv;
    static constexpr auto LAYOUT_TRS = "trs"sv;
    static constexpr auto LAYOUT_MATRIX = "matrix"sv;

    // TODO: Change this as well, transform logic should not be in loader
    // I think we need to layer these kind of things in an intermediate
    // system that sits between loader and tracer. (Which should be on GPU as well
    // or it will be slow)
    std::string_view type = nodes[0].CommonData<std::string_view>(NodeNames::TYPE);
    if(type != SINGLE_TRANSFORM_TYPE && type != MULTI_TRANSFORM_TYPE)
        return GenericAttributeLoad(totalCounts, list, nodes);

    //
    assert(list.size() == 1);
    assert(totalCounts.size() == 1);
    std::vector<TransientData> result;
    result.push_back(TransientData(std::in_place_type_t<Matrix4x4>{},
                                   totalCounts.front()));

    const GenericAttributeInfo& info = list.front();
    for(const auto& n : nodes)
    {
        bool isArray = (std::get<GenericAttributeInfo::IS_ARRAY_INDEX>(info) ==
                        AttributeIsArray::IS_ARRAY);

        std::string_view layout = nodes[0].CommonData<std::string_view>(LAYOUT);
        if(layout == LAYOUT_TRS)
        {
            static constexpr auto TRANSLATE = "translate"sv;
            static constexpr auto ROTATE = "rotate"sv;
            static constexpr auto SCALE = "scale"sv;

            auto GenTransformFromTRS = [](const Vector3& t,
                                          const Vector3& r,
                                          const Vector3& s)
            {
                Matrix4x4 transform = TransformGen::Scale(s[0], s[1], s[2]);
                transform = TransformGen::Rotate(r[0], Vector3::XAxis()) * transform;
                transform = TransformGen::Rotate(r[1], Vector3::YAxis()) * transform;
                transform = TransformGen::Rotate(r[2], Vector3::ZAxis()) * transform;
                transform = TransformGen::Translate(t) * transform;
                return transform;
            };

            if(isArray)
            {
                using OptionalVec3List = std::vector<Optional<Vector3>>;

                Optional<TransientData> tL = n.AccessOptionalDataArray<Vector3>(TRANSLATE);
                Optional<TransientData> rL = n.AccessOptionalDataArray<Vector3>(ROTATE);
                Optional<TransientData> sL = n.AccessOptionalDataArray<Vector3>(SCALE);

                Span<const Vector3> tSpan = (tL.has_value())
                                                ? tL.value().AccessAs<Vector3>()
                                                : Span<const Vector3>();
                Span<const Vector3> rSpan = (rL.has_value())
                                                ? rL.value().AccessAs<Vector3>()
                                                : Span<const Vector3>();
                Span<const Vector3> sSpan = (sL.has_value())
                                                ? sL.value().AccessAs<Vector3>()
                                                : Span<const Vector3>();

                for(size_t i = 0; i < tSpan.size(); i++)
                {
                    Vector3 t = (tSpan.empty()) ? tSpan[i] : Vector3::Zero();
                    Vector3 r = (rSpan.empty()) ? rSpan[i] : Vector3::Zero();
                    Vector3 s = (sSpan.empty()) ? sSpan[i] : Vector3(1);

                    Matrix4x4 transform = GenTransformFromTRS(t, r, s);
                    result[0].Push(Span<const Matrix4x4>(&transform, 1));
                }
            }
            else
            {
                Vector3 t = n.AccessOptionalData<Vector3>(TRANSLATE).value_or(Vector3::Zero());
                Vector3 r = n.AccessOptionalData<Vector3>(ROTATE).value_or(Vector3::Zero());
                Vector3 s = n.AccessOptionalData<Vector3>(SCALE).value_or(Vector3(1));

                Matrix4x4 transform = GenTransformFromTRS(t, r, s);
                result[0].Push(Span<const Matrix4x4>(&transform, 1));
            }

        }
        else if(layout == LAYOUT_MATRIX)
        {
            static constexpr auto MATRIX = "matrix"sv;

            if(isArray)
            {
                TransientData t = n.AccessDataArray<Matrix4x4>(MATRIX);
                result[0].Push(t.AccessAs<const Matrix4x4>());
            }
            else
            {
                Matrix4x4 t = n.AccessData<Matrix4x4>(MATRIX);
                result[0].Push(Span<const Matrix4x4>(&t, 1));
            }
        }
        else
        {
            throw MRayError("Unkown transform layout");
        }
    }

    return result;
}

void SceneLoaderMRay::ExceptionList::AddException(MRayError&& err)
{
    size_t location = size.fetch_add(1);
    // If too many exceptions skip it
    if(location < MaxExceptionSize)
        exceptions[location] = std::move(err);
}


LightSurfaceStruct SceneLoaderMRay::LoadBoundary(const nlohmann::json& n)
{
    LightSurfaceStruct boundary = n.get<LightSurfaceStruct>();
    if(boundary.lightId == std::numeric_limits<uint32_t>::max())
        throw MRayError("Boundary light must be set!");
    if(boundary.mediumId == std::numeric_limits<uint32_t>::max())
        throw MRayError("Boundary medium must be set!");
    if(boundary.transformId)
        throw MRayError("Boundary transform must be set!");
    return boundary;
}

std::vector<SurfaceStruct> SceneLoaderMRay::LoadSurfaces(const nlohmann::json& nArray)
{
    std::vector<SurfaceStruct> result;
    for(const auto& n : nArray)
    {
        result.push_back(n.get<SurfaceStruct>());
    }
    return result;
}

std::vector<CameraSurfaceStruct> SceneLoaderMRay::LoadCamSurfaces(const nlohmann::json& nArray,
                                                                  uint32_t boundaryMediumId)
{

    std::vector<CameraSurfaceStruct> result;
    for(const auto& n : nArray)
    {
        result.push_back(n.get<CameraSurfaceStruct>());
        if(result.back().mediumId == std::numeric_limits<uint32_t>::max())
            result.back().mediumId = boundaryMediumId;
    }
    return result;
}

std::vector<LightSurfaceStruct> SceneLoaderMRay::LoadLightSurfaces(const nlohmann::json& nArray,
                                                                   uint32_t boundaryMediumId)
{
    std::vector<LightSurfaceStruct> result;
    for(const auto& n : nArray)
    {
        result.push_back(n.get<LightSurfaceStruct>());
        if(result.back().mediumId == std::numeric_limits<uint32_t>::max())
            result.back().mediumId = boundaryMediumId;
    }
    return result;
}

void SceneLoaderMRay::DryRunLightsForPrim(std::vector<uint32_t>& primIds,
                                          const TypeMappedNodes& lightNodes,
                                          const TracerI& tracer)
{
    for(const auto& l : lightNodes)
    {
        std::string annotatedType = AddAddLightPrefix(l.first);
        LightAttributeInfoList lightAttributes = tracer.AttributeInfoLight(annotatedType);
        // We already annotated primitive-backed light names with "(P)..."
        // suffix, check if the first part is "Primitive"
        if(l.first.find(NodeNames::LIGHT_TYPE_PRIMITIVE) == std::string::npos)
            continue;
        // Light Type is primitive, it has to have "primitive" field
        for(const auto& node : l.second)
        {
            uint32_t primId = node.AccessData<uint32_t>(NodeNames::PRIMITIVE);
            primIds.push_back(primId);
        }
    }
}

template <class TracerInterfaceFunc, class AnnotateFunc>
void SceneLoaderMRay::DryRunNodesForTex(std::vector<NodeTexStruct>& textureIds,
                                        const TypeMappedNodes& nodes,
                                        const TracerI& tracer,
                                        AnnotateFunc&& Annotate,
                                        TracerInterfaceFunc&& AcquireAttributeInfo)
{
    for(const auto& n : nodes)
    {
        TexturedAttributeInfoList texAttributes = std::invoke(AcquireAttributeInfo,
                                                              tracer, Annotate(n.first));
        for(const auto& node : n.second)
        for(const auto& att : texAttributes)
        {
            AttributeTexturable texturable = std::get<MatAttributeInfo::TEXTURABLE_INDEX>(att);
            AttributeOptionality optional = std::get<MatAttributeInfo::OPTIONALITY_INDEX>(att);
            std::string_view name = std::get<MatAttributeInfo::LOGIC_INDEX>(att);
            if(texturable == AttributeTexturable::MR_CONSTANT_ONLY)
                continue;

            if(texturable == AttributeTexturable::MR_TEXTURE_ONLY)
            {
                if(optional == AttributeOptionality::MR_OPTIONAL)
                {
                    auto ts = node.AccessOptionalData<NodeTexStruct>(name);
                    if(ts.has_value()) textureIds.push_back(ts.value());
                }
                else
                {
                    auto ts = node.AccessData<NodeTexStruct>(name);
                    textureIds.push_back(ts);
                }
            }
            else if(texturable == AttributeTexturable::MR_TEXTURE_OR_CONSTANT)
            {
                MRayDataTypeRT dataType = std::get<MatAttributeInfo::LAYOUT_INDEX>(att);
                std::visit([&node, name, &textureIds](auto&& dataType)
                {
                    using T = std::remove_cvref_t<decltype(dataType)>::Type;
                    auto value = node.AccessTexturableData<T>(name);
                    if(std::holds_alternative<NodeTexStruct>(value))
                        textureIds.push_back(std::get<NodeTexStruct>(value));
                }, dataType);
            }
        }
    }
}

template<class Loader, class GroupIdType, class IdType>
void GenericLoadGroups(typename SceneLoaderMRay::MutexedMap<std::map<uint32_t, Pair<GroupIdType, IdType>>>& outputMappings,
                       typename SceneLoaderMRay::ExceptionList& exceptions,
                       const typename SceneLoaderMRay::TypeMappedNodes& nodeLists,
                       BS::thread_pool& threadPool,
                       Loader&& loader)
{
    using KeyValuePair  = Pair<uint32_t, Pair<GroupIdType, IdType>>;
    using PerGroupList  = std::vector<KeyValuePair>;
    using IdList        = std::vector<IdType>;

    for(const auto& [typeName, nodes] : nodeLists)
    {
        const uint32_t groupEntityCount = static_cast<uint32_t>(nodes.size());
        auto groupEntityList = std::make_shared<PerGroupList>(groupEntityCount);

        GroupIdType groupId = loader.CreateGroup(typeName);

        // Construct Barrier
        auto BarrierFunc = [groupId, loader]() noexcept
        {
            // Explicitly copy the loader
            // Doing this because lambda capture trick
            // [loader = loader] did not work (maybe MSVC bug?)
            auto loaderIn = loader;

            // When barrier completed
            // Reserve the space for mappings
            // Commit group reservations
            loaderIn.CommitReservations(groupId);
        };
        // Determine the thread size
        uint32_t threadCount = std::min(threadPool.get_thread_count(),
                                        groupEntityCount);

        using Barrier = std::barrier<decltype(BarrierFunc)>;
        auto barrier = std::make_shared<Barrier>(threadCount, BarrierFunc);

        // BS_threadpool submit_blocks (all every submit variant I think, passes
        // the lambda by forwarding reference, however in our case we utilize the
        // lambda's scope as a thread local storage, forwarding does not work in this case
        // shared_ptrs are deleted when the 'first' thread exits the scope.
        //
        // So here we force the lambda to be 'const' thus compiler has to forward by copy
        // and achieve expected behaviour.
        // Copy the shared pointers, capture by reference the rest
        const auto LoadTask = [&, loader, barrier = barrier,
                               groupEntityList](size_t start, size_t end)
        {
            // Explicitly copy the loader
            // Doing this because lambda capture trick
            // [loader = loader] did not work (maybe MSVC bug?)
            auto loaderIn = loader;

            bool barrierPassed = false;
            size_t localCount = end - start;
            auto nodeRange = Span<const JsonNode>(nodes.cbegin() + start, localCount);
            IdList generatedIds;
            try
            {
                generatedIds = loaderIn.THRDReserveEntities(groupId, nodeRange);
                for(size_t i = start; i < end; i++)
                {
                    size_t localI = i - start;
                    auto& groupList = *groupEntityList;
                    Pair<GroupIdType, IdType> value(groupId,
                                                    generatedIds[localI]);
                    groupList[i] = std::make_pair(nodes[i].Id(), value);
                }

                // Commit barrier
                barrier->arrive_and_wait();
                barrierPassed = true;
                // Group is committed, now we can issue writes
                loaderIn.THRDLoadEntities(groupId, generatedIds, nodeRange);
            }
            catch(MRayError& e)
            {
                exceptions.AddException(std::move(e));

                if(!barrierPassed) barrier->arrive_and_drop();
            }
            catch(nlohmann::json::exception& e)
            {
                exceptions.AddException(MRayError("Json Error ({})",
                                                  std::string(e.what())));

                if(!barrierPassed) barrier->arrive_and_drop();
            }
            catch(std::exception& e)
            {
                exceptions.AddException(MRayError("Unknown Error ({})",
                                                  std::string(e.what())));

                if(!barrierPassed) barrier->arrive_and_drop();
            }
        };
        auto future = threadPool.submit_blocks(std::size_t(0),
                                               std::size_t(groupEntityCount),
                                               LoadTask, threadCount);

        future.wait();

        // Move future to shared_ptr
        using FutureSharedPtr = std::shared_ptr<BS::multi_future<void>>;
        FutureSharedPtr futureShared = std::make_shared<BS::multi_future<void>>(std::move(future));

        // Issue a one final task that pushes the primitives to the global map
        threadPool.detach_task([&, future = futureShared, groupEntityList]()
        {
            // Wait other tasks to complere
            future->wait();
            // After this point groupBatchList is fully loaded
            std::scoped_lock lock(outputMappings.mutex);
            outputMappings.map.insert(groupEntityList->cbegin(), groupEntityList->cend());
        });
    }
}

void SceneLoaderMRay::LoadTextures(TracerI& tracer, ExceptionList& exceptions)
{
    using TextureIdList = std::vector<std::pair<NodeTexStruct, TextureId>>;

    // Construct Image Loader
    std::shared_ptr<ImageLoaderI> imgLoader = CreateImageLoader();
    auto texIdListPtr = std::make_shared<TextureIdList>(textureNodes.size());

    // Issue loads to the thread pool
    auto BarrierFunc = [&tracer]() noexcept
    {
        // When barrier completed
        // Reserve the space for mappings
        // Commit textures greoups reservations
        tracer.CommitTextures();
    };

    // Determine the thread size
    uint32_t threadCount = std::min(threadPool.get_thread_count(),
                                    static_cast<uint32_t>(textureNodes.size()));

    using Barrier = std::barrier<decltype(BarrierFunc)>;
    auto barrier = std::make_shared<Barrier>(threadCount, BarrierFunc);

    // Same situation discribed in 'GenericLoad' function,
    // force pass by copy.
    // Copy the shared pointers, capture by reference the rest
    const auto TextureLoadTask = [&, texIdListPtr, imgLoader, barrier](size_t start, size_t end)
    {
        // TODO: check if we twice opening is bottleneck?
        // We are opening here to determining size/format
        // and on the other iteration we actual memcpy it
        bool barrierPassed = false;
        try
        {
            for(size_t i = start; i < end; i++)
            {
                const auto& [texStruct, jsonNode, is2D] = textureNodes[i];
                auto fileName = jsonNode.AccessData<std::string>(NodeNames::TEX_NODE_FILE);
                auto isColor = jsonNode.AccessOptionalData<bool>(NodeNames::TEX_NODE_IS_COLOR)
                                .value_or(NodeNames::TEX_NODE_IS_COLOR_DEFAULT);
                fileName = Filesystem::RelativePathToAbsolute(fileName, scenePath);

                // Currently no flags are utilized on header load time
                // TODO: Check here if this fails
                TextureId tId;
                if(is2D)
                {
                    Expected<ImageHeader<2>> headerE = imgLoader->ReadImageHeader2D(fileName);
                    if(!headerE.has_value())
                    {
                        exceptions.AddException(std::move(headerE.error()));
                        barrier->arrive_and_drop();
                        return;
                    }

                    // Always expand to 4 channel due to HW limitation
                    headerE.value().pixelType = ImageLoaderI::TryExpandTo4CFormat(headerE.value().pixelType);

                    const auto& header = headerE.value();
                    tId = tracer.CreateTexture2D(header.dimensions,
                                                 header.mipCount,
                                                 header.pixelType.Name(),
                                                 isColor
                                                    ? AttributeIsColor::IS_COLOR
                                                    : AttributeIsColor::IS_PURE_DATA);
                }
                else
                {
                    exceptions.AddException(MRayError("3D Textures are not supported atm."));
                    barrier->arrive_and_drop();
                    return;
                }

                auto& texIdList = *texIdListPtr;
                texIdList[i] = std::make_pair(texStruct, tId);
            }

            // Barrier code is invoked, and all textures are allocated
            barrier->arrive_and_wait();
            barrierPassed = true;

            for(size_t i = start; i < end; i++)
            {
                const auto& [texStruct, jsonNode, is2D] = textureNodes[i];
                bool loadAsSigned = jsonNode.AccessOptionalData<bool>(NodeNames::TEX_NODE_AS_SIGNED)
                                        .value_or(NodeNames::TEX_NODE_AS_SIGNED_DEFAULT);
                auto isColor = jsonNode.AccessOptionalData<bool>(NodeNames::TEX_NODE_IS_COLOR)
                                        .value_or(NodeNames::TEX_NODE_IS_COLOR_DEFAULT);
                auto fileName = jsonNode.AccessData<std::string>(NodeNames::TEX_NODE_FILE);
                fileName = Filesystem::RelativePathToAbsolute(fileName, scenePath);

                using enum ImageIOFlags::F;
                ImageIOFlags flags;
                flags[DISREGARD_COLOR_SPACE] = !isColor;
                flags[LOAD_AS_SIGNED] = loadAsSigned;
                flags[TRY_3C_4C_CONVERSION] = true;     // Always do channel expand (HW limitation)

                if(is2D)
                {
                    Expected<Image<2>> imgE = imgLoader->ReadImage2D(fileName, flags);
                    if(!imgE.has_value())
                    {
                        exceptions.AddException(std::move(imgE.error()));
                        return;
                    }

                    auto& img = imgE.value();
                    // Send data mip by mip
                    for(uint32_t j = 0; j < img.header.mipCount; j++)
                    {
                        auto& texIdList = *texIdListPtr;
                        tracer.PushTextureData(texIdList[i].second, j,
                                               std::move(img.imgData[j].pixels));
                    }
                }
                else
                {
                    exceptions.AddException(MRayError("3D Textures are not supported atm."));
                    barrier->arrive_and_drop();
                    return;
                }
            }
        }
        catch(MRayError& e)
        {
            exceptions.AddException(std::move(e));
            if(!barrierPassed) barrier->arrive_and_drop();
        }
        catch(nlohmann::json::exception& e)
        {
            exceptions.AddException(MRayError("Json Error ({})",
                                              std::string(e.what())));

            if(!barrierPassed) barrier->arrive_and_drop();
        }
        catch(std::exception& e)
        {
            exceptions.AddException(MRayError("Unknown Error ({})",
                                              std::string(e.what())));

            if(!barrierPassed) barrier->arrive_and_drop();
        }
    };

    auto future = threadPool.submit_blocks(std::size_t(0), textureNodes.size(),
                                           TextureLoadTask, threadCount);

    // Move the future to shared ptr
    using FutureSharedPtr = std::shared_ptr<BS::multi_future<void>>;
    FutureSharedPtr futureShared = std::make_shared<BS::multi_future<void>>(std::move(future));

    // Issue a one final task that pushes the primitives to the global map
    threadPool.detach_task([&, this, future = futureShared, texIdListPtr]()
    {
        // Wait other tasks to complere
        future->wait();
        // Thread Generated Textures are finalized
        for(const auto& pair : (*texIdListPtr))
            texMappings.emplace(pair.first, pair.second);
    });
    // All important data is in shared_ptrs we can safely exit scope.
}

void SceneLoaderMRay::LoadMediums(TracerI& tracer, ExceptionList& exceptions)
{
    struct MediumLoader
    {
        private:
        TracerI&                    tracer;
        const TextureIdMappings&    texMappings;
        AttributeCountList          totalCounts;

        public:
        MediumLoader(TracerI& t, const TextureIdMappings& texMappings)
            : tracer(t)
            , texMappings(texMappings)
        {}

        MediumGroupId CreateGroup(std::string gn)
        {
            return tracer.CreateMediumGroup(AddMediumPrefix(gn));
        }

        void CommitReservations(MediumGroupId groupId)
        {
            return tracer.CommitMediumReservations(groupId);
        }

        MediumIdList THRDReserveEntities(MediumGroupId groupId,
                                         Span<const JsonNode> nodes)
        {
            std::vector<AttributeCountList> attributeCounts = {};
            attributeCounts.reserve(nodes.size());
            // Find the first arrayed type in nodes
            // Accumulate the type
            // Reserve transforms accordingly
            MediumAttributeInfoList list = tracer.AttributeInfo(groupId);
            totalCounts = GenericFindAttributeCounts(attributeCounts, list, nodes);
            return tracer.ReserveMediums(groupId, std::move(attributeCounts));
        }

        void THRDLoadEntities(MediumGroupId groupId,
                              const MediumIdList& ids,
                              Span<const JsonNode> nodes)
        {
            MediumAttributeInfoList list = tracer.AttributeInfo(groupId);
            auto dataOut = TexturableAttributeLoad(totalCounts, list, nodes, texMappings);
            uint32_t attribIndex = 0;
            for(auto& data : dataOut)
            {
                MediumId idStart = ids.front();
                MediumId idEnd = ids.front();
                auto range = Vector2ui(static_cast<uint32_t>(idStart),
                                       static_cast<uint32_t>(idEnd));
                tracer.PushMediumAttribute(groupId, range, attribIndex,
                                           std::move(data.data),
                                           std::move(data.textures));
                attribIndex++;
            }
        }
    };
    GenericLoadGroups(mediumMappings, exceptions,
                      mediumNodes, threadPool,
                      MediumLoader(tracer, texMappings));
}

void SceneLoaderMRay::LoadMaterials(TracerI& tracer,
                                    ExceptionList& exceptions,
                                    uint32_t boundaryMediumId)
{
    struct MaterialLoader
    {
        private:
        TracerI&                    tracer;
        uint32_t                    boundaryMediumId;
        const TextureIdMappings&    texMappings;
        const MediumIdMappings&     mediumMappings;
        AttributeCountList          totalCounts;

        public:
        MaterialLoader(TracerI& t, uint32_t boundaryMediumId,
                       const TextureIdMappings& texMappings,
                       const MediumIdMappings& mediumMappings)
            : tracer(t)
            , boundaryMediumId(boundaryMediumId)
            , texMappings(texMappings)
            , mediumMappings(mediumMappings)
        {}

        MatGroupId CreateGroup(std::string gn)
        {
            return tracer.CreateMaterialGroup(AddMaterialPrefix(gn));
        }

        void CommitReservations(MatGroupId groupId)
        {
            return tracer.CommitMatReservations(groupId);
        }

        MaterialIdList THRDReserveEntities(MatGroupId groupId,
                                           Span<const JsonNode> nodes)
        {
            std::vector<AttributeCountList> attributeCounts = {};
            attributeCounts.reserve(nodes.size());

            // Check mediums
            std::vector<MediumPair> ioMediums;
            ioMediums.reserve(nodes.size());
            for(const JsonNode& node : nodes)
            {
                using namespace NodeNames;
                uint32_t medInId = node.AccessOptionalData<uint32_t>(MEDIUM_FRONT).value_or(boundaryMediumId);
                uint32_t medOutId = node.AccessOptionalData<uint32_t>(MEDIUM_BACK).value_or(boundaryMediumId);
                ioMediums.emplace_back(mediumMappings.at(medInId).second,
                                       mediumMappings.at(medOutId).second);
            }

            // Find the first arrayed type in nodes
            // Accumulate the type
            // Reserve transforms accordingly
            MatAttributeInfoList list = tracer.AttributeInfo(groupId);
            totalCounts = GenericFindAttributeCounts(attributeCounts, list, nodes);
            return tracer.ReserveMaterials(groupId, std::move(attributeCounts), ioMediums);
        }

        void THRDLoadEntities(MatGroupId groupId,
                              const MaterialIdList& ids,
                              Span<const JsonNode> nodes)
        {
            MatAttributeInfoList list = tracer.AttributeInfo(groupId);
            auto dataOut = TexturableAttributeLoad(totalCounts, list, nodes, texMappings);
            uint32_t attribIndex = 0;
            for(auto& data : dataOut)
            {
                MaterialId idStart = ids.front();
                MaterialId idEnd = ids.front();
                auto range = Vector2ui(static_cast<uint32_t>(idStart),
                                       static_cast<uint32_t>(idEnd));
                tracer.PushMatAttribute(groupId, range, attribIndex,
                                        std::move(data.data),
                                        std::move(data.textures));
                attribIndex++;
            }
        }
    };

    GenericLoadGroups(matMappings, exceptions,
                      materialNodes, threadPool,
                      MaterialLoader(tracer, boundaryMediumId,
                                     texMappings, mediumMappings.map));
}

void SceneLoaderMRay::LoadTransforms(TracerI& tracer, ExceptionList& exceptions)
{
    struct TransformLoader
    {
        private:
        TracerI&            tracer;
        AttributeCountList  totalCounts;

        public:
        TransformLoader(TracerI& t) : tracer(t) {}

        TransGroupId CreateGroup(std::string gn)
        {
            return tracer.CreateTransformGroup(AddTransformPrefix(gn));
        }

        void CommitReservations(TransGroupId groupId)
        {
            return tracer.CommitTransReservations(groupId);
        }

        TransformIdList THRDReserveEntities(TransGroupId groupId,
                                            Span<const JsonNode> nodes)
        {
            std::vector<AttributeCountList> attributeCounts = {};
            attributeCounts.reserve(nodes.size());
            // Find the first arrayed type in nodes
            // Accumulate the type
            // Reserve transforms accordingly
            TransAttributeInfoList list = tracer.AttributeInfo(groupId);
            totalCounts = GenericFindAttributeCounts(attributeCounts, list, nodes);
            return tracer.ReserveTransformations(groupId, std::move(attributeCounts));
        }

        void THRDLoadEntities(TransGroupId groupId,
                              const TransformIdList& ids,
                              Span<const JsonNode> nodes)
        {
            TransAttributeInfoList list = tracer.AttributeInfo(groupId);
            auto dataOut = TransformAttributeLoad(totalCounts,
                                                  list,
                                                  nodes);
            uint32_t attribIndex = 0;
            for(auto& data : dataOut)
            {
                TransformId idStart = ids.front();
                TransformId idEnd = ids.front();
                auto range = Vector2ui(static_cast<uint32_t>(idStart),
                                       static_cast<uint32_t>(idEnd));
                tracer.PushTransAttribute(groupId, range, attribIndex,
                                          std::move(data));
                attribIndex++;
            }
        }
    };

    GenericLoadGroups(transformMappings, exceptions,
                      transformNodes, threadPool,
                      TransformLoader(tracer));
}

void SceneLoaderMRay::LoadPrimitives(TracerI& tracer, ExceptionList& exceptions)
{
    std::shared_ptr<const MeshLoaderPoolI> meshLoaderPool = CreateMeshLoaderPool();

    struct PrimitiveLoader
    {
        private:
        TracerI&                                tracer;
        const std::string&                      scenePath;
        std::shared_ptr<const MeshLoaderPoolI>  meshLoaderPool;
        // TODO: Too many micro allocations due to map
        // revise over this.
        // TODO: 'loaders' and 'meshFiles' were unique_ptr but we need to
        // empty copy these and these are non-copyable. Converted these to
        // shared ptr. Revise this later
        //
        // Per "dll" loaders
        // Key is a "tag" entry on the json. For example, "assimp" tag
        // will call assimp mesh loader. If "assimp" does not support .obj for example
        // loader will generate an error.
        //
        // The reason for this approach is to provide a utility for user to add new dlls
        // for specific file extensions via per mesh file basis.
        // If assimp "sucks" for new fbx files, however robustly works on old fbx files
        // user can write new loader with a tag and load these files.
        std::map<std::string, std::shared_ptr<MeshLoaderI>> loaders;
        // Mesh files that opened via some loader
        // This is here unfortunately as a limitation/functionality of the assimp.
        // Assimp can post process meshes crates tangents/optimization etc. This means
        // we cannot pre-determine the size of the vertices/indices just by looking a some
        // form of header on the file. So we load the entire mesh and store before the commit
        // of the primitive group.
        //
        // Key is the full path of the mesh file. For in node primitives,
        //  it is the scene "primitiveId".
        std::map<std::string, std::shared_ptr<MeshFileI>> meshFiles;
        // Each mesh may have multiple submeshes so we don't wastefully open the same file
        // multiple times
        std::vector<Pair<uint32_t, const MeshFileI*>> batchFiles;

        public:
        PrimitiveLoader(TracerI& t, const std::string& sp,
                        std::shared_ptr<const MeshLoaderPoolI> mlp)
            : tracer(t)
            , scenePath(sp)
            , meshLoaderPool(mlp)
        {}

        PrimGroupId CreateGroup(std::string gn)
        {
            return tracer.CreatePrimitiveGroup(AddPrimitivePrefix(gn));
        }

        void CommitReservations(PrimGroupId groupId)
        {
            return tracer.CommitPrimReservations(groupId);
        }

        PrimBatchIdList THRDReserveEntities(PrimGroupId groupId,
                                            Span<const JsonNode> nodes)
        {
            batchFiles.reserve(nodes.size());
            PrimBatchIdList idList;
            idList.reserve(nodes.size());

            for(const JsonNode& node : nodes)
            {
                std::string tag = std::string(node.Tag());

                uint32_t innerIndex = 0;
                const MeshFileI* meshFile = nullptr;
                if(tag == NodeNames::NODE_PRIM_TRI ||
                   tag == NodeNames::NODE_PRIM_TRI_INDEXED)
                {
                    using namespace NodeNames;
                    bool isIndexed = (tag == NODE_PRIM_TRI_INDEXED);
                    auto meshFilePtr = std::make_unique<JsonTriangle>(node, isIndexed);
                    meshFile = meshFiles.emplace(std::to_string(node.Id()),
                                                 std::move(meshFilePtr)).first->second.get();
                }
                else if(tag == NodeNames::NODE_PRIM_SPHERE)
                {
                    using namespace NodeNames;
                    auto meshFilePtr = std::make_unique<JsonSphere>(node);
                    meshFile = meshFiles.emplace(std::to_string(node.Id()),
                                                 std::move(meshFilePtr)).first->second.get();
                }
                else
                {
                    std::string fileName = node.CommonData<std::string>(NodeNames::FILE);
                    fileName = Filesystem::RelativePathToAbsolute(fileName, scenePath);
                    innerIndex = node.AccessData<uint32_t>(NodeNames::INNER_INDEX);

                    // Find a Loader
                    auto loaderIt = loaders.emplace(tag, nullptr);
                    if(loaderIt.second)
                        loaderIt.first->second = meshLoaderPool->AcquireALoader(tag);
                    const auto& meshLoader = loaderIt.first->second;

                    // Find mesh file
                    // TODO: this is slow probably due to long file name as key
                    auto fileIt = meshFiles.emplace(fileName, nullptr);
                    if(fileIt.second) fileIt.first->second = meshLoader->OpenFile(fileName, innerIndex);
                    meshFile = fileIt.first->second.get();
                }
                // Finally Reserve primitives
                PrimCount pc
                {
                    .primCount = meshFile->MeshPrimitiveCount(),
                    .attributeCount = meshFile->MeshAttributeCount()
                };
                PrimBatchId tracerId = tracer.ReservePrimitiveBatch(groupId, pc);
                idList.push_back(tracerId);
                batchFiles.emplace_back(innerIndex, meshFile);
            }
            return idList;
        }

        void THRDLoadEntities(PrimGroupId groupId,
                              const PrimBatchIdList& ids,
                              Span<const JsonNode> nodes)
        {
            for(size_t i = 0; i < nodes.size(); i++)
            {
                const auto& [innerIndex, meshFile] = batchFiles[i];
                LoadPrimitive(tracer, groupId, ids[i],
                              innerIndex, meshFile);
            }
        }
    };

    GenericLoadGroups(primMappings, exceptions,
                      primNodes, threadPool,
                      PrimitiveLoader(tracer, scenePath,
                                      meshLoaderPool));
}

void SceneLoaderMRay::LoadCameras(TracerI& tracer, ExceptionList& exceptions)
{
    struct CameraLoader
    {
        private:
        TracerI&            tracer;
        AttributeCountList  totalCounts;

        public:
        CameraLoader(TracerI& t) : tracer(t) {}
        CameraGroupId CreateGroup(std::string gn)
        {
            return tracer.CreateCameraGroup(AddCameraPrefix(gn));
        }
        void CommitReservations(CameraGroupId groupId)
        {
            return tracer.CommitCamReservations(groupId);
        }
        CameraIdList THRDReserveEntities(CameraGroupId groupId,
                                         Span<const JsonNode> nodes)
        {
            std::vector<AttributeCountList> attributeCounts = {};
            attributeCounts.reserve(nodes.size());
            // Find the first arrayed type in nodes
            // Accumulate the type
            // Reserve transforms accordingly
            CamAttributeInfoList list = tracer.AttributeInfo(groupId);
            totalCounts = GenericFindAttributeCounts(attributeCounts, list, nodes);
            return tracer.ReserveCameras(groupId, attributeCounts);
        }
        void THRDLoadEntities(CameraGroupId groupId,
                              const CameraIdList& ids,
                              Span<const JsonNode> nodes)
        {
            CamAttributeInfoList list = tracer.AttributeInfo(groupId);
            auto dataOut = GenericAttributeLoad(totalCounts,
                                                list,
                                                nodes);
            uint32_t attribIndex = 0;
            for(auto& data : dataOut)
            {
                CameraId idStart = ids.front();
                CameraId idEnd = ids.front();
                auto range = Vector2ui(static_cast<uint32_t>(idStart),
                                       static_cast<uint32_t>(idEnd));
                tracer.PushCamAttribute(groupId, range, attribIndex,
                                        std::move(data));
                attribIndex++;
            }
        }
    };

    GenericLoadGroups(camMappings, exceptions,
                      cameraNodes, threadPool,
                      CameraLoader(tracer));
}

void SceneLoaderMRay::LoadLights(TracerI& tracer, ExceptionList& exceptions)
{
    struct LightLoader
    {
        private:
        TracerI&                    tracer;
        const TextureIdMappings&    texMappings;
        const PrimIdMappings&       primMappings;
        AttributeCountList          totalCounts;

        public:
        LightLoader(TracerI& t, const TextureIdMappings& texMappings,
                    const PrimIdMappings& primMappings)
            : tracer(t)
            , texMappings(texMappings)
            , primMappings(primMappings)
        {}

        LightGroupId CreateGroup(std::string gn)
        {
            gn = std::string(TracerConstants::LIGHT_PREFIX) + gn;
            return tracer.CreateLightGroup(std::move(gn));
        }

        void CommitReservations(LightGroupId groupId)
        {
            return tracer.CommitLightReservations(groupId);
        }

        LightIdList THRDReserveEntities(LightGroupId groupId,
                                        Span<const JsonNode> nodes)
        {
            std::vector<AttributeCountList> attributeCounts = {};
            attributeCounts.reserve(nodes.size());

            // If this light group is primitive-backed,
            // we require non-empty batch list, calculate it.
            std::vector<PrimBatchId> primBatches;

            // TODO: Quite ghetto way to find if this group is prim-backed
            // Change this
            using namespace NodeNames;
            auto tempId = nodes[0].AccessOptionalData<uint32_t>(PRIMITIVE);
            bool isPrimitiveBacked = tempId.has_value();

            if(isPrimitiveBacked)
            {
                primBatches.reserve(nodes.size());
                for(const auto& node : nodes)
                {
                    uint32_t primId = node.AccessData<uint32_t>(PRIMITIVE);
                    primBatches.push_back(primMappings.at(primId).second);
                }
            }

            // Find the first arrayed type in nodes
            // Accumulate the type
            // Reserve transforms accordingly
            LightAttributeInfoList list = tracer.AttributeInfo(groupId);
            totalCounts = GenericFindAttributeCounts(attributeCounts, list, nodes);
            return tracer.ReserveLights(groupId, std::move(attributeCounts),
                                        primBatches);
        }

        void THRDLoadEntities(LightGroupId groupId,
                              const LightIdList& ids,
                              Span<const JsonNode> nodes)
        {
            LightAttributeInfoList list = tracer.AttributeInfo(groupId);
            auto dataOut = TexturableAttributeLoad(totalCounts, list, nodes, texMappings);
            uint32_t attribIndex = 0;
            for(auto& data : dataOut)
            {
                LightId idStart = ids.front();
                LightId idEnd = ids.front();
                auto range = Vector2ui(static_cast<uint32_t>(idStart),
                                       static_cast<uint32_t>(idEnd));
                tracer.PushLightAttribute(groupId, range, attribIndex,
                                          std::move(data.data),
                                          std::move(data.textures));
                attribIndex++;
            }
        }
    };

    GenericLoadGroups(lightMappings, exceptions,
                      lightNodes, threadPool,
                      LightLoader(tracer, texMappings,
                                  primMappings.map));
}

void SceneLoaderMRay::CreateTypeMapping(const TracerI& tracer,
                                        const SceneSurfList& surfaces,
                                        const SceneCamSurfList& camSurfaces,
                                        const SceneLightSurfList& lightSurfaces,
                                        const LightSurfaceStruct& boundary)
{
    // Given N definition items, and M references on those items
    // where M >= N, create a map of common definitions -> referred definition list.

    // Definition items assumed to be random. Worst case this is O(N * M).
    // On a proper scene with transforms, M >> N. But N >> M can be possible if we
    // allow including multiple definition arrays (and some form of #include equavilence)
    //
    // This implementations assumes the first case is most common. So we create a HT
    // of N's and their locations. Then iterate over the M's and create the type
    constexpr uint32_t ARRAY_INDEX = 0;
    constexpr uint32_t INNER_INDEX = 1;
    using ItemLocation = Pair<uint32_t, uint32_t>;
    using ItemLocationMap = std::unordered_map<uint32_t, ItemLocation>;
    using AnnotationFunction = std::string(&)(std::string_view);


    auto CreateHT = [](ItemLocationMap& result,
                       const nlohmann::json& definitions) -> void
    {
        for(uint32_t i = 0; i < definitions.size(); i++)
        {
            const auto& node = definitions[i];
            const auto idNode = node.at(NodeNames::ID);
            ItemLocation itemLoc;
            std::get<ARRAY_INDEX>(itemLoc) = i;
            if(!idNode.is_array())
            {
                std::get<INNER_INDEX>(itemLoc) = 0;
                result.emplace(idNode.get<uint32_t>(), itemLoc);
            }
            else for(uint32_t j = 0; j < idNode.size(); j++)
            {
                const auto& id = idNode[j];
                std::get<INNER_INDEX>(itemLoc) = j;
                result.emplace(id.get<uint32_t>(), itemLoc);
            }
        }
    };

    using namespace TracerConstants;
    // Prims
    ItemLocationMap primHT;
    primHT.reserve(surfaces.size() * MaxPrimBatchPerSurface +
                   lightSurfaces.size());
    std::future<void> primHTReady = threadPool.submit_task(
    [&primHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        CreateHT(primHT, sceneJson.at(NodeNames::PRIMITIVE_LIST));
    });
    // Materials
    ItemLocationMap matHT;
    matHT.reserve(surfaces.size() * MaxPrimBatchPerSurface);
    std::future<void> matHTReady = threadPool.submit_task(
    [&matHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        CreateHT(matHT, sceneJson.at(NodeNames::MATERIAL_LIST));
    });
    // Cameras
    ItemLocationMap camHT;
    camHT.reserve(camSurfaces.size());
    std::future<void> camHTReady = threadPool.submit_task(
    [&camHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        CreateHT(camHT, sceneJson.at(NodeNames::CAMERA_LIST));
    });
    // Lights
    // +1 Comes from boundary light
    ItemLocationMap lightHT;
    lightHT.reserve(lightSurfaces.size() + 1);
    std::future<void> lightHTReady = threadPool.submit_task(
    [&lightHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        CreateHT(lightHT, sceneJson.at(NodeNames::LIGHT_LIST));
    });
    // Transforms
    ItemLocationMap transformHT;
    transformHT.reserve(lightSurfaces.size() +
                        surfaces.size() +
                        camSurfaces.size() + 1);
    std::future<void> transformHTReady = threadPool.submit_task(
    [&transformHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        CreateHT(transformHT, sceneJson.at(NodeNames::TRANSFORM_LIST));
    });

    // Mediums
    // Medium worst case is practically impossible
    // Each surface has max of 8 materials, each may require two
    // (inside/outside medium) + every (light + camera) surface
    // having unique medium (Utilizing arbitrary count of 512)
    // Worst case, we will have couple of rehashes nothing critical.
    ItemLocationMap mediumHT;
    mediumHT.reserve(512);
    std::future<void> mediumHTReady = threadPool.submit_task(
    [&mediumHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        return CreateHT(mediumHT, sceneJson.at(NodeNames::MEDIUM_LIST));
    });
    // Textures
    // It is hard to find estimate the worst case texture count as well.
    // Simple Heuristic: Each surface has unique material, each requiring
    // two textures, there are total of 16 mediums each require a single
    // texture
    ItemLocationMap textureHT;
    textureHT.reserve(surfaces.size() * MaxPrimBatchPerSurface * 2 +
                      16);
    std::future<void> textureHTReady = threadPool.submit_task(
    [&textureHT, CreateHT, &sceneJson = this->sceneJson]()
    {
        return CreateHT(textureHT, sceneJson.at(NodeNames::TEXTURE_LIST));
    });

    // Check boundary first
    auto PushToTypeMapping =
    [&sceneJson = std::as_const(sceneJson)](TypeMappedNodes& typeMappings,
                                            const ItemLocationMap& map, uint32_t id,
                                            const std::string_view& listName,
                                            //const AnnotationFunction& Annotate,
                                            bool skipUnknown = false)
    {
        const auto it = map.find(id);
        if(skipUnknown && it == map.end()) return;

        if(it == map.end())
            throw MRayError("Id({:d}) could not be "
                            "located in \"{:s}\"",
                            id, listName);
        const auto& location =  it->second;
        uint32_t arrayIndex = std::get<ARRAY_INDEX>(location);
        uint32_t innerIndex = std::get<INNER_INDEX>(location);

        auto node = JsonNode(sceneJson[listName][arrayIndex], innerIndex);
        //std::string type = Annotate(std::string(node.Type()));
        std::string type = std::string(node.Type());
        typeMappings[type].emplace_back(std::move(node));
    };

    // Start with boundary
    using namespace NodeNames;
    lightHTReady.wait();
    PushToTypeMapping(lightNodes, lightHT, boundary.lightId, LIGHT_LIST);
    mediumHTReady.wait();
    PushToTypeMapping(mediumNodes, mediumHT, boundary.mediumId, MEDIUM_LIST);
    transformHTReady.wait();
    PushToTypeMapping(transformNodes, transformHT, boundary.transformId, TRANSFORM_LIST);

    // Prim/Material Surfaces
    matHTReady.wait();
    primHTReady.wait();

    std::vector<NodeTexStruct> textureIds;
    textureIds.reserve(surfaces.size() * 2);
    for(const auto& s : surfaces)
    {
        for(uint8_t i = 0; i < s.pairCount; i++)
        {
            uint32_t matId = std::get<SurfaceStruct::MATERIAL_INDEX>(s.matPrimBatchPairs[i]);
            uint32_t primId = std::get<SurfaceStruct::PRIM_INDEX>(s.matPrimBatchPairs[i]);
            PushToTypeMapping(materialNodes, matHT, matId, MATERIAL_LIST);
            PushToTypeMapping(primNodes, primHT, primId, PRIMITIVE_LIST);
            PushToTypeMapping(transformNodes, transformHT,
                              s.transformId, TRANSFORM_LIST);

            if(s.alphaMaps[i].has_value())
                textureIds.push_back(s.alphaMaps[i].value());
        }
    }
    // Camera Surfaces
    camHTReady.wait();
    for(const auto& c : camSurfaces)
    {
        PushToTypeMapping(cameraNodes, camHT, c.cameraId, CAMERA_LIST);
        PushToTypeMapping(mediumNodes, mediumHT, c.mediumId, MEDIUM_LIST);
        PushToTypeMapping(transformNodes, transformHT,
                          c.transformId, TRANSFORM_LIST, true);
    }
    // Light Surfaces
    lightHTReady.wait();
    for(const auto& l : lightSurfaces)
    {
        // We could not use "PushToTypeMapping" here
        // lights are slightly different
        // If a light is primitive-backed
        // we need to push the type of the primitive
        const auto lightLoc = lightHT.find(l.lightId);

        if(lightLoc == lightHT.end())
            throw MRayError("Id({:d}) could not be "
                            "located in \"{:s}\"",
                            l.lightId, LIGHT_LIST);
        const auto& location = lightLoc->second;
        uint32_t arrayIndex = std::get<ARRAY_INDEX>(location);
        uint32_t innerIndex = std::get<INNER_INDEX>(location);

        auto node = JsonNode(sceneJson[LIGHT_LIST][arrayIndex], innerIndex);
        std::string_view lightTypeName = node.Type();


        std::string finalTypeName = std::string(lightTypeName);
        // Append prim type name if light is prim-backed
        if(lightTypeName == LIGHT_TYPE_PRIMITIVE)
        {
            uint32_t primId = node.AccessData<uint32_t>(PRIMITIVE);
            const auto primLoc = primHT.find(primId);
            if(primLoc == primHT.end())
                throw MRayError("Id({:d}) could not be "
                                "located in \"{:s}\". Requested by Light({:d})",
                                primId, PRIMITIVE_LIST, l.lightId);

            uint32_t primListIndex = std::get<ARRAY_INDEX>(primLoc->second);
            std::string_view primTypeName = sceneJson[PRIMITIVE_LIST]
                                                        [primListIndex]
                                                        [TYPE];
            finalTypeName = CreatePrimBackedLightType(primTypeName);
        }
        lightNodes[finalTypeName].emplace_back(std::move(node));


        PushToTypeMapping(mediumNodes, mediumHT, l.mediumId, MEDIUM_LIST);
        PushToTypeMapping(transformNodes, transformHT,
                          l.transformId, TRANSFORM_LIST, true);
    }

    // Now double indirections...
    // We need to iterate lights once to find primitive's that are required
    // by these lights. Lights are generic so we need look at the light via
    // the tracer. We call this dry run, since we do most of the scene-related
    // load work but we do not actually load the data.
    std::vector<uint32_t> primitiveIds;
    primitiveIds.reserve(lightSurfaces.size());
    DryRunLightsForPrim(primitiveIds, lightNodes, tracer);
    for(const auto& p : primitiveIds)
        PushToTypeMapping(primNodes, primHT, p, PRIMITIVE_LIST);

    // This is true for textures as well. Materials may/may not require textures
    // (or mediums) so we need to check these as well
    DryRunNodesForTex(textureIds, materialNodes, tracer,
                      &AddMaterialPrefix,
                      &TracerI::AttributeInfoMat);
    DryRunNodesForTex(textureIds, mediumNodes, tracer,
                      &AddMediumPrefix,
                      &TracerI::AttributeInfoMedium);

    // And finally, materials can define in/out mediums
    for(const auto& nodes : materialNodes)
    for(const auto& node : nodes.second)
    {
        auto optMedFront = node.AccessOptionalData<uint32_t>(MEDIUM_FRONT);
        auto optMedBack = node.AccessOptionalData<uint32_t>(MEDIUM_BACK);
        if(optMedFront.has_value())
            PushToTypeMapping(mediumNodes, mediumHT,
                              optMedFront.value(), MEDIUM_LIST);
        if(optMedBack.has_value())
            PushToTypeMapping(mediumNodes, mediumHT,
                              optMedBack.value(), MEDIUM_LIST);
    }

    // And finally create texture mappings
    textureHTReady.wait();
    for(const auto& t : textureIds)
    {
        const auto& it = textureHT.find(t.texId);
        if(it == textureHT.end())
            throw MRayError("Id({:d}) could not be "
                            "located in {:s}",
                            t.texId, TEXTURE_LIST);
        const auto& location = it->second;
        uint32_t arrayIndex = std::get<ARRAY_INDEX>(location);
        uint32_t innerIndex = std::get<INNER_INDEX>(location);
        auto node = JsonNode(sceneJson[TEXTURE_LIST][arrayIndex], innerIndex);
        // TODO: Add support for 3D texs.
        textureNodes.emplace_back(t, std::move(node), true);
    }

    // Eliminate the duplicates
    auto EliminateDuplicates = [](std::vector<JsonNode>& nodes)
    {
        std::sort(nodes.begin(), nodes.end());
        auto endIt = std::unique(nodes.begin(), nodes.end(),
                                 [](const JsonNode& n0,
                                    const JsonNode& n1)
        {
            return n0.Id() == n1.Id();
        });
        nodes.erase(endIt, nodes.end());
    };

    // TODO: Load balance here maybe?
    // Per-type per-array will be too fine grained?
    // (i.e., for cameras, probably a scene has at most 1-2 camera types 5-10 camera,
    // but a primitive may have thousands of primitives (not indiviual, as a batch))
    threadPool.detach_task([this, &EliminateDuplicates]()
    {
        for(auto& p : primNodes) EliminateDuplicates(p.second);
    });
    threadPool.detach_task([this, &EliminateDuplicates]()
    {
        for(auto& c : cameraNodes) EliminateDuplicates(c.second);
    });
    threadPool.detach_task([this, &EliminateDuplicates]()
    {
        for(auto& t : transformNodes) EliminateDuplicates(t.second);
    });
    threadPool.detach_task([this, &EliminateDuplicates]()
    {
        for(auto& l : lightNodes) EliminateDuplicates(l.second);
    });
    threadPool.detach_task([this, &EliminateDuplicates]()
    {
        for(auto& m : materialNodes) EliminateDuplicates(m.second);
    });
    threadPool.detach_task([this, &EliminateDuplicates]()
    {
        for(auto& m : mediumNodes) EliminateDuplicates(m.second);
    });
    threadPool.detach_task([this]()
    {
        auto LessThan = [](const auto& lhs, const auto& rhs)
        {
            return std::get<0>(lhs) < std::get<0>(rhs);
        };
        auto Equal = [](const auto& lhs, const auto& rhs)
        {
            return std::get<0>(lhs) == std::get<0>(rhs);
        };
        std::sort(textureNodes.begin(), textureNodes.end(), LessThan);
        auto last = std::unique(textureNodes.begin(), textureNodes.end(), Equal);
        textureNodes.erase(last, textureNodes.end());
    });

    threadPool.wait();
}

void SceneLoaderMRay::CreateSurfaces(TracerI& tracer, const std::vector<SurfaceStruct>& surfs)
{
    mRaySurfaces.reserve(surfs.size());
    for(const auto& surf : surfs)
    {
        TransformId transformId;
        SurfacePrimList primList;
        SurfaceMatList matList;
        OptionalAlphaMapList alphaMaps;
        CullBackfaceFlagList cullFace;

        transformId = transformMappings.map.at(surf.transformId).second;
        for(uint8_t i = 0; i < surf.pairCount; i++)
        {
            static constexpr size_t PI = SurfaceStruct::PRIM_INDEX;
            static constexpr size_t MI = SurfaceStruct::MATERIAL_INDEX;

            PrimBatchId pId = primMappings.map.at(std::get<PI>(surf.matPrimBatchPairs[i])).second;
            MaterialId mId = matMappings.map.at(std::get<MI>(surf.matPrimBatchPairs[i])).second;
            Optional<TextureId> tId;
            if(surf.alphaMaps[i].has_value())
                tId = texMappings.at(surf.alphaMaps[i].value());

            primList.push_back(pId);
            matList.push_back(mId);
            alphaMaps.push_back(tId);
            cullFace.push_back(surf.doCullBackFace[i]);
        }
        SurfaceId mRaySurf = tracer.CreateSurface(primList, matList,
                                                  transformId,
                                                  alphaMaps,
                                                  cullFace);
        mRaySurfaces.push_back(mRaySurf);
    }
}

void SceneLoaderMRay::CreateLightSurfaces(TracerI& tracer, const std::vector<LightSurfaceStruct>& surfs)
{
    mRayLightSurfaces.reserve(surfs.size());
    for(const auto& surf : surfs)
    {
        LightId lId = lightMappings.map.at(surf.lightId).second;
        MediumId mId = mediumMappings.map.at(surf.mediumId).second;
        TransformId tId = (surf.transformId == EMPTY_TRANSFORM)
                            ? TracerConstants::IdentityTransformId
                            : transformMappings.map.at(surf.transformId).second;

        LightSurfaceId mRaySurf = tracer.CreateLightSurface(lId, tId, mId);
        mRayLightSurfaces.push_back(mRaySurf);
    }
}

void SceneLoaderMRay::CreateCamSurfaces(TracerI& tracer, const std::vector<CameraSurfaceStruct>& surfs)
{
    mRayLightSurfaces.reserve(surfs.size());
    for(const auto& surf : surfs)
    {
        CameraId cId = camMappings.map.at(surf.cameraId).second;
        MediumId mId = mediumMappings.map.at(surf.mediumId).second;
        TransformId tId = (surf.transformId == EMPTY_TRANSFORM)
                            ? TracerConstants::IdentityTransformId
                            : transformMappings.map.at(surf.transformId).second;

        CamSurfaceId mRaySurf = tracer.CreateCameraSurface(cId, tId, mId);
        mRayCamSurfaces.push_back(mRaySurf);
    }
}

MRayError SceneLoaderMRay::LoadAll(TracerI& tracer)
{
    using Node = Optional<const nlohmann::json*>;
    auto FindNode = [this](std::string_view str) -> Node
    {
        const auto i = sceneJson.find(str);
        if(i == sceneJson.end()) return std::nullopt;
        return &(*i);
    };
    using namespace NodeNames;

    const nlohmann::json emptyJson;
    Node camSurfJson    = FindNode(CAMERA_SURFACE_LIST);
    Node lightSurfJson  = FindNode(LIGHT_SURFACE_LIST);
    Node surfJson       = FindNode(SURFACE_LIST);
    if(!camSurfJson.has_value())
        return MRayError("Scene file does not contain "
                         "\"{}\" array", CAMERA_SURFACE_LIST);
    if(!lightSurfJson.has_value())
        return MRayError("Scene file does not contain "
                         "\"{}\" array", LIGHT_SURFACE_LIST);
    if(!surfJson.has_value())
        return MRayError("Scene file does not contain "
                         "\"{}\" array", SURFACE_LIST);
    // Check the boundary light
    Node boundaryJson = FindNode(BOUNDARY);
    if(!boundaryJson.has_value())
        return MRayError("Scene file does not contain "
                         "\"{}\" object", BOUNDARY);

    // Now many things may fail (wrong name's, wrong types etc)
    // go full exceptions here (utilize the json as much as possible)
    // TODO: Change this to std::expected maybe c++23?
    try
    {
        LightSurfaceStruct boundary = LoadBoundary(*boundaryJson.value());
        SceneSurfList surfaces = LoadSurfaces(*surfJson.value());
        SceneCamSurfList camSurfs = LoadCamSurfaces(*camSurfJson.value(),
                                                    boundary.mediumId);
        SceneLightSurfList lightSurfs = LoadLightSurfaces(*lightSurfJson.value(),
                                                          boundary.mediumId);
        // Surfaces are loaded now create type/ node pairings
        // These are stored in the loader's state
        CreateTypeMapping(tracer, surfaces, camSurfs,
                          lightSurfs, boundary);

        // Multi-threaded section
        ExceptionList exceptionList;
        // Concat to a single exception and return it
        auto ConcatIfError = [&exceptionList]() -> MRayError
        {
            if(exceptionList.size != 0)
            {
                MRayError err("Errors from Threads:\n");
                size_t exceptionCount = exceptionList.size;
                for(size_t i = 0; i < exceptionCount; i++)
                {
                    using namespace std::literals;
                    const auto& e = exceptionList.exceptions[i];
                    err.AppendInfo(" "s + e.GetError() + "\n");
                }
                return err;
            }
            else return MRayError::OK;
        };

        // Many things depend on textures, so this is first
        // (currently only materials/alpha mapped surfaces,
        // and mediums)
        LoadTextures(tracer, exceptionList);
        // Technically, we should not wait here, only materials
        // and surfaces depend on textures.
        // We are waiting here for future proofness, in future
        // or a user may create a custom primitive type that holds
        // a texture etc.
        threadPool.wait();
        // We already bottlenecked ourselves here (by waiting),
        // might as well check if errors are occured and return early
        if(auto e = ConcatIfError(); e) return e;

        // Types that depend on textures
        LoadMediums(tracer, exceptionList);
        // Waiting here because Materials depend on mediums
        // In mray, materials seperate two mediums.
        threadPool.wait();
        // Same as above
        if(auto e = ConcatIfError(); e) return e;

        LoadMaterials(tracer, exceptionList, boundary.mediumId);
        // Does not depend on textures but may depend on later
        LoadTransforms(tracer, exceptionList);
        LoadPrimitives(tracer, exceptionList);
        LoadCameras(tracer, exceptionList);
        // Lights may depend on primitives (primitive-backed lights)
        // So we need to wait primitive id mappings to complete
        threadPool.wait();
        if(auto e = ConcatIfError(); e) return e;

        LoadLights(tracer, exceptionList);

        // Finally, wait all load operations to complete
        threadPool.wait();
        if(auto e = ConcatIfError(); e) return e;

        // Scene id -> tracer id mappings are created
        // and reside on the object's state.
        // Issue surface mappings to the tracer
        // Back to single thread here, a scene (even large)
        // probably consists of mid-thousands of surfaces.
        // Also this has a single bottleneck unlike tracer groups,
        // so it probably not worth it.
        CreateSurfaces(tracer, surfaces);
        CreateLightSurfaces(tracer, lightSurfs);
        CreateCamSurfaces(tracer, camSurfs);
    }
    // MRay related errros
    catch(const MRayError& e)
    {
        threadPool.purge();
        threadPool.wait();
        return e;
    }
    // Json related errors
    catch(const nlohmann::json::exception& e)
    {
        threadPool.purge();
        threadPool.wait();
        return MRayError("Json Error ({})", std::string(e.what()));
    }
    return MRayError::OK;
}

MRayError SceneLoaderMRay::OpenFile(const std::string& filePath)
{
    scenePath = std::filesystem::path(filePath).remove_filename().string();

    std::ifstream file(filePath);
    if(!file.is_open())
        return MRayError("Scene file \"{}\" is not found",
                         filePath);
    // Parse Json
    try
    {
        sceneJson = nlohmann::json::parse(file, nullptr, true, true);
    }
    catch(const nlohmann::json::parse_error& e)
    {
        return MRayError("Json Error ({})", std::string(e.what()));
    }
    return MRayError::OK;
}

MRayError SceneLoaderMRay::ReadStream(std::istream& sceneData)
{
    // For stream loads, we relate the path to cwd.
    scenePath = std::filesystem::current_path().string();

    // Parse Json
    try
    {
        sceneJson = nlohmann::json::parse(sceneData, nullptr, true, true);
    }
    catch(const nlohmann::json::parse_error& e)
    {
        return MRayError("Json Error ({})", std::string(e.what()));
    }
    return MRayError::OK;
}

TracerIdPack SceneLoaderMRay::MoveIdPack(double durationMS)
{
    return TracerIdPack
    {
        .prims = std::move(primMappings.map),
        .cams = std::move(camMappings.map),
        .lights = std::move(lightMappings.map),
        .transforms = std::move(transformMappings.map),
        .mats = std::move(matMappings.map),
        .mediums = std::move(mediumMappings.map),
        .surfaces = std::move(mRaySurfaces),
        .camSurfaces = std::move(mRayCamSurfaces),
        .lightSurfaces = std::move(mRayLightSurfaces),

        .loadTimeMS = durationMS
    };
}

void SceneLoaderMRay::ClearIntermediateBuffers()
{
    primNodes.clear();
    cameraNodes.clear();
    transformNodes.clear();
    lightNodes.clear();
    materialNodes.clear();
    mediumNodes.clear();
    textureNodes.clear();
}

SceneLoaderMRay::SceneLoaderMRay(BS::thread_pool& pool)
    :threadPool(pool)
{}

Expected<TracerIdPack> SceneLoaderMRay::LoadScene(TracerI& tracer,
                                                  const std::string& filePath)
{
    Timer t; t.Start();
    MRayError e = MRayError::OK;
    if(e = OpenFile(filePath)) return e;
    if(e = LoadAll(tracer)) return e;
    t.Split();

    ClearIntermediateBuffers();
    return MoveIdPack(t.Elapsed<Millisecond>());
}

Expected<TracerIdPack> SceneLoaderMRay::LoadScene(TracerI& tracer,
                                                  std::istream& sceneData)
{
    Timer t; t.Start();
    MRayError e = MRayError::OK;
    if(e = ReadStream(sceneData)) return e;
    if(e = LoadAll(tracer)) return e;
    t.Split();

    ClearIntermediateBuffers();
    return MoveIdPack(t.Elapsed<Millisecond>());
}

void SceneLoaderMRay::ClearScene()
{
    scenePath.clear();
    sceneJson.clear();
    transformMappings.map.clear();
    mediumMappings.map.clear();
    primMappings.map.clear();
    matMappings.map.clear();
    camMappings.map.clear();
    lightMappings.map.clear();
    texMappings.clear();
    mRaySurfaces.clear();
    mRayLightSurfaces.clear();
    mRayCamSurfaces.clear();
    primNodes.clear();
    cameraNodes.clear();
    transformNodes.clear();
    lightNodes.clear();
    materialNodes.clear();
    mediumNodes.clear();
    textureNodes.clear();
}

