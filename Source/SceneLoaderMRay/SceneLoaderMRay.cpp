#include "SceneLoaderMRay.h"

#include "Core/TracerI.h"
#include "Core/Log.h"
#include "Core/Timer.h"
#include "Core/Filesystem.h"
#include "Core/TypeNameGenerators.h"
#include "Core/GraphicsFunctions.h"
#include "Core/ThreadPool.h"
#include "Core/Profiling.h"

#include "ImageLoader/EntryPoint.h"
#include "MeshLoader/EntryPoint.h"
#include "MeshLoaderJson.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string_view>
#include <barrier>


using namespace TypeNameGen::Runtime;

struct TexturedAttributeData
{
    TransientData                       data;
    std::vector<Optional<TextureId>>    textures;
};

Expected<MRayTextureReadMode>
DetermineTextureReadMode(MRayTextureReadMode imageReadMode,
                         Optional<MRayTextureReadMode> userReadModeRequest,
                         const MRayPixelTypeRT& pixelType,
                         const std::string& filePath)
{
    if(!userReadModeRequest) return imageReadMode;

    using enum MRayTextureReadMode;
    const auto& userRM = *userReadModeRequest;
    // User don't care return the image's read mode
    if(userRM == MR_PASSTHROUGH) return imageReadMode;
    //
    else if(userRM == MR_AS_3C_TS_NORMAL_BASIC ||
            userRM == MR_AS_3C_TS_NORMAL_COOCTA)
    {
        if(pixelType.ChannelCount() != 2)
            return MRayError("A non 2-channel texture requested "
                             "to be converted to normal \"{}\"",
                             filePath);
        // Image wants to drop the channels,
        // but user wants to expand it back to 3-channel
        // This is not allowed
        if(imageReadMode == MR_DROP_1 || imageReadMode == MR_DROP_2)
            return MRayError("A compressed texture with pixel type {} requested to be read as "
                             "2 channels but also requested to be read as 3 channel normal. "
                             "This is not allowed. \"{}\"",
                             MRayPixelTypeStringifier::ToString(pixelType.Name()),
                             filePath);
        else return userRM;
    }
    else return imageReadMode;
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
            std::string_view name = l.name;
            AttributeOptionality optional = l.isOptional;
            AttributeIsArray isArray = l.isArray;

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
        list[i].dataType.SwitchCase([&](auto&& dataType)
        {
            using T = std::remove_cvref_t<decltype(dataType)>::Type;
            result.emplace_back(std::in_place_type_t<T>{}, totalCounts[i]);
        });
    }

    // Now data is set we can load
    for(const JsonNode& node : nodes)
    {
        uint32_t i = 0;
        for(const auto& l : list)
        {
            std::string_view name = l.name;
            AttributeOptionality optional = l.isOptional;
            AttributeIsArray isArray = l.isArray;

            l.dataType.SwitchCase([&](auto&& dataType)
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
            });
            i++;
        }
    }
    // clang evaluates the constexpr if
    // warns about unused local typedef.
    // So putting in macro
    #if MRAY_DEBUG
    {
        for(const auto& r : result)
            assert(r.IsFull());
    }
    #endif
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
        //auto texturability = std::get<TexturedAttributeInfo::TEXTURABLE_INDEX>(list[i]);
        auto texturability = list[i].isTexturable;
        list[i].dataType.SwitchCase([&](auto&& dataType)
        {
            using T = std::remove_cvref_t<decltype(dataType)>::Type;
            using enum AttributeTexturable;
            auto initData = TexturedAttributeData
            {
                .data = TransientData(std::in_place_type_t<T>{},
                                      (texturability == MR_TEXTURE_ONLY)
                                        ? 0
                                        : totalCounts[i]),
                .textures = std::vector<Optional<TextureId>>()
            };
            result.push_back(std::move(initData));
        });
    }

    // Now data is set we can load
    for(const JsonNode& node : nodes)
    {
        uint32_t i = 0;
        for(const auto& l : list)
        {
            std::string_view name = l.name;
            AttributeOptionality optional = l.isOptional;
            AttributeIsArray isArray = l.isArray;
            AttributeTexturable texturability = l.isTexturable;

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
                    SceneTexId texStruct = node.AccessTexture(name);
                    result[i].textures.push_back(texMappings.at(texStruct));
                }
                else if(optional == AttributeOptionality::MR_OPTIONAL)
                {
                    Optional<SceneTexId> texStruct = node.AccessOptionalTexture(name);
                    Optional<TextureId> id = (texStruct.has_value())
                                            ? Optional<TextureId>(texMappings.at(texStruct.value()))
                                            : std::nullopt;
                    result[i].textures.push_back(id);
                }
            }
            // Same as GenericAttributeInfo
            // TODO: Share functionality,  this is copy pase code
            if(texturability == AttributeTexturable::MR_CONSTANT_ONLY)
            {
                l.dataType.SwitchCase([&](auto&& dataType)
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
                });
                assert(result[i].data.IsFull());
            }
            //  Now the hairy part
            if(texturability == AttributeTexturable::MR_TEXTURE_OR_CONSTANT)
            {
                l.dataType.SwitchCase([&](auto&& dataType)
                {
                    using T = std::remove_cvref_t<decltype(dataType)>::Type;
                    Variant<SceneTexId, T> texturable = node.AccessTexturableData<T>(name);
                    if(std::holds_alternative<SceneTexId>(texturable))
                    {
                        TextureId id = texMappings.at(std::get<SceneTexId>(texturable));
                        result[i].textures.emplace_back(id);
                        T phony;
                        result[i].data.Push(Span<const T>(&phony, 1));
                    }
                    else
                    {
                        result[i].textures.emplace_back(std::nullopt);
                        result[i].data.Push(Span<const T>(&std::get<T>(texturable), 1));
                    }
                });
            }
            i++;
        }
    }

    // Sanity check
    // clang evaluates the constexpr if
    // warns about unused local typedef
    // so putting in macro
    #ifdef MRAY_DEBUG
    {
        uint32_t i = 0;
        for(const auto& l : list)
        {
            AttributeTexturable texturability = l.isTexturable;
            if(texturability == AttributeTexturable::MR_TEXTURE_ONLY)
            {
                assert(result[i].data.IsEmpty());
            }
            else if(texturability == AttributeTexturable::MR_TEXTURE_OR_CONSTANT ||
                    texturability == AttributeTexturable::MR_CONSTANT_ONLY)
            {
                assert(result[i].data.IsFull());
            }
            i++;
        }
    }
    #endif
    return result;
}

void LoadPrimitive(TracerI& tracer,
                   PrimGroupId groupId,
                   PrimBatchId batchId,
                   const MeshFileViewI* meshFileView)
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
    // This is all fun and games until this point.
    //
    // Some primitive groups may mandate some attributes, others may not.
    // Some files may have that attribute some may not.
    // In between this matrix of dependencies we can generate these from other attributes
    // (if available). All these things complicates the implementation.
    // assimp is good
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
        PrimitiveAttributeLogic attribLogic = attribute.logic;
        AttributeOptionality optionality = attribute.isOptional;
        MRayDataTypeRT groupsLayout = attribute.dataType;
        MRayDataTypeRT filesLayout = meshFileView->AttributeLayout(attribLogic);

        // Is this data available?
        if(!meshFileView->HasAttribute(attribLogic) &&
           optionality == AttributeOptionality::MR_MANDATORY)
        {
            throw MRayError("Mesh File{:s}:[{:d}] do not have \"{}\" "
                            "which is mandatory for {}",
                            meshFileView->Name(), meshFileView->InnerIndex(),
                            PrimAttributeStringifier::ToString(attribLogic),
                            tracer.TypeName(groupId));
        }
        // Data is available...
        // "Normal" attribute's special case, if mesh has tangents
        // Convert normal/tangent to quaternion, store it as normal
        // normals are defined as to tangent space transformations
        // (shading tangent space that is)
        if(attribLogic == NORMAL && groupsLayout.Name() == MR_QUATERNION &&
           meshFileView->HasAttribute(TANGENT) && meshFileView->HasAttribute(BITANGENT) &&
           meshFileView->AttributeLayout(TANGENT).Name() == MR_VECTOR_3 &&
           meshFileView->AttributeLayout(BITANGENT).Name() == MR_VECTOR_3 &&
           meshFileView->AttributeLayout(NORMAL).Name() == MR_VECTOR_3)
        {
            static const ProfilerAnnotation _("Prim Normal to Quat");
            auto annotation = _.AnnotateScope();

            size_t normalCount = meshFileView->MeshAttributeCount();
            TransientData quats(std::in_place_type_t<Quaternion>{}, normalCount);
            // Utilize TBN matrix directly
            TransientData tData = meshFileView->GetAttribute(TANGENT);
            TransientData bData = meshFileView->GetAttribute(BITANGENT);
            TransientData nData = meshFileView->GetAttribute(attribLogic);

            Span<const Vector3> tangents = tData.AccessAs<const Vector3>();
            Span<const Vector3> bitangents = bData.AccessAs<const Vector3>();
            Span<const Vector3> normals = nData.AccessAs<const Vector3>();

            for(size_t i = 0; i < normalCount; i++)
            {
                using Math::Normalize;
                Vector3 t = Normalize(tangents[i]);
                Vector3 b = Normalize(bitangents[i]);
                Vector3 n = Normalize(normals[i]);
                // If the tangents are left-handed,
                // convert them to right-handed
                if(Math::Dot(Math::Cross(t, b), n) < Float(0))
                    t = -t;
                auto [newT, newB] = Graphics::GSOrthonormalize(t, b, n);
                Quaternion q = TransformGen::ToSpaceQuat(newT, newB, n);
                quats.Push(Span<const Quaternion>(&q, 1));
            }
            assert(quats.IsFull());
            tracer.PushPrimAttribute(groupId, batchId, attribIndex,
                                     std::move(quats));
        }
        // All Good, load and send
        else if(groupsLayout.Name() == filesLayout.Name())
        {
            tracer.PushPrimAttribute(groupId, batchId, attribIndex,
                                     meshFileView->GetAttribute(attribLogic));

        }
        // Data's layout does not match with the primitive group
        else
        {
            // We require exact match currently
            throw MRayError("Mesh File {:s}:[{:d}]'s data layout of \"{}\" "
                            "(has type {:s}) does not match the {}'s data layout "
                            "(which is {:s})",
                            meshFileView->Name(), meshFileView->InnerIndex(),
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
    // system that sits between loader and tracer. (Which should be on GPU maybe)
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
        bool isArray = (info.isArray == AttributeIsArray::IS_ARRAY);
        std::string_view layout = n.CommonData<std::string_view>(LAYOUT);
        if(layout == LAYOUT_TRS)
        {
            static constexpr auto TRANSLATE = "translate"sv;
            static constexpr auto ROTATE = "rotate"sv;
            static constexpr auto SCALE = "scale"sv;

            auto GenTransformFromTRS = [](const Vector3& t,
                                          const Vector3& r,
                                          const Vector3& s)
            {
                Vector3 rRadians = r * MathConstants::DegToRadCoef<Float>();
                Matrix4x4 transform = TransformGen::Scale(s[0], s[1], s[2]);
                transform = TransformGen::Rotate(rRadians[0], Vector3::XAxis()) * transform;
                transform = TransformGen::Rotate(rRadians[1], Vector3::YAxis()) * transform;
                transform = TransformGen::Rotate(rRadians[2], Vector3::ZAxis()) * transform;
                transform = TransformGen::Translate(t) * transform;
                return transform;
            };

            if(isArray)
            {
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

    #ifdef MRAY_DEBUG
    {
        for(const auto& d : result)
            assert(d.IsFull());
    }
    #endif
    return result;
}

std::vector<TransientData> CameraAttributeLoad(const AttributeCountList& totalCounts,
                                               const GenericAttributeInfoList& list,
                                               Span<const JsonNode> nodes)
{
    using namespace std::literals;
    static constexpr auto PINHOLE_CAM_TYPE  = "Pinhole"sv;
    static constexpr auto IS_FOV_X_NAME     = "isFovX"sv;
    static constexpr auto FOV_NAME          = "fov"sv;
    static constexpr auto ASPECT_NAME       = "aspect"sv;
    static constexpr auto PLANES_NAME       = "planes"sv;
    static constexpr auto GAZE_NAME         = "gaze"sv;
    static constexpr auto POSITION_NAME     = "position"sv;
    static constexpr auto UP_NAME           = "up"sv;

    // TODO: Change this as well, camera logic should not be in loader
    // I think we need to layer these kind of things in an intermediate
    // system that sits between loader and tracer. (Which should be on GPU maybe)
    std::string_view type = nodes[0].CommonData<std::string_view>(NodeNames::TYPE);
    if(type != PINHOLE_CAM_TYPE)
        return GenericAttributeLoad(totalCounts, list, nodes);
    //
    assert(list.size() == 4);
    assert(totalCounts.size() == 4);
    std::vector<TransientData> result;
    result.push_back(TransientData(std::in_place_type_t<Vector4>{},
                                   totalCounts[0]));
    result.push_back(TransientData(std::in_place_type_t<Vector3>{},
                                   totalCounts[1]));
    result.push_back(TransientData(std::in_place_type_t<Vector3>{},
                                   totalCounts[2]));
    result.push_back(TransientData(std::in_place_type_t<Vector3>{},
                                   totalCounts[3]));

    for(const auto& n : nodes)
    {
        bool isFovX = n.AccessData<bool>(IS_FOV_X_NAME);
        Float fov = n.AccessData<Float>(FOV_NAME);
        Float aspect = n.AccessData<Float>(ASPECT_NAME);
        Vector2 planes = n.AccessData<Vector2>(PLANES_NAME);
        Vector4 fnp = Vector4(fov, fov, planes[0], planes[1]);
        if(isFovX)
        {
            fnp[0] *= MathConstants::DegToRadCoef<Float>();
            fnp[1] = Float(2.0) * std::atan(std::tan(fnp[0] * Float(0.5)) / aspect);
        }
        else
        {
            fnp[1] *= MathConstants::DegToRadCoef<Float>();
            fnp[0] = Float(2.0) * std::atan(std::tan(fnp[1] * Float(0.5)) * aspect);
        }
        result[0].Push(Span<const Vector4>(&fnp, 1));


        // Load these directly
        Vector3 gaze = n.AccessData<Vector3>(GAZE_NAME);
        Vector3 position = n.AccessData<Vector3>(POSITION_NAME);
        Vector3 up = n.AccessData<Vector3>(UP_NAME);
        result[1].Push(Span<const Vector3>(&gaze, 1));
        result[2].Push(Span<const Vector3>(&position, 1));
        result[3].Push(Span<const Vector3>(&up, 1));
    }

    #ifdef MRAY_DEBUG
    {
        for(const auto& d : result)
            assert(d.IsFull());
    }
    #endif
    return result;
}

LightSurfaceStruct SceneLoaderMRay::LoadBoundary(const nlohmann::json& n)
{
    LightSurfaceStruct boundary = n.get<LightSurfaceStruct>();
    if(boundary.lightId == std::numeric_limits<uint32_t>::max())
        throw MRayError("Boundary light must be set!");
    if(boundary.mediumId == std::numeric_limits<uint32_t>::max())
        throw MRayError("Boundary medium must be set!");
    if(boundary.transformId == std::numeric_limits<uint32_t>::max())
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
        std::string annotatedType = AddLightPrefix(l.first);
        LightAttributeInfoList lightAttributes = tracer.AttributeInfoLight(annotatedType);
        // We already annotated primitive-backed light names
        // so check "(L)Prim" prefix
        if(!IsPrimBackedLightType(annotatedType))
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
void SceneLoaderMRay::DryRunNodesForTex(std::vector<SceneTexId>& textureIds,
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
            AttributeTexturable texturable = att.isTexturable;
            AttributeOptionality optional = att.isOptional;
            std::string_view name = att.name;
            if(texturable == AttributeTexturable::MR_CONSTANT_ONLY)
                continue;

            if(texturable == AttributeTexturable::MR_TEXTURE_ONLY)
            {
                if(optional == AttributeOptionality::MR_OPTIONAL)
                {
                    auto ts = node.AccessOptionalData<SceneTexId>(name);
                    if(ts.has_value()) textureIds.push_back(ts.value());
                }
                else
                {
                    auto ts = node.AccessData<SceneTexId>(name);
                    textureIds.push_back(ts);
                }
            }
            else if(texturable == AttributeTexturable::MR_TEXTURE_OR_CONSTANT)
            {
                MRayDataTypeRT dataType = att.dataType;
                dataType.SwitchCase([&node, name, &textureIds](auto&& dataType)
                {
                    using T = std::remove_cvref_t<decltype(dataType)>::Type;
                    auto value = node.AccessTexturableData<T>(name);
                    if(std::holds_alternative<SceneTexId>(value))
                        textureIds.push_back(std::get<SceneTexId>(value));
                });
            }
        }
    }
}

template<bool FeedFirstNode, class Loader, class GroupIdType, class IdType>
void GenericLoadGroups(typename SceneLoaderMRay::MutexedMap<std::map<uint32_t, Pair<GroupIdType, IdType>>>& outputMappings,
                       ErrorList& exceptions,
                       const typename SceneLoaderMRay::TypeMappedNodes& nodeLists,
                       ThreadPool& threadPool,
                       Loader&& loader)
{
    using KeyValuePair  = Pair<uint32_t, Pair<GroupIdType, IdType>>;
    using PerGroupList  = std::vector<KeyValuePair>;
    using IdList        = std::vector<IdType>;

    for(const auto& [typeName, nodes] : nodeLists)
    {
        const uint32_t groupEntityCount = static_cast<uint32_t>(nodes.size());
        auto groupEntityList = std::make_shared<PerGroupList>(groupEntityCount);

        // TODO: Dirty fix to feed LightLoader the first node
        // so that it fetches primGroupId from it.
        // Change this later.
        GroupIdType groupId;
        if constexpr(FeedFirstNode)
            groupId = loader.CreateGroup(typeName, nodes.front());
        else
            groupId = loader.CreateGroup(typeName);

        // Construct Barrier
        auto BarrierFunc = [groupId, loader]() noexcept
        {
            // Explicitly copy the loader
            // Doing this because lambda capture trick
            // [loader = loader] did not work (maybe MSVC bug?)
            auto loaderIn = loader;

            // When barrier completed
            // Barrier function can not throw by design
            // So we need to catch any exceptions here
            // and delegate to the exception list.

            try
            {
                // Allocate the space for mappings
                // Commit group reservations
                loaderIn.CommitReservations(groupId);
            }
            // Only proper exception here is the out of memory by the GPU probably.
            // So abruptly terminate the process.
            catch(MRayError& e)
            {
                MRAY_ERROR_LOG("Fatal Error ({:s})", e.GetError());
                std::exit(1);
            }
            catch(std::exception& e)
            {
                MRAY_ERROR_LOG("Unknown Error ({:s})", e.what());
                std::exit(1);
            }
        };
        // Determine the thread size
        uint32_t threadCount = std::min(uint32_t(threadPool.ThreadCount()),
                                        groupEntityCount);

        using Barrier = std::barrier<decltype(BarrierFunc)>;
        auto barrier = std::make_shared<Barrier>(threadCount, BarrierFunc);

        // We are using our own thread pool now, this is no longer required
        // out thread pool copies/moves the lambda to a shared_ptr once
        // then each thread uses it etc.
        // ============= //
        //  OLD COMMENT  //
        // ============= //
        // BS_threadpool SubmitTask (all every submit variant I think, passes
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
            static const ProfilerAnnotation _(loader.Name());
            auto annotation = _.AnnotateScope();

            // Explicitly copy the loader
            // Doing this because lambda capture trick
            // [loader = loader] did not work (maybe MSVC bug?)
            auto loaderIn = loader;

            bool barrierPassed = false;
            size_t localCount = end - start;
            using ItDiff = decltype(nodes.cbegin())::difference_type;
            auto startConv = static_cast<ItDiff>(start);
            auto nodeRange = Span<const JsonNode>(nodes.cbegin() + startConv, localCount);
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
                    groupList[i] = Pair(nodes[i].Id(), value);
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
        auto future = threadPool.SubmitBlocks(groupEntityCount,
                                              LoadTask, threadCount);

        future.WaitAll();

        // Move future to shared_ptr
        using FutureSharedPtr = std::shared_ptr<MultiFuture<void>>;
        FutureSharedPtr futureShared = std::make_shared<MultiFuture<void>>(std::move(future));

        // Issue a one final task that pushes the primitives to the global map
        threadPool.SubmitDetachedTask([&, future = futureShared, groupEntityList]()
        {
            // Wait other tasks to complete
            future->WaitAll();
            // After this point groupBatchList is fully loaded
            std::scoped_lock lock(outputMappings.mutex);
            outputMappings.map.insert(groupEntityList->cbegin(), groupEntityList->cend());
        });
    }
}

void SceneLoaderMRay::LoadTextures(TracerI& tracer, ErrorList& exceptions)
{
    static const ProfilerAnnotation _("LoadTextures");
    auto annotation = _.AnnotateScope();

    using TextureIdList = std::vector<Pair<SceneTexId, TextureId>>;

    // Construct Image Loader
    std::shared_ptr<ImageLoaderI> imgLoader = CreateImageLoader();
    auto texIdListPtr = std::make_shared<TextureIdList>(textureNodes.size());

    // Issue loads to the thread pool
    auto BarrierFunc = [&tracer]() noexcept
    {
        // When barrier completed
        // Reserve the space for mappings
        // Commit textures group reservations
        try
        {
            tracer.CommitTextures();
        }
        // Only proper exception here is the out of memory by the GPU probably.
        // So abruptly terminate the process.
        catch(MRayError& e)
        {
            MRAY_ERROR_LOG("Fatal Error ({:s})", e.GetError());
            std::exit(1);
        }
        catch(std::exception& e)
        {
            MRAY_ERROR_LOG("Unknown Error ({:s})", e.what());
            std::exit(1);
        }
    };

    // Determine the thread size
    uint32_t threadCount = std::min(threadPool.ThreadCount(),
                                    uint32_t(textureNodes.size()));

    using Barrier = std::barrier<decltype(BarrierFunc)>;
    auto barrier = std::make_shared<Barrier>(threadCount, BarrierFunc);

    // Same situation discribed in 'GenericLoad' function,
    // force pass by copy.
    // Copy the shared pointers, capture by reference the rest
    const auto TextureLoadTask = [&, texIdListPtr, imgLoader, barrier](size_t start, size_t end)
    {
        static const ProfilerAnnotation ldTexAnnot("LoadTextures");
        auto annotation = ldTexAnnot.AnnotateScope();
        // TODO: check if the twice opening is a bottleneck?
        // We are opening here to determining size/format
        // and on the other iteration we actual memcpy it
        bool barrierPassed = false;
        std::vector<ImageFilePtr> localTexFiles;
        localTexFiles.reserve(end - start);
        try
        {
            for(size_t i = start; i < end; i++)
            {
                using namespace NodeNames;
                const auto& [sceneTexId, jsonNode] = textureNodes[i];
                auto fileName = jsonNode.AccessData<std::string>(TEX_NODE_FILE);
                auto isColor = jsonNode.AccessOptionalData<bool>(TEX_NODE_IS_COLOR)
                                .value_or(TEX_NODE_IS_COLOR_DEFAULT);
                bool loadAsSigned = jsonNode.AccessOptionalData<bool>(NodeNames::TEX_NODE_AS_SIGNED)
                                    .value_or(NodeNames::TEX_NODE_AS_SIGNED_DEFAULT);
                auto edgeResolve = jsonNode.AccessOptionalData<MRayTextureEdgeResolveEnum>(TEX_NODE_EDGE_RESOLVE);
                auto interp = jsonNode.AccessOptionalData<MRayTextureInterpEnum>(TEX_NODE_INTERPOLATION);
                auto colorSpace= jsonNode.AccessOptionalData<MRayColorSpaceEnum>(TEX_NODE_COLOR_SPACE);
                auto gamma = jsonNode.AccessOptionalData<Float>(TEX_NODE_GAMMA);
                auto userReadMode = jsonNode.AccessOptionalData<MRayTextureReadMode>(TEX_NODE_READ_MODE);
                //auto is3D = jsonNode.AccessOptionalData<bool>(TEX_NODE_IS_3D)
                //            .value_or(TEX_NODE_IS_3D_DEFAULT);
                auto channelLayout = jsonNode.AccessOptionalData<ImageSubChannelType>(TEX_NODE_CHANNELS)
                                                .value_or(ImageSubChannelType::ALL);
                auto ignoreResClamp = jsonNode.AccessOptionalData<bool>(TEX_NODE_IGNORE_CLAMP)
                                .value_or(TEX_NODE_IGNORE_CLAMP_DEFAULT);
                fileName = Filesystem::RelativePathToAbsolute(fileName, scenePath);

                using enum ImageIOFlags::F;
                ImageIOFlags flags;
                flags[DISREGARD_COLOR_SPACE] = !isColor;
                flags[LOAD_AS_SIGNED] = loadAsSigned;
                // Always do channel expand (HW limitation)
                flags[TRY_3C_4C_CONVERSION] = true;

                auto imgFileE = imgLoader->OpenFile(fileName,
                                                    channelLayout,
                                                    flags);
                if(imgFileE.has_error())
                {
                    exceptions.AddException(std::move(imgFileE.error()));
                    barrier->arrive_and_drop();
                    return;
                }
                localTexFiles.emplace_back(std::move(imgFileE.value()));

                Expected<ImageHeader> headerE = localTexFiles.back()->ReadHeader();
                if(!headerE.has_value())
                {
                    exceptions.AddException(std::move(headerE.error()));
                    barrier->arrive_and_drop();
                    return;
                }

                const auto& header = headerE.value();
                using enum AttributeIsColor;
                MRayTextureParameters params =
                {
                    .pixelType = header.pixelType,
                    .colorSpace = header.colorSpace.second,
                    .gamma = header.colorSpace.first,
                    .ignoreResClamp = ignoreResClamp
                };
                // Check and add user params
                if(edgeResolve.has_value()) params.edgeResolve = *edgeResolve;
                if(interp.has_value()) params.interpolation = *interp;
                // Overwrite color space related info, user has precedence.
                if(colorSpace.has_value()) params.colorSpace = *colorSpace;
                if(gamma.has_value()) params.gamma = *gamma;
                // Check user request of read mode
                // and the loader returned read mode to find the readmode
                auto readModeE = DetermineTextureReadMode(header.readMode, userReadMode,
                                                          header.pixelType, fileName);
                if(!readModeE.has_value())
                {
                    exceptions.AddException(std::move(headerE.error()));
                    barrier->arrive_and_drop();
                    return;
                }
                params.readMode = readModeE.value();

                TextureId tId;
                if(header.Is2D())
                {
                    tId = tracer.CreateTexture2D(Vector2ui(header.dimensions),
                                                 header.mipCount,
                                                 params);
                }
                else
                {
                    tId = tracer.CreateTexture3D(header.dimensions,
                                                 header.mipCount,
                                                 params);
                }

                auto& texIdList = *texIdListPtr;
                texIdList[i] = Pair(sceneTexId, tId);
            }

            // Barrier code is invoked, and all textures are allocated
            barrier->arrive_and_wait();
            barrierPassed = true;

            for(size_t i = start; i < end; i++)
            {
                Expected<Image> imgE = localTexFiles[i - start]->ReadImage();
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

    auto future = threadPool.SubmitBlocks(uint32_t(textureNodes.size()),
                                          TextureLoadTask, threadCount);

    // Move the future to shared ptr
    using FutureSharedPtr = std::shared_ptr<MultiFuture<void>>;
    FutureSharedPtr futureShared = std::make_shared<MultiFuture<void>>(std::move(future));

    // Issue a one final task that pushes the primitives to the global map
    threadPool.SubmitDetachedTask([&, this, future = futureShared, texIdListPtr]()
    {
        // Wait other tasks to complete
        future->WaitAll();

        // If no textures are loaded, commit the tracer
        // texture system. So it will let mapping to be accessed
        if(texIdListPtr->empty())
            tracer.CommitTextures();
        // Thread Generated Textures are finalized
        else for(const auto& pair : (*texIdListPtr))
            texMappings.emplace(pair.first, pair.second);
    });
    // All important data is in shared_ptrs we can safely exit scope.
}

void SceneLoaderMRay::LoadMediums(TracerI& tracer, ErrorList& exceptions)
{
    static const ProfilerAnnotation _("LoadMedia");
    auto annotation = _.AnnotateScope();

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

        std::string_view Name() const { return "LoadMedia"; }

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
            for(uint32_t attribIndex = 0;  attribIndex < dataOut.size();
                attribIndex++)
            {
                using enum AttributeTexturable;
                auto& data = dataOut[attribIndex];
                MediumId idStart = ids.front();
                MediumId idEnd = ids.back();
                auto range = CommonIdRange(std::bit_cast<CommonId>(idStart),
                                           std::bit_cast<CommonId>(idEnd));
                if(list[attribIndex].isTexturable == MR_CONSTANT_ONLY)
                    tracer.PushMediumAttribute(groupId, range, attribIndex,
                                               std::move(data.data));
                else
                    tracer.PushMediumAttribute(groupId, range, attribIndex,
                                               std::move(data.data),
                                               std::move(data.textures));
            }
        }
    };
    GenericLoadGroups<false>(mediumMappings, exceptions,
                             mediumNodes, threadPool,
                             MediumLoader(tracer, texMappings));
}

void SceneLoaderMRay::LoadMaterials(TracerI& tracer,
                                    ErrorList& exceptions,
                                    uint32_t boundaryMediumId)
{
    static const ProfilerAnnotation _("LoadMaterials");
    auto annotation = _.AnnotateScope();

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

        std::string_view Name() const { return "LoadMaterials"; }

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
            for(uint32_t attribIndex = 0; attribIndex < dataOut.size();
                attribIndex++)
            {
                using enum AttributeTexturable;
                auto& data = dataOut[attribIndex];
                MaterialId idStart = ids.front();
                MaterialId idEnd = ids.back();
                auto range = CommonIdRange(std::bit_cast<CommonId>(idStart),
                                           std::bit_cast<CommonId>(idEnd));
                if(list[attribIndex].isTexturable == MR_CONSTANT_ONLY)
                    tracer.PushMatAttribute(groupId, range, attribIndex,
                                            std::move(data.data));
                else
                    tracer.PushMatAttribute(groupId, range, attribIndex,
                                            std::move(data.data),
                                            std::move(data.textures));
            }
        }
    };

    GenericLoadGroups<false>(matMappings, exceptions,
                             materialNodes, threadPool,
                             MaterialLoader(tracer, boundaryMediumId,
                                            texMappings, mediumMappings.map));
}

void SceneLoaderMRay::LoadTransforms(TracerI& tracer, ErrorList& exceptions)
{
    static const ProfilerAnnotation _("LoadTransforms");
    auto annotation = _.AnnotateScope();
    struct TransformLoader
    {
        private:
        TracerI&            tracer;
        AttributeCountList  totalCounts;

        public:
        TransformLoader(TracerI& t) : tracer(t) {}

        std::string_view Name() const { return "LoadTransforms"; }

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
                TransformId idEnd = ids.back();
                auto range = CommonIdRange(std::bit_cast<CommonId>(idStart),
                                           std::bit_cast<CommonId>(idEnd));
                tracer.PushTransAttribute(groupId, range, attribIndex,
                                          std::move(data));
                attribIndex++;
            }
        }
    };

    GenericLoadGroups<false>(transformMappings, exceptions,
                             transformNodes, threadPool,
                             TransformLoader(tracer));
}

void SceneLoaderMRay::LoadPrimitives(TracerI& tracer, ErrorList& exceptions)
{
    static const ProfilerAnnotation _("LoadPrimitives");
    auto annotation = _.AnnotateScope();

    std::shared_ptr<const MeshLoaderPoolI> meshLoaderPool = CreateMeshLoaderPool();

    // Most of the shared pointers (except "meshLoaderPool")
    // are shared_pointer because this struct will be copied to each thread.
    // Default copy constructor of unique ptr does not exist so compiler does not
    // let us use unique_ptr. We do only copy the empty instantiations, so we do not even
    // properly use it. I could've mock create an empty copy constructor but
    // shared pointer is better.
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
        // form of header on the file. So we load the entire mesh and store it before the commit
        // of the primitive group.
        //
        // Key is the full path of the mesh file. For in node primitives,
        //  it is the scene "primitiveId".
        std::map<std::string, std::shared_ptr<MeshFileI>> meshFiles;
        // Each mesh may have multiple sub-meshes so we don't wastefully open the same file
        // multiple times
        std::vector<std::shared_ptr<MeshFileViewI>> meshViews;

        public:
        PrimitiveLoader(TracerI& t, const std::string& sp,
                        std::shared_ptr<const MeshLoaderPoolI> mlp)
            : tracer(t)
            , scenePath(sp)
            , meshLoaderPool(mlp)
        {}

        std::string_view Name() const { return "LoadPrimitives"; }

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
            meshViews.reserve(nodes.size());
            PrimBatchIdList idList;
            idList.reserve(nodes.size());

            for(const JsonNode& node : nodes)
            {
                std::string tag = std::string(node.Tag());

                //uint32_t innerIndex = 0;
                std::unique_ptr<MeshFileViewI> meshFileView;
                if(tag == NodeNames::NODE_PRIM_TRI ||
                   tag == NodeNames::NODE_PRIM_TRI_INDEXED)
                {
                    using namespace NodeNames;
                    bool isIndexed = (tag == NODE_PRIM_TRI_INDEXED);
                    meshFileView = std::make_unique<JsonTriangle>(node, isIndexed);
                }
                else if(tag == NodeNames::NODE_PRIM_SPHERE)
                {
                    using namespace NodeNames;
                    meshFileView = std::make_unique<JsonSphere>(node);
                }
                else
                {
                    std::string fileName = node.CommonData<std::string>(NodeNames::FILE);
                    fileName = Filesystem::RelativePathToAbsolute(fileName, scenePath);
                    uint32_t innerIndex = node.AccessData<uint32_t>(NodeNames::INNER_INDEX);

                    // Find a Loader
                    auto loaderIt = loaders.emplace(tag, nullptr);
                    if(loaderIt.second)
                        loaderIt.first->second = meshLoaderPool->AcquireALoader(tag);
                    const auto& meshLoader = loaderIt.first->second;

                    // Find mesh file
                    // TODO: this is slow probably due to long file name as key
                    auto fileIt = meshFiles.emplace(fileName, nullptr);
                    if(fileIt.second) fileIt.first->second = meshLoader->OpenFile(fileName);
                    auto meshFile = fileIt.first->second.get();

                    meshFileView = meshFile->ViewMesh(innerIndex);
                }
                // Finally Reserve primitives
                PrimCount pc
                {
                    .primCount = meshFileView->MeshPrimitiveCount(),
                    .attributeCount = meshFileView->MeshAttributeCount()
                };
                PrimBatchId tracerId = tracer.ReservePrimitiveBatch(groupId, pc);
                idList.push_back(tracerId);
                meshViews.emplace_back(std::move(meshFileView));
            }
            return idList;
        }

        void THRDLoadEntities(PrimGroupId groupId,
                              const PrimBatchIdList& ids,
                              Span<const JsonNode> nodes)
        {
            for(size_t i = 0; i < nodes.size(); i++)
            {
                LoadPrimitive(tracer, groupId, ids[i],
                              meshViews[i].get());
            }
        }
    };

    GenericLoadGroups<false>(primMappings, exceptions,
                             primNodes, threadPool,
                             PrimitiveLoader(tracer, scenePath,
                                             meshLoaderPool));
}

void SceneLoaderMRay::LoadCameras(TracerI& tracer, ErrorList& exceptions)
{
    static const ProfilerAnnotation _("LoadCameras");
    auto annotation = _.AnnotateScope();

    struct CameraLoader
    {
        private:
        TracerI&            tracer;
        AttributeCountList  totalCounts;

        public:
        CameraLoader(TracerI& t) : tracer(t) {}

        std::string_view Name() const { return "LoadCameras"; }

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
            auto dataOut = CameraAttributeLoad(totalCounts,
                                               list,
                                               nodes);
            uint32_t attribIndex = 0;
            for(auto& data : dataOut)
            {
                CameraId idStart = ids.front();
                CameraId idEnd = ids.back();
                auto range = CommonIdRange(std::bit_cast<CommonId>(idStart),
                                           std::bit_cast<CommonId>(idEnd));
                tracer.PushCamAttribute(groupId, range, attribIndex,
                                        std::move(data));
                attribIndex++;
            }
        }
    };

    GenericLoadGroups<false>(camMappings, exceptions,
                             cameraNodes, threadPool,
                             CameraLoader(tracer));
}

void SceneLoaderMRay::LoadLights(TracerI& tracer, ErrorList& exceptions)
{
    static const ProfilerAnnotation _("LoadLights");
    auto annotation = _.AnnotateScope();

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

        std::string_view Name() const { return "LoadLights"; }

        LightGroupId CreateGroup(std::string gn, const JsonNode& firstNode)
        {
            gn = std::string(TracerConstants::LIGHT_PREFIX) + gn;
            using namespace NodeNames;
            auto primId = firstNode.AccessOptionalData<uint32_t>(PRIMITIVE);
            if(primId.has_value())
            {
                PrimGroupId groupId = primMappings.at(primId.value()).first;
                return tracer.CreateLightGroup(std::move(gn), groupId);
            }
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
            for(uint32_t attribIndex = 0; attribIndex < dataOut.size();
                attribIndex++)
            {
                using enum AttributeTexturable;
                auto& data = dataOut[attribIndex];
                LightId idStart = ids.front();
                LightId idEnd = ids.back();
                auto range = CommonIdRange(std::bit_cast<CommonId>(idStart),
                                           std::bit_cast<CommonId>(idEnd));
                if(list[attribIndex].isTexturable == MR_CONSTANT_ONLY)
                    tracer.PushLightAttribute(groupId, range, attribIndex,
                                              std::move(data.data));
                else
                    tracer.PushLightAttribute(groupId, range, attribIndex,
                                              std::move(data.data),
                                              std::move(data.textures));
            }
        }
    };

    GenericLoadGroups<true>(lightMappings, exceptions,
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
    static const ProfilerAnnotation ctmAnnot("GenTypeMapping");
    auto annotation = ctmAnnot.AnnotateScope();
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

    auto CreateHT = [](ItemLocationMap& result,
                       const nlohmann::json& definitions) -> void
    {
        for(uint32_t i = 0; i < definitions.size(); i++)
        {
            const auto& node = definitions[i];
            const auto idNode = node.at(NodeNames::ID);
            ItemLocation itemLoc;
            get<ARRAY_INDEX>(itemLoc) = i;
            if(!idNode.is_array())
            {
                get<INNER_INDEX>(itemLoc) = 0;
                result.emplace(idNode.get<uint32_t>(), itemLoc);
            }
            else for(uint32_t j = 0; j < idNode.size(); j++)
            {
                const auto& id = idNode[j];
                get<INNER_INDEX>(itemLoc) = j;
                result.emplace(id.get<uint32_t>(), itemLoc);
            }
        }
    };

    using namespace TracerConstants;
    // Prims
    ItemLocationMap primHT;
    primHT.reserve(surfaces.size() * MaxPrimBatchPerSurface +
                   lightSurfaces.size());
    std::future<void> primHTReady = threadPool.SubmitTask(
    [&primHT, CreateHT, &sceneJsonIn = this->sceneJson]()
    {
        static const ProfilerAnnotation _("Prim HT Gen");
        auto annotation = _.AnnotateScope();
        CreateHT(primHT, sceneJsonIn.at(NodeNames::PRIMITIVE_LIST));
    });
    // Materials
    ItemLocationMap matHT;
    matHT.reserve(surfaces.size() * MaxPrimBatchPerSurface);
    std::future<void> matHTReady = threadPool.SubmitTask(
    [&matHT, CreateHT, &sceneJsonIn = this->sceneJson]()
    {
        static const ProfilerAnnotation _("Mat HT Gen");
        auto annotation = _.AnnotateScope();
        CreateHT(matHT, sceneJsonIn.at(NodeNames::MATERIAL_LIST));
    });
    // Cameras
    ItemLocationMap camHT;
    camHT.reserve(camSurfaces.size());
    std::future<void> camHTReady = threadPool.SubmitTask(
    [&camHT, CreateHT, &sceneJsonIn = this->sceneJson]()
    {
        static const ProfilerAnnotation _("Cam HT Gen");
        auto annotation = _.AnnotateScope();
        CreateHT(camHT, sceneJsonIn.at(NodeNames::CAMERA_LIST));
    });
    // Lights
    // +1 Comes from boundary light
    ItemLocationMap lightHT;
    lightHT.reserve(lightSurfaces.size() + 1);
    std::future<void> lightHTReady = threadPool.SubmitTask(
    [&lightHT, CreateHT, &sceneJsonIn = this->sceneJson]()
    {
        static const ProfilerAnnotation _("Light HT Gen");
        auto annotation = _.AnnotateScope();
        CreateHT(lightHT, sceneJsonIn.at(NodeNames::LIGHT_LIST));
    });
    // Transforms
    ItemLocationMap transformHT;
    transformHT.reserve(lightSurfaces.size() +
                        surfaces.size() +
                        camSurfaces.size() + 1);
    std::future<void> transformHTReady = threadPool.SubmitTask(
    [&transformHT, CreateHT, &sceneJsonIn = this->sceneJson]()
    {
        static const ProfilerAnnotation _("Trans HT Gen");
        auto annotation = _.AnnotateScope();
        CreateHT(transformHT, sceneJsonIn.at(NodeNames::TRANSFORM_LIST));
    });

    // Mediums
    // Medium worst case is practically impossible
    // Each surface has max of 8 materials, each may require two
    // (inside/outside medium) + every (light + camera) surface
    // having unique medium (Utilizing arbitrary count of 512)
    // Worst case, we will have couple of rehashes nothing critical.
    ItemLocationMap mediumHT;
    mediumHT.reserve(512);
    std::future<void> mediumHTReady = threadPool.SubmitTask(
    [&mediumHT, CreateHT, &sceneJsonIn = this->sceneJson]()
    {
        static const ProfilerAnnotation _("Media HT Gen");
        auto annotation = _.AnnotateScope();
        CreateHT(mediumHT, sceneJsonIn.at(NodeNames::MEDIUM_LIST));
    });
    // Textures
    // It is hard to find estimate the worst case texture count as well.
    // Simple Heuristic: Each surface has unique material, each requiring
    // two textures, there are total of 16 mediums each require a single
    // texture
    ItemLocationMap textureHT;
    textureHT.reserve(surfaces.size() * MaxPrimBatchPerSurface * 2 +
                      16);
    std::future<void> textureHTReady = threadPool.SubmitTask(
    [&textureHT, CreateHT, &sceneJsonIn = this->sceneJson]()
    {
        static const ProfilerAnnotation _("Texture HT Gen");
        auto annotation = _.AnnotateScope();
        CreateHT(textureHT, sceneJsonIn.at(NodeNames::TEXTURE_LIST));
    });

    // Check boundary first
    auto PushToTypeMapping =
    [&sceneJsonIn = std::as_const(sceneJson)](TypeMappedNodes& typeMappings,
                                              const ItemLocationMap& map, uint32_t id,
                                              const std::string_view& listName,
                                              bool skipUnknown = false)
    {
        const auto it = map.find(id);
        if(skipUnknown && it == map.end()) return;

        if(it == map.end())
            throw MRayError("Id({:d}) could not be "
                            "located in \"{:s}\"",
                            id, listName);
        const auto& location =  it->second;
        auto [arrayIndex, innerIndex] = location;
        auto node = JsonNode(sceneJsonIn[listName][arrayIndex], innerIndex);
        //std::string type = Annotate(std::string(node.Type()));
        std::string type = std::string(node.Type());
        typeMappings[type].emplace_back(std::move(node));
    };

    // Start with boundary
    using namespace NodeNames;
    lightHTReady.get();
    PushToTypeMapping(lightNodes, lightHT, boundary.lightId, LIGHT_LIST);
    mediumHTReady.get();
    PushToTypeMapping(mediumNodes, mediumHT, boundary.mediumId, MEDIUM_LIST);
    transformHTReady.get();
    PushToTypeMapping(transformNodes, transformHT, boundary.transformId, TRANSFORM_LIST);

    // Prim/Material Surfaces
    matHTReady.get();
    primHTReady.get();

    std::vector<SceneTexId> textureIds;
    textureIds.reserve(surfaces.size() * 2);
    for(const auto& s : surfaces)
    {
        for(uint8_t i = 0; i < s.pairCount; i++)
        {
            uint32_t matId = get<SurfaceStruct::MATERIAL_INDEX>(s.matPrimBatchPairs[i]);
            uint32_t primId = get<SurfaceStruct::PRIM_INDEX>(s.matPrimBatchPairs[i]);
            PushToTypeMapping(materialNodes, matHT, matId, MATERIAL_LIST);
            PushToTypeMapping(primNodes, primHT, primId, PRIMITIVE_LIST);
            PushToTypeMapping(transformNodes, transformHT,
                              s.transformId, TRANSFORM_LIST);

            if(s.alphaMaps[i].has_value())
                textureIds.push_back(s.alphaMaps[i].value());
        }
    }
    // Camera Surfaces
    camHTReady.get();
    for(const auto& c : camSurfaces)
    {
        PushToTypeMapping(cameraNodes, camHT, c.cameraId, CAMERA_LIST);
        PushToTypeMapping(mediumNodes, mediumHT, c.mediumId, MEDIUM_LIST);
        PushToTypeMapping(transformNodes, transformHT,
                          c.transformId, TRANSFORM_LIST, true);
    }
    // Light Surfaces
    // Already waited for light hash table
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
        uint32_t arrayIndex = get<ARRAY_INDEX>(location);
        uint32_t innerIndex = get<INNER_INDEX>(location);

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

            uint32_t primListIndex = get<ARRAY_INDEX>(primLoc->second);
            std::string_view primTypeName = sceneJson[PRIMITIVE_LIST]
                                                     [primListIndex]
                                                     [TYPE].get<std::string_view>();
            finalTypeName = CreatePrimBackedLightTypeName(primTypeName);
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
    DryRunNodesForTex(textureIds, lightNodes, tracer,
                      &AddLightPrefix,
                      &TracerI::AttributeInfoLight);

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
    textureHTReady.get();
    for(const auto& t : textureIds)
    {
        const auto& it = textureHT.find(uint32_t(t));
        if(it == textureHT.end())
            throw MRayError("Id({:d}) could not be "
                            "located in {:s}",
                            uint32_t(t), TEXTURE_LIST);
        const auto& location = it->second;
        uint32_t arrayIndex = get<ARRAY_INDEX>(location);
        uint32_t innerIndex = get<INNER_INDEX>(location);
        auto node = JsonNode(sceneJson[TEXTURE_LIST][arrayIndex], innerIndex);
        // TODO: Add support for 3D textures.
        textureNodes.emplace_back(t, std::move(node));
    }

    // Eliminate the duplicates
    auto EliminateDuplicates = [](std::vector<JsonNode>& nodes)
    {
        static const ProfilerAnnotation _("EliminateDuplicates");
        auto annotation = _.AnnotateScope();

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
    // but a primitive may have thousands of primitives (not individual, as a batch))
    threadPool.SubmitDetachedTask([this, &EliminateDuplicates]()
    {
        for(auto& p : primNodes) EliminateDuplicates(p.second);
    });
    threadPool.SubmitDetachedTask([this, &EliminateDuplicates]()
    {
        for(auto& c : cameraNodes) EliminateDuplicates(c.second);
    });
    threadPool.SubmitDetachedTask([this, &EliminateDuplicates]()
    {
        for(auto& t : transformNodes) EliminateDuplicates(t.second);
    });
    threadPool.SubmitDetachedTask([this, &EliminateDuplicates]()
    {
        for(auto& l : lightNodes) EliminateDuplicates(l.second);
    });
    threadPool.SubmitDetachedTask([this, &EliminateDuplicates]()
    {
        for(auto& m : materialNodes) EliminateDuplicates(m.second);
    });
    threadPool.SubmitDetachedTask([this, &EliminateDuplicates]()
    {
        for(auto& m : mediumNodes) EliminateDuplicates(m.second);
    });
    threadPool.SubmitDetachedTask([this]()
    {
        static const ProfilerAnnotation _("EliminateTexDuplicates");
        auto annotation = _.AnnotateScope();

        auto LessThan = [](const auto& lhs, const auto& rhs)
        {
            return get<0>(lhs) < get<0>(rhs);
        };
        auto Equal = [](const auto& lhs, const auto& rhs)
        {
            return get<0>(lhs) == get<0>(rhs);
        };
        std::sort(textureNodes.begin(), textureNodes.end(), LessThan);
        auto last = std::unique(textureNodes.begin(), textureNodes.end(), Equal);
        textureNodes.erase(last, textureNodes.end());
    });

    threadPool.Wait();
}

void SceneLoaderMRay::CreateSurfaces(TracerI& tracer, const std::vector<SurfaceStruct>& surfs)
{
    uint32_t surfaceId = 0;
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

            PrimBatchId pId = primMappings.map.at(get<PI>(surf.matPrimBatchPairs[i])).second;
            MaterialId mId = matMappings.map.at(get<MI>(surf.matPrimBatchPairs[i])).second;
            Optional<TextureId> tId;
            if(surf.alphaMaps[i].has_value())
                tId = texMappings.at(surf.alphaMaps[i].value());

            primList.push_back(pId);
            matList.push_back(mId);
            alphaMaps.push_back(tId);
            cullFace.push_back(surf.doCullBackFace[i]);
        }
        SurfaceParams surfParams
        {
            primList,
            matList,
            transformId,
            alphaMaps,
            cullFace
        };
        SurfaceId mRaySurf = tracer.CreateSurface(surfParams);
        mRaySurfaces.push_back(Pair(surfaceId++, mRaySurf));
    }
}

void SceneLoaderMRay::CreateLightSurfaces(TracerI& tracer, const std::vector<LightSurfaceStruct>& surfs,
                                          const LightSurfaceStruct& boundary)
{
    auto SurfStructToSurfParams = [this](const LightSurfaceStruct& surf)
    {
        LightId lId = lightMappings.map.at(surf.lightId).second;
        MediumId mId = mediumMappings.map.at(surf.mediumId).second;
        TransformId tId = (surf.transformId == EMPTY_TRANSFORM)
            ? TracerConstants::IdentityTransformId
            : transformMappings.map.at(surf.transformId).second;
        return LightSurfaceParams
        {
            lId, tId, mId
        };
    };
    uint32_t lightSurfaceId = 0;
    mRayLightSurfaces.reserve(surfs.size());
    for(const auto& surf : surfs)
    {

        LightSurfaceParams lSurfParams = SurfStructToSurfParams(surf);
        LightSurfaceId mRaySurf = tracer.CreateLightSurface(lSurfParams);
        mRayLightSurfaces.push_back(Pair(lightSurfaceId++, mRaySurf));
    }
    mRayBoundaryLightSurface = tracer.SetBoundarySurface(SurfStructToSurfParams(boundary));
}

void SceneLoaderMRay::CreateCamSurfaces(TracerI& tracer, const std::vector<CameraSurfaceStruct>& surfs)
{
    uint32_t camSurfaceId = 0;
    mRayLightSurfaces.reserve(surfs.size());
    for(const auto& surf : surfs)
    {
        CameraId cId = camMappings.map.at(surf.cameraId).second;
        MediumId mId = mediumMappings.map.at(surf.mediumId).second;
        TransformId tId = (surf.transformId == EMPTY_TRANSFORM)
                            ? TracerConstants::IdentityTransformId
                            : transformMappings.map.at(surf.transformId).second;

        CameraSurfaceParams cSurfParams
        {
            cId, tId, mId
        };
        CamSurfaceId mRaySurf = tracer.CreateCameraSurface(cSurfParams);
        mRayCamSurfaces.push_back(Pair(camSurfaceId++, mRaySurf));
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
        ErrorList exceptionList;
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
        threadPool.Wait();
        // We already bottlenecked ourselves here (by waiting),
        // might as well check if errors are occurred and return early
        if(auto e = ConcatIfError(); e) return e;

        // Types that depend on textures
        LoadMediums(tracer, exceptionList);
        // Waiting here because Materials depend on mediums
        // In mray, materials separate two mediums.
        threadPool.Wait();
        // Same as above
        if(auto e = ConcatIfError(); e) return e;

        LoadMaterials(tracer, exceptionList, boundary.mediumId);
        // Does not depend on textures but may depend on later
        LoadTransforms(tracer, exceptionList);
        LoadPrimitives(tracer, exceptionList);
        LoadCameras(tracer, exceptionList);
        // Lights may depend on primitives (primitive-backed lights)
        // So we need to wait primitive id mappings to complete
        threadPool.Wait();
        if(auto e = ConcatIfError(); e) return e;

        LoadLights(tracer, exceptionList);

        // Finally, wait all load operations to complete
        threadPool.Wait();
        if(auto e = ConcatIfError(); e) return e;

        // Scene id -> tracer id mappings are created
        // and reside on the object's state.
        // Issue surface mappings to the tracer
        // Back to single thread here, a scene (even large)
        // probably consists of mid-thousands of surfaces.
        // Also this has a single bottleneck unlike tracer groups,
        // so it probably not worth it.
        {
            static const ProfilerAnnotation _("Create Surfaces");
            auto annotation = _.AnnotateScope();

            CreateSurfaces(tracer, surfaces);
            CreateLightSurfaces(tracer, lightSurfs, boundary);
            CreateCamSurfaces(tracer, camSurfs);
        }
    }
    // MRay related errors
    catch(const MRayError& e)
    {
        threadPool.ClearTasks();
        threadPool.Wait();
        return e;
    }
    // Json related errors
    catch(const nlohmann::json::exception& e)
    {
        threadPool.ClearTasks();
        threadPool.Wait();
        return MRayError("Json Error ({})", std::string(e.what()));
    }

    threadPool.Wait();
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
        .concatStrings = std::vector<char>(),
        .stringRanges = std::vector<Vector2ul>(),
        .prims = std::move(primMappings.map),
        .cams = std::move(camMappings.map),
        .lights = std::move(lightMappings.map),
        .transforms = std::move(transformMappings.map),
        .mats = std::move(matMappings.map),
        .mediums = std::move(mediumMappings.map),
        .textures = std::move(texMappings),
        .surfaces = std::move(mRaySurfaces),
        .camSurfaces = std::move(mRayCamSurfaces),
        .lightSurfaces = std::move(mRayLightSurfaces),
        .boundarySurface = Pair(0u, mRayBoundaryLightSurface),
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

SceneLoaderMRay::SceneLoaderMRay(ThreadPool& pool)
    :threadPool(pool)
{}

Expected<TracerIdPack> SceneLoaderMRay::LoadScene(TracerI& tracer,
                                                  const std::string& filePath)
{
    static const ProfilerAnnotation _("Load Scene from File");
    auto annotation = _.AnnotateScope();

    Timer t; t.Start();
    MRayError e = MRayError::OK;
    if((e = OpenFile(filePath))) return e;
    if((e = LoadAll(tracer))) return e;
    t.Split();

    ClearIntermediateBuffers();
    return MoveIdPack(t.Elapsed<Millisecond>());
}

Expected<TracerIdPack> SceneLoaderMRay::LoadScene(TracerI& tracer,
                                                  std::istream& sceneData)
{
    static const ProfilerAnnotation _("Load Scene from Stream");
    auto annotation = _.AnnotateScope();

    Timer t; t.Start();
    MRayError e = MRayError::OK;
    if((e = ReadStream(sceneData))) return e;
    if((e = LoadAll(tracer))) return e;
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