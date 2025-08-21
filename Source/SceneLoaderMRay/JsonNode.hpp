#pragma once

#include "JsonNode.h"

inline bool IsDashed(const nlohmann::json& n)
{
    using namespace std::literals;
    static constexpr std::string_view DASH = "-"sv;
    return (n.is_string() && n.get<std::string_view>() == DASH);
}

inline void from_json(const nlohmann::json& n, SceneTexId& ts)
{
    using namespace NodeNames;
    ts = SceneTexId(n.at(TEXTURE_NAME).get<std::underlying_type_t<SceneTexId>>());
}

inline void from_json(const nlohmann::json& n, SurfaceStruct& s)
{
    auto it = n.find(NodeNames::TRANSFORM);
    s.transformId = (it == n.end())
                        ? EMPTY_TRANSFORM
                        : it->get<uint32_t>();

    const auto& matArray = n.at(NodeNames::MATERIAL);
    const auto& primArray = n.at(NodeNames::PRIMITIVE);
    if(matArray.size() != primArray.size())
        throw MRayError("Material/Primitive pair lists does not match on a surface!");

    // As a default we do back face culling (it is helpful for self occlusions)
    s.doCullBackFace.fill(true);
    // and no alpha maps (all prims are opaque)
    s.alphaMaps.fill(std::nullopt);

    if(matArray.is_number_integer() &&
       primArray.is_number_integer())
    {
        typename SurfaceStruct::IdPair p;
        get<SurfaceStruct::MATERIAL_INDEX>(p) = matArray;
        get<SurfaceStruct::PRIM_INDEX>(p) = primArray;
        s.pairCount = 1;
        s.matPrimBatchPairs[0] = p;

        auto alphaIt = n.find(NodeNames::ALPHA_MAP);
        if(alphaIt != n.cend())
            s.alphaMaps[0] = (*alphaIt).get<SceneTexId>();

        auto cullIt = n.find(NodeNames::CULL_FACE);
        if(cullIt != n.cend())
            s.doCullBackFace[0] = (*cullIt).get<bool>();
    }
    else
    {
        for(size_t i = 0; i < matArray.size(); i++)
        {
            typename SurfaceStruct::IdPair p;
            get<SurfaceStruct::MATERIAL_INDEX>(p) = matArray[i];
            get<SurfaceStruct::PRIM_INDEX>(p) = primArray[i];
            s.matPrimBatchPairs[i] = p;

            auto alphaIt = n.find(NodeNames::ALPHA_MAP);
            // Technically "-" should be supported only but
            // check if it is string here actual type is object.
            if(alphaIt != n.cend() && !(*alphaIt)[i].is_string())
                s.alphaMaps[i] = (*alphaIt)[i].get<SceneTexId>();

            auto cullIt = n.find(NodeNames::CULL_FACE);
            if(cullIt != n.cend() && !(*cullIt)[i].is_string())
                s.doCullBackFace[i] = (*cullIt)[i].get<bool>();

        }
        s.pairCount = static_cast<int8_t>(matArray.size());
    }
}

inline void from_json(const nlohmann::json& n, LightSurfaceStruct& s)
{
    auto itT = n.find(NodeNames::TRANSFORM);
    s.transformId = (itT == n.end())
                        ? EMPTY_TRANSFORM
                        : itT->get<uint32_t>();
    auto itM = n.find(NodeNames::MEDIUM);
    s.mediumId = (itM == n.end())
                    ? EMPTY_MEDIUM
                    : itM->get<uint32_t>();

    s.lightId = n.at(NodeNames::LIGHT);
}

inline void from_json(const nlohmann::json& n, CameraSurfaceStruct& s)
{
    auto itT = n.find(NodeNames::TRANSFORM);
    s.transformId = (itT == n.end())
                        ? EMPTY_TRANSFORM
                        : itT->get<uint32_t>();
    auto itM = n.find(NodeNames::MEDIUM);
    s.mediumId = (itM == n.end())
                    ? EMPTY_MEDIUM
                    : itM->get<uint32_t>();

    s.cameraId = n.at(NodeNames::CAMERA);
}

inline void from_json(const nlohmann::json& node, MRayTextureEdgeResolveEnum& t)
{
    auto name = node.get<std::string_view>();
    MRayTextureEdgeResolveEnum e = MRayTextureEdgeResolveStringifier::FromString(name);
    if(e == MRayTextureEdgeResolveEnum::MR_END)
        throw MRayError("Unknown edge resolve \"{}\"", name);
    t = e;
}

inline void from_json(const nlohmann::json& node, MRayTextureInterpEnum& t)
{
    auto name = node.get<std::string_view>();
    MRayTextureInterpEnum e = MRayTextureInterpStringifier::FromString(name);
    if(e == MRayTextureInterpEnum::MR_END)
        throw MRayError("Unknown texture interp \"{}\"", name);
    t = e;
}

inline void from_json(const nlohmann::json& node, MRayColorSpaceEnum& t)
{
    auto name = node.get<std::string_view>();
    MRayColorSpaceEnum e = MRayColorSpaceStringifier::FromString(name);
    if(e == MRayColorSpaceEnum::MR_END)
        throw MRayError("Unknown color space \"{}\"", name);
    t = e;
}

inline void from_json(const nlohmann::json& node, MRayTextureReadMode& t)
{
    auto name = node.get<std::string_view>();
    MRayTextureReadMode e = MRayTextureReadModeStringifier::FromString(name);
    // "drop" read modes are reserved for internal use
    if(e == MRayTextureReadMode::MR_END    ||
       e == MRayTextureReadMode::MR_DROP_1 ||
       e == MRayTextureReadMode::MR_DROP_2 ||
       e == MRayTextureReadMode::MR_DROP_3)
        throw MRayError("Unknown read mode \"{}\"", name);
    t = e;
}

inline void from_json(const nlohmann::json& node, ImageSubChannelType& t)
{
    using namespace std::literals;
    std::string_view l = node.get<std::string_view>();
         if(l == "r"sv) t = ImageSubChannelType::R;
    else if(l == "g"sv) t = ImageSubChannelType::G;
    else if(l == "b"sv) t = ImageSubChannelType::B;
    else if(l == "a"sv) t = ImageSubChannelType::A;
    //
    else if(l == "rg"sv) t = ImageSubChannelType::RG;
    else if(l == "gb"sv) t = ImageSubChannelType::GB;
    else if(l == "ba"sv) t = ImageSubChannelType::BA;
    //
    else if(l == "rgb"sv) t = ImageSubChannelType::RGB;
    else if(l == "gba"sv) t = ImageSubChannelType::GBA;
    //
    else if(l == "rgba"sv) t = ImageSubChannelType::RGBA;
    else throw MRayError("Unknown texture access layout");
}

inline JsonNode::JsonNode(const nlohmann::json& n, uint32_t innerIndex)
    : node(&n)
    , innerIndex(innerIndex)
{
    // Check the multi-nodeness
    auto inner = node->at(NodeNames::ID);
    isMultiNode = inner.is_array();
}

inline auto JsonNode::operator<=>(const JsonNode other) const
{
    return Id() <=> other.Id();
}

inline const nlohmann::json& JsonNode::RawNode() const
{
    return *node;
}

inline std::string_view JsonNode::Type() const
{
    return node->at(NodeNames::TYPE).get<std::string_view>();
}

inline std::string_view JsonNode::Tag() const
{
    return node->at(NodeNames::TAG).get<std::string_view>();
}

inline uint32_t JsonNode::Id() const
{
    const auto& nodeArray = (isMultiNode) ? node->at(NodeNames::ID).at(innerIndex)
                                          : node->at(NodeNames::ID);
    return nodeArray.get<uint32_t>();
}

template<class T>
T JsonNode::CommonData(std::string_view name) const
{
    return node->at(name).get<T>();
}

template<class T>
TransientData JsonNode::CommonDataArray(std::string_view name) const
{
    const auto& nodeArray = node->at(name);
    TransientData input(std::in_place_type_t<T>{}, nodeArray.size());

    for(size_t i = 0; i < nodeArray.size(); i++)
    {
        T val = nodeArray[i].get<T>();
        input.Push(Span<const T>(&val, 1));
    }
    return input;
}

inline size_t JsonNode::CheckDataArraySize(std::string_view name) const
{
    const auto& n = (isMultiNode) ? node->at(name).at(innerIndex)
                                  : node->at(name);
    return n.size();
}

inline size_t JsonNode::CheckOptionalDataArraySize(std::string_view name) const
{
    // Entire entry is missing (which is not defined all items on this node)
    if(node->find(name) == node->cend()) return 0;

    const auto& n = (isMultiNode) ? node->at(name).at(innerIndex)
                                  : node->at(name);
    return IsDashed(n) ? 0 : n.size();
}

inline bool JsonNode::CheckOptionalData(std::string_view name) const
{
    // Entire entry is missing (which is not defined all items on this node)
    if(node->find(name) == node->cend()) return false;

    const auto& n = (isMultiNode) ? node->at(name).at(innerIndex)
                                  : node->at(name);
    return IsDashed(n);
}

template<class T>
T JsonNode::AccessData(std::string_view name) const
{
    const auto& n = (isMultiNode) ? node->at(name).at(innerIndex)
                                  : node->at(name);
    return n.get<T>();
}

template<class T>
TransientData JsonNode::AccessDataArray(std::string_view name) const
{
    const auto& nodeArray = (isMultiNode) ? node->at(name).at(innerIndex)
                                          : node->at(name);
    TransientData input(std::in_place_type_t<T>{}, nodeArray.size());

    for(size_t i = 0; i < nodeArray.size(); i++)
    {
        T val = nodeArray[i].get<T>();
        input.Push(Span<const T>(&val, 1));
    }
    return input;
}
// Optional Data
template<class T>
Optional<T> JsonNode::AccessOptionalData(std::string_view name) const
{
    // Entire entry is missing (which is not defined all items on this node)
    if(node->find(name) == node->cend()) return std::nullopt;

    // Elements are available fetch the inner entry
    const auto& n = (isMultiNode) ? node->at(name).at(innerIndex)
                                  : node->at(name);

    // Here user may provide "-" string which means data is
    // not present for this item in node
    if(IsDashed(n)) return std::nullopt;
    else            return n.get<T>();
}

template<class T>
Optional<TransientData> JsonNode::AccessOptionalDataArray(std::string_view name) const
{
    // Entire entry is missing (which is not defined all items on this node)
    if(node->find(name) == node->cend()) return std::nullopt;

    // Elements are available fetch the inner entry
    const auto& nodeArray = (isMultiNode) ? node->at(name).at(innerIndex)
                                          : node->at(name);

    // Here user may provide "-" string which means data is
    // not present for this item in node
    TransientData input(std::in_place_type_t<T>{}, nodeArray.size());
    if(IsDashed(nodeArray))
        return std::nullopt;
    else
    {
        for(size_t i = 0; i < nodeArray.size(); i++)
        {
            T val = nodeArray[i].get<T>();
            input.Push(Span<const T>(&val, 1));
        }
        return input;
    }
}

// Texturable (either data T, or texture struct)
template<class T>
Variant<SceneTexId, T> JsonNode::AccessTexturableData(std::string_view name) const
{
    using V = Variant<SceneTexId, T>;
    const auto& n = (isMultiNode) ? node->at(name).at(innerIndex)
                                  : node->at(name);
    return (n.is_object()) ? V(n.get<SceneTexId>())
                           : V(n.get<T>());
}

inline SceneTexId JsonNode::AccessTexture(std::string_view name) const
{
    const auto& n = (isMultiNode) ? node->at(name).at(innerIndex)
                                  : node->at(name);
    return n.get<SceneTexId>();
}

inline Optional<SceneTexId> JsonNode::AccessOptionalTexture(std::string_view name) const
{
    // Entire entry is missing (which is not defined all items on this node)
    if(node->find(name) == node->cend()) return std::nullopt;

    const auto& n = (isMultiNode) ? node->at(name).at(innerIndex)
                                  : node->at(name);
    if(IsDashed(n)) return std::nullopt;
    else            return n.get<SceneTexId>();
}